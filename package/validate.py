"""
Independent checks of package.py.

1. Axisymmetric finite-element model (8-node quadrilaterals, written from scratch) of the
   same stack, linear elastic, cooled from the 125 C stress-free state to -40 C.
   Compared with the 1D shear-lag model run with an elastic solder bond:
     - warpage of the package bottom surface, centre to corner
     - shear strain in the C4 layer near the interposer edge
   The FE model keeps everything the 1D model throws away: through-thickness shear and
   normal stress in every layer, the bond's own stiffness and expansion, hoop stiffness,
   and the traction-free edge.

2. Anand integrator: at constant strain rate the stress must saturate at the analytical
   steady-state value  sigma* = (s*/xi) asinh[(eps_dot/A exp(Q/RT))^m],
   s* = s_hat (eps_dot/A exp(Q/RT))^n.

3. Time step, cycle count and grid convergence of the cyclic solution.
"""
import json
import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from package import (Package, section, make_grid, solve_history, warpage, run_case,
                     anand_update, sac_G, SI, EMC, SUB, UF, SAC_NU, KELVIN)


# ----------------------------------------------------------------------------------------
# 1. Axisymmetric Q8 finite elements
# ----------------------------------------------------------------------------------------
def graded(a, b, h_a, h_b, growth=1.15):
    """1D node list from a to b with spacing growing from h_a (at a) to at most h_b."""
    pts = [a]
    h = h_a
    while pts[-1] + h < b - 1e-12:
        pts.append(pts[-1] + h)
        h = min(h * growth, h_b)
    if b - pts[-1] < 0.3 * h and len(pts) > 1:
        pts[-1] = b
    else:
        pts.append(b)
    return np.array(pts)


def two_sided(a, b, h_edge, h_max, growth=1.15):
    """Fine at both ends, coarse in the middle."""
    mid = 0.5 * (a + b)
    left = graded(a, mid, h_edge, h_max, growth)
    right = b - graded(0, b - mid, h_edge, h_max, growth)[::-1]
    return np.unique(np.concatenate([left, right]))


def build_mesh(pkg, h_edge=0.01, growth=1.12, nz_scale=1):
    """Radial element boundaries: coarse under the die, refined toward the die edge and
    strongly refined (h_edge) at the interposer edge, where the bond shear peaks."""
    xd, L, Ls = pkg.x_die, pkg.L, pkg.L_sub
    r = np.concatenate([
        graded(0, xd, 2.0, 2.0)[:-1],
        xd + graded(0, L - xd - 3.0, 0.05, 1.5, growth)[:-1],
        L - graded(0, 3.0, h_edge, 0.4, growth)[::-1],
        L + graded(0, Ls - L, h_edge, 1.5, growth)])
    r = np.unique(np.round(r, 9))
    # through-thickness element boundaries
    zs = [0.0]
    layers = [("sub", pkg.t_sub, int(10 * nz_scale)), ("bond", pkg.h, int(6 * nz_scale)),
              ("int", pkg.t_int, int(3 * nz_scale)), ("top", pkg.t_die, int(8 * nz_scale))]
    for name, t, ne in layers:
        z0 = zs[-1]
        zs.extend(list(z0 + t * np.arange(1, ne + 1) / ne))
    z = np.array(zs)
    return r, z, layers


def material_at(pkg, name, rc):
    f = pkg.f
    if name == "sub":
        return dict(SUB, E=pkg.E_sub, alpha=pkg.alpha_sub)
    if name == "int":
        return SI
    if name == "top":
        return SI if rc <= pkg.x_die else EMC
    if name == "bond":
        Gs = float(sac_G(-40.0))
        Es = 2 * (1 + SAC_NU) * Gs
        Guf = pkg.E_uf / (2 * (1 + pkg.nu_uf)) if pkg.underfill else 0.0
        Gb = (1 - f) * Guf + f * Gs
        nu_b = 0.35
        Eb = 2 * (1 + nu_b) * Gb
        a_s, a_uf = 21.5e-6, UF["alpha"]
        Euf = pkg.E_uf if pkg.underfill else 0.0
        ab = (f * Es * a_s + (1 - f) * Euf * a_uf) / (f * Es + (1 - f) * Euf)
        return dict(E=Eb, nu=nu_b, alpha=ab)
    raise KeyError(name)


def Dmat(E, nu):
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((4, 4))
    D[:3, :3] = c * nu
    for i in range(3):
        D[i, i] = c * (1 - nu)
    D[3, 3] = E / (2 * (1 + nu))
    return D


# Q8 serendipity shape functions on [-1,1]^2; node order: corners (ccw) then mid-sides
XI = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], float)


def q8(xi, eta):
    N = np.zeros(8)
    dN = np.zeros((8, 2))
    for i in range(4):
        a, b = XI[i]
        N[i] = 0.25 * (1 + a * xi) * (1 + b * eta) * (a * xi + b * eta - 1)
        dN[i, 0] = 0.25 * a * (1 + b * eta) * (2 * a * xi + b * eta)
        dN[i, 1] = 0.25 * b * (1 + a * xi) * (a * xi + 2 * b * eta)
    for i in range(4, 8):
        a, b = XI[i]
        if a == 0:
            N[i] = 0.5 * (1 - xi ** 2) * (1 + b * eta)
            dN[i, 0] = -xi * (1 + b * eta)
            dN[i, 1] = 0.5 * b * (1 - xi ** 2)
        else:
            N[i] = 0.5 * (1 + a * xi) * (1 - eta ** 2)
            dN[i, 0] = 0.5 * a * (1 - eta ** 2)
            dN[i, 1] = -eta * (1 + a * xi)
    return N, dN


def fe_solve(pkg, dT, **mesh):
    r_b, z_b, layers = build_mesh(pkg, **mesh)
    nr, nz = len(r_b) - 1, len(z_b) - 1
    # layer of each element row
    row_layer = []
    for name, t, ne in layers:
        row_layer += [name] * ne
    # node grid: (2nr+1) x (2nz+1), with mid-side nodes; centre nodes unused
    R = np.zeros(2 * nr + 1)
    R[0::2] = r_b
    R[1::2] = 0.5 * (r_b[:-1] + r_b[1:])
    Z = np.zeros(2 * nz + 1)
    Z[0::2] = z_b
    Z[1::2] = 0.5 * (z_b[:-1] + z_b[1:])
    nid = -np.ones((2 * nr + 1, 2 * nz + 1), int)
    elems, mats = [], []
    for j in range(nz):
        name = row_layer[j]
        for i in range(nr):
            rc = 0.5 * (r_b[i] + r_b[i + 1])
            if name != "sub" and rc > pkg.L:
                continue
            I, J = 2 * i, 2 * j
            loc = [(I, J), (I + 2, J), (I + 2, J + 2), (I, J + 2),
                   (I + 1, J), (I + 2, J + 1), (I + 1, J + 2), (I, J + 1)]
            elems.append(loc)
            mats.append(material_at(pkg, name, rc))
    count = 0
    conn = np.zeros((len(elems), 8), int)
    for e, loc in enumerate(elems):
        for k, (a, b) in enumerate(loc):
            if nid[a, b] < 0:
                nid[a, b] = count
                count += 1
            conn[e, k] = nid[a, b]
    coords = np.zeros((count, 2))
    for a in range(2 * nr + 1):
        for b in range(2 * nz + 1):
            if nid[a, b] >= 0:
                coords[nid[a, b]] = (R[a], Z[b])
    ndof = 2 * count
    gp = np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)])
    gw = np.array([5 / 9, 8 / 9, 5 / 9])
    shp = [(q8(a, b), wa * wb) for a, wa in zip(gp, gw) for b, wb in zip(gp, gw)]
    rows, cols, vals = [], [], []
    F = np.zeros(ndof)
    for e in range(len(conn)):
        xy = coords[conn[e]]
        m = mats[e]
        D = Dmat(m["E"], m["nu"])
        eth = m["alpha"] * dT * np.array([1, 1, 1, 0.0])
        Ke = np.zeros((16, 16))
        fe = np.zeros(16)
        for (N, dN), w in shp:
            Jm = dN.T @ xy
            detJ = np.linalg.det(Jm)
            dNx = np.linalg.solve(Jm, dN.T).T      # d/dr, d/dz
            r = N @ xy[:, 0]
            B = np.zeros((4, 16))
            B[0, 0::2] = dNx[:, 0]
            B[1, 1::2] = dNx[:, 1]
            B[2, 0::2] = N / r
            B[3, 0::2] = dNx[:, 1]
            B[3, 1::2] = dNx[:, 0]
            dV = r * detJ * w
            Ke += B.T @ D @ B * dV
            fe += B.T @ D @ eth * dV
        dofs = np.empty(16, int)
        dofs[0::2] = 2 * conn[e]
        dofs[1::2] = 2 * conn[e] + 1
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(Ke.ravel())
        F[dofs] += fe
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ndof, ndof))
    # constraints: u_r = 0 on the axis, u_z = 0 at the bottom centre node
    fixed = list(2 * np.where(np.isclose(coords[:, 0], 0.0))[0])
    bc = np.where(np.isclose(coords[:, 0], 0.0) & np.isclose(coords[:, 1], 0.0))[0][0]
    fixed.append(2 * bc + 1)
    free = np.setdiff1d(np.arange(ndof), fixed)
    U = np.zeros(ndof)
    U[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
    return dict(U=U, coords=coords, conn=conn, nid=nid, R=R, Z=Z, z_b=z_b, layers=layers,
                ndof=ndof, n_el=len(conn))


def fe_outputs(pkg, sol):
    U, coords, nid, R, Z = sol["U"], sol["coords"], sol["nid"], sol["R"], sol["Z"]
    # bottom-surface warpage (relative to the centre)
    b = nid[:, 0]
    mask = b >= 0
    r_bot = R[mask]
    w_bot = U[2 * b[mask] + 1]
    w_bot = w_bot - w_bot[0]
    # bond layer: bottom and top faces
    t_sub, h = pkg.t_sub, pkg.h
    jb = int(np.argmin(np.abs(Z - t_sub)))
    jt = int(np.argmin(np.abs(Z - (t_sub + h))))
    jm = int(np.argmin(np.abs(Z - (t_sub + h / 2))))
    ok = (nid[:, jb] >= 0) & (nid[:, jt] >= 0) & (nid[:, jm] >= 0)
    rb = R[ok]
    ur_b = U[2 * nid[ok, jb]]
    ur_t = U[2 * nid[ok, jt]]
    uz_m = U[2 * nid[ok, jm] + 1]
    slope = np.gradient(uz_m, rb)
    gamma = (ur_t - ur_b) / h + slope          # thickness-averaged engineering shear strain
    return dict(r_bot=r_bot, w_bot=w_bot, r_bond=rb, gamma=gamma)


def oned_elastic(pkg, T_end):
    """1D model, elastic solder (no creep), one load step from T_ref to T_end."""
    x = make_grid(pkg, cluster=1.03)
    rec = solve_history(pkg, np.array([0.0, 1.0]), np.array([pkg.T_ref, T_end]), x=x, solder=False)
    dT = T_end - pkg.T_ref
    xw, w = warpage(pkg, x, rec["N"][-1], dT)
    return dict(x=x, gamma=rec["gamma"][-1], xw=xw, w=w)


# ----------------------------------------------------------------------------------------
# 2. Anand check
# ----------------------------------------------------------------------------------------
def anand_saturation_check(p, rates=(1e-5, 1e-4, 1e-3), temps=(-40.0, 25.0, 125.0)):
    out = []
    for T in temps:
        for rate in rates:
            # drive shear at gamma_dot = sqrt(3) * eps_dot until saturated
            gdot = math.sqrt(3) * rate
            gam_end = 0.3 if rate >= 1e-4 else 0.2
            nstep = 6000
            dt = gam_end / gdot / nstep
            gp = np.zeros(1); s = np.array([p["s0"]]); tau = np.zeros(1)
            for k in range(1, nstep + 1):
                tau, _, gp, s = anand_update(np.array([gdot * dt * k]), gp, s, T, dt, p)
            sig_num = math.sqrt(3) * tau[0]
            Z = rate / p["A"] * math.exp(p["Q_R"] / (T + KELVIN))
            s_star = p["s_hat"] * Z ** p["n"]
            sig_an = s_star / p["xi"] * math.asinh(Z ** p["m"])
            out.append(dict(T=T, rate=rate, sigma_num=sig_num, sigma_analytic=sig_an,
                            err_pct=100 * (sig_num - sig_an) / sig_an))
    return out


def main():
    res = {}
    pkg = Package()
    T_end = -40.0
    dT = T_end - pkg.T_ref

    t0 = time.time()
    sol = fe_solve(pkg, dT)
    fe = fe_outputs(pkg, sol)
    od = oned_elastic(pkg, T_end)
    print(f"FE: {sol['n_el']} elements, {sol['ndof']} dof, {time.time() - t0:.1f} s")

    # compare warpage at the substrate corner and at the interposer edge
    w_corner_fe = float(fe["w_bot"][-1])
    w_corner_1d = float(od["w"][-1])
    w_int_fe = float(np.interp(pkg.L, fe["r_bot"], fe["w_bot"]))
    w_int_1d = float(np.interp(pkg.L, od["xw"], od["w"]))
    x_c = pkg.L - pkg.pitch / 2
    g_fe = float(np.interp(x_c, fe["r_bond"], fe["gamma"]))
    g_1d = float(np.interp(x_c, od["x"], od["gamma"]))
    g_fe_max = float(np.max(np.abs(fe["gamma"])))
    r_at_max = float(fe["r_bond"][np.argmax(np.abs(fe["gamma"]))])
    g_1d_edge = float(od["gamma"][-1])
    g_mid_fe = float(np.interp(pkg.L - 1.0, fe["r_bond"], fe["gamma"]))
    g_mid_1d = float(np.interp(pkg.L - 1.0, od["x"], od["gamma"]))
    res["fe_vs_1d"] = dict(
        dT=dT, n_elements=sol["n_el"], n_dof=sol["ndof"],
        warp_corner_um=dict(fe=1e3 * w_corner_fe, model=1e3 * w_corner_1d,
                            diff_pct=100 * (w_corner_1d - w_corner_fe) / w_corner_fe),
        warp_interposer_edge_um=dict(fe=1e3 * w_int_fe, model=1e3 * w_int_1d,
                                     diff_pct=100 * (w_int_1d - w_int_fe) / w_int_fe),
        gamma_critical_bump=dict(x=x_c, fe=g_fe, model=g_1d, diff_pct=100 * (g_1d - g_fe) / g_fe),
        gamma_1mm_inboard=dict(fe=g_mid_fe, model=g_mid_1d, diff_pct=100 * (g_mid_1d - g_mid_fe) / g_mid_fe),
        gamma_peak=dict(fe=g_fe_max, fe_location_from_edge_um=1e3 * (pkg.L - r_at_max),
                        model_edge=g_1d_edge))
    print(json.dumps(res["fe_vs_1d"], indent=1))

    # mesh refinement of the FE check: half the edge spacing, slower growth, 2x layers
    t0 = time.time()
    sol2 = fe_solve(pkg, dT, h_edge=0.005, growth=1.06, nz_scale=2)
    fe2 = fe_outputs(pkg, sol2)
    g2 = float(np.interp(x_c, fe2["r_bond"], fe2["gamma"]))
    res["fe_mesh_refinement"] = dict(
        n_elements=sol2["n_el"], warp_corner_um=1e3 * float(fe2["w_bot"][-1]), gamma_critical_bump=g2,
        change_warp_pct=100 * (fe2["w_bot"][-1] - w_corner_fe) / w_corner_fe,
        change_gamma_pct=100 * (g2 - g_fe) / g_fe)
    print("refined FE", res["fe_mesh_refinement"], f"{time.time() - t0:.1f} s")

    # Anand saturation
    res["anand_saturation"] = anand_saturation_check(pkg.anand)
    worst = max(abs(r["err_pct"]) for r in res["anand_saturation"])
    print("Anand saturation worst error %.3f %%" % worst)

    # convergence of the cyclic solution
    conv = []
    for kw in [dict(dt=20, n_cycles=2), dict(dt=10, n_cycles=2), dict(dt=5, n_cycles=2),
               dict(dt=10, n_cycles=4)]:
        m, _ = run_case(pkg, **kw)
        conv.append(dict(kw, W=m["W"], eps_acc=m["eps_acc"], N_energy=m["N_energy"]))
    for c in (1.12, 1.03):
        m, _ = run_case(pkg, dt=10, n_cycles=2, x=make_grid(pkg, cluster=c))
        conv.append(dict(cluster=c, nodes=len(make_grid(pkg, cluster=c)), W=m["W"], N_energy=m["N_energy"]))
    res["convergence"] = conv
    print(json.dumps(conv, indent=1))

    # store profiles for plotting
    np.savez("validation_profiles.npz", r_bot=fe["r_bot"], w_bot=fe["w_bot"], r_bond=fe["r_bond"],
             gamma_fe=fe["gamma"], x1d=od["x"], gamma_1d=od["gamma"], xw1d=od["xw"], w1d=od["w"])
    with open("validation.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
