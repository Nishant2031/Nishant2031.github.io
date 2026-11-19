"""
Independent numerical check of the cold-plate correlations (finite-volume, written from scratch).
  Part A: conjugate heat transfer in one half-channel unit cell
          - fully developed laminar velocity from a 2D Poisson solve
          - 3D temperature field in copper + coolant by implicit marching along the flow
  Part B: 3D conduction in the copper base (30 mm die footprint on a 60 mm plate)
This is NOT ANSYS; it is a second, independent solution of the same physics.
"""
import numpy as np, json, os, sys
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg, spsolve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))["selected"]
P, PLATE, DIE, T_IN = 1000.0, 0.060, 0.030, 35.0
K_CU, T_BASE = 390.0, 0.002
rho, cp, kf, mu = 1015.0, 3900.0, 0.50, 1.6e-3
wc, ww, H, N = R["w_c"]*1e-6, R["w_w"]*1e-6, R["H"]*1e-6, R["N"]
Q = R["Q_lpm"]/60000.0
V = Q/(N*wc*H); Dh = 2*wc*H/(wc+H); L = PLATE
qpp = P/PLATE**2

def lap2d(kx_face, ky_face, nx, ny):
    """Assemble 5-point conductance matrix. kx_face[j,i] couples (j,i)-(j,i+1); ky_face[j,i] couples (j,i)-(j+1,i)."""
    idx = np.arange(nx*ny).reshape(ny, nx)
    r = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    c = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    g = np.concatenate([kx_face.ravel(), ky_face.ravel()])
    A = sp.coo_matrix((np.concatenate([-g, -g]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(nx*ny,)*2).tocsr()
    d = -np.asarray(A.sum(axis=1)).ravel()
    return A + sp.diags(d)

def unit_cell(nxf, nxs, nyc, nyb, nz, tb=0.4e-3):
    """Half fin (nxs cells) + half channel (nxf cells); base slab tb below the channel floor."""
    dxs, dxf = (ww/2)/nxs, (wc/2)/nxf
    dx = np.r_[np.full(nxs, dxs), np.full(nxf, dxf)]
    dy = np.r_[np.full(nyb, tb/nyb), np.full(nyc, H/nyc)]
    nx, ny = nxs+nxf, nyb+nyc
    fluid = np.zeros((ny, nx), bool); fluid[nyb:, nxs:] = True
    k = np.where(fluid, kf, K_CU)
    DX, DY = np.meshgrid(dx, dy)
    # ---- velocity: mu * lap(u) = dp/dz, no-slip on fin wall, floor and (adiabatic) cover
    uf = np.where(fluid, 1.0, 0.0)
    gx = np.where(fluid[:, :-1] & fluid[:, 1:], DY[:, :-1]/((DX[:, :-1]+DX[:, 1:])/2), 0.0)
    gy = np.where(fluid[:-1, :] & fluid[1:, :], DX[:-1, :]/((DY[:-1, :]+DY[1:, :])/2), 0.0)
    A = lap2d(gx, gy, nx, ny).tolil()
    wall = np.zeros((ny, nx))
    wall[nyb:, nxs] += DY[nyb:, nxs]/(dxf/2)             # fin wall
    wall[nyb, nxs:] += DX[nyb, nxs:]/(DY[nyb, nxs:]/2)   # floor
    wall[-1, nxs:] += DX[-1, nxs:]/(DY[-1, nxs:]/2)      # cover
    A = (A + sp.diags(np.where(fluid, wall, 1.0).ravel())).tocsr()
    u = spsolve(A.tocsc(), (uf*DX*DY).ravel()).reshape(ny, nx)*uf
    Af = (DX*DY)[fluid].sum(); um = (u*DX*DY).sum()/Af
    dpdz = mu*V/um                      # Pa/m
    u *= V/um
    fRe = dpdz*Dh**2/(2*mu*V)
    # ---- energy: harmonic-mean face conductances
    kx = DY[:, :-1]/(DX[:, :-1]/2/k[:, :-1] + DX[:, 1:]/2/k[:, 1:])
    ky = DX[:-1, :]/(DY[:-1, :]/2/k[:-1, :] + DY[1:, :]/2/k[1:, :])
    Kc = lap2d(kx, ky, nx, ny)
    dz = L/nz
    C = (rho*cp*u*DX*DY/dz).ravel()
    lu = splu((Kc + sp.diags(C)).tocsc())
    src = np.zeros((ny, nx)); src[0, :] = qpp*dx
    T = np.full(nx*ny, T_IN); Tb, Tfl, Tbase_bot, Tfields = [], [], [], []
    for _ in range(nz):
        T = lu.solve(src.ravel() + C*T)
        Tm = T.reshape(ny, nx)
        Tb.append((Tm*u*DX*DY).sum()/(V*Af))
        Tfl.append((Tm[nyb-1]*dx).sum()/dx.sum() + qpp*(dy[nyb-1]/2)/K_CU)   # fin-base plane, pitch average
        Tbase_bot.append((Tm[0]*dx).sum()/dx.sum())
    Tb, Tfl = np.array(Tb), np.array(Tfl)
    z = (np.arange(nz)+1)*dz
    dT = Tfl - Tb
    # equivalent Nusselt number using the same wetted area as the correlation (floor + two side walls)
    return dict(fRe=fRe, dP_fd=dpdz*L, dT_out=Tb[-1]-T_IN, R_conv=dT.mean()/P, R_conv_out=dT[-1]/P,
                z=z, dT=dT, Tb=Tb, Tfl=Tfl, Tm=Tm, dx=dx, dy=dy, fluid=fluid, u=u, cells=nx*ny*nz)

def base_spread(n, nzb, h, die=DIE):
    """Quarter-symmetric 3D conduction: die flux on the bottom, lumped convection h on top. Returns K/W."""
    Lq = PLATE/2; d = Lq/n; dzb = T_BASE/nzb
    idx = np.arange(n*n*nzb).reshape(nzb, n, n)
    gxy, gz = K_CU*dzb, K_CU*d*d/dzb
    pairs = [(idx[:, :, :-1], idx[:, :, 1:], gxy), (idx[:, :-1, :], idx[:, 1:, :], gxy), (idx[:-1], idx[1:], gz)]
    r = np.concatenate([a.ravel() for a, _, _ in pairs]); c = np.concatenate([b.ravel() for _, b, _ in pairs])
    g = np.concatenate([np.full(a.size, w) for a, _, w in pairs])
    A = sp.coo_matrix((np.r_[-g, -g], (np.r_[r, c], np.r_[c, r])), shape=(idx.size,)*2).tocsr()
    diag = -np.asarray(A.sum(axis=1)).ravel()
    gtop = d*d/(dzb/2/K_CU + 1/h); diag[idx[-1].ravel()] += gtop
    A = A + sp.diags(diag)
    xc = (np.arange(n)+.5)*d; X, Y = np.meshgrid(xc, xc)
    m = (X < die/2) & (Y < die/2)
    b = np.zeros(idx.shape); b[0][m] = P/die**2*d*d
    T, info = cg(A, b.ravel(), rtol=1e-10, maxiter=20000, M=sp.diags(1/A.diagonal()))
    T = T.reshape(idx.shape)
    Tbot = T[0] + (b[0]/(d*d))*(dzb/2)/K_CU
    return Tbot[m].mean()/P, Tbot.max()/P, Tbot, m

if __name__ == "__main__":
    out = {}
    grids = {"coarse": (8, 5, 50, 8, 300), "medium": (15, 10, 100, 16, 600), "fine": (24, 16, 160, 24, 1200)}
    for name, g in grids.items():
        a = unit_cell(*g)
        out[name] = {k: float(a[k]) for k in ["fRe", "dP_fd", "dT_out", "R_conv", "R_conv_out", "cells"]}
        print(name, out[name]); sys.stdout.flush()
    fine = a
    h_num = 1/(fine["R_conv"]*PLATE**2); h_mod = 1/(R["R"]["conv"]*PLATE**2)
    for name, (n, nzb) in {"coarse": (60, 8), "fine": (120, 16)}.items():
        ra, rm, Tbot, m = base_spread(n, nzb, h_num)
        out["base_"+name] = dict(R_avg=float(ra), R_max=float(rm))
        print("base", name, ra, rm)
    ra_mod, rm_mod, _, _ = base_spread(120, 16, h_mod)
    out["base_with_model_h"] = dict(R_avg=float(ra_mod), R_max=float(rm_mod))
    out["h_num"], out["h_model"] = h_num, h_mod
    np.savez(os.path.join(HERE, "validation_fields.npz"), z=fine["z"], dT=fine["dT"], Tb=fine["Tb"], Tfl=fine["Tfl"],
             Tm=fine["Tm"], dx=fine["dx"], dy=fine["dy"], u=fine["u"], Tbot=Tbot)
    json.dump(out, open(os.path.join(HERE, "validation.json"), "w"), indent=2)
