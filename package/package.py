"""
CoWoS-style package: warpage, C4 solder-joint fatigue and lifetime statistics.

Nishant Subedi, 2026.

What this models
----------------
A 30 x 30 mm, 1 kW logic die (the one from my cold plate project) sits on a 50 x 50 mm
silicon interposer, surrounded by mold compound. The interposer is joined to a 70 x 70 mm
organic substrate by C4 solder bumps (SAC305) with capillary underfill. Silicon expands at
about 2.6 ppm/K and the substrate at about 15 ppm/K, so every temperature change shears the
C4 layer and bends the whole stack.

Model (one dimension, along the package diagonal, from the centre to the corner):
  * Top adherend  = interposer + die (centre) or interposer + mold (outer ring).
  * Bottom adherend = organic substrate.
  * Bond layer    = C4 bumps (+ underfill), loaded in shear.
  Each adherend is a laminated plate with biaxial modulus E/(1-nu) (equal bending in both
  directions, as in a square package). The two adherends share one curvature. The bond
  layer transfers shear tau(x); the axial force N(x) it builds up bends the stack.
  This is the shear-lag / "bi-material assembly" model of Suhir, generalised to
  non-uniform adherends and a non-linear bond, and solved numerically.

  Governing equations (x from centre, 0 <= x <= L = interposer half-diagonal):
      dN/dx     = -tau(gamma)
      h dgamma/dx = F0(x) dT - lambda(x) N,   lambda = 1/A1 + 1/A2 + e^2/(D1 + D2)
      gamma(0) = 0 (symmetry),  N(L) = 0 (free edge)
  Curvature  kappa = (D1 kT1 + D2 kT2 - N e) / (D1 + D2);  warpage w'' = kappa.

  Bond shear stress: tau = (1-f) G_uf gamma + f tau_solder, where f is the bump area
  fraction and tau_solder follows the Anand viscoplastic model for SAC305 (creep and
  plasticity in one law), integrated implicitly at every node.

Life: per-cycle creep strain-energy density and accumulated creep strain at the corner bump
(Syed 2004 models for SnAgCu), plus Engelmaier's strain-range model for comparison.
Statistics: Monte Carlo over geometry/material scatter, a simulated 77-part test-to-failure,
censored Weibull fit, and Norris-Landzberg acceleration to a data-centre power cycle.

Units: mm, N, MPa, s, K (temperatures in degC where noted).
"""
import json
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn

KELVIN = 273.15

# ----------------------------------------------------------------------------------------
# Materials (linear elastic below Tg; MPa and 1/K)
# ----------------------------------------------------------------------------------------
SI = dict(E=130e3, nu=0.28, alpha=2.6e-6)        # E/(1-nu) = 180.6 GPa = (100) Si biaxial modulus
EMC = dict(E=20e3, nu=0.30, alpha=8.0e-6)        # mold compound, below Tg
SUB = dict(E=28e3, nu=0.30, alpha=15.0e-6)       # organic core + build-up substrate, in-plane effective
UF = dict(E=9e3, nu=0.33, alpha=28e-6)           # capillary underfill, below Tg

# SAC305 Anand constants, as tabulated by Eckermann et al., Microelectron. Reliab. 54 (2014) 1235
ANAND_SAC305 = dict(s0=2.150, Q_R=9970.0, A=17.994, xi=0.350, m=0.153,
                    h0=1525.98, s_hat=2.536, n=0.028, a=1.690)
SAC_NU = 0.36


def sac_E(T_C):
    """SAC305 Young's modulus [MPa], linear in temperature (about 50 GPa at 25 C)."""
    return 50e3 - 120.0 * (np.asarray(T_C) - 25.0)


def sac_G(T_C):
    return sac_E(T_C) / (2.0 * (1.0 + SAC_NU))


# ----------------------------------------------------------------------------------------
# Package definition
# ----------------------------------------------------------------------------------------
@dataclass
class Package:
    die: float = 30.0            # square die edge [mm]
    interposer: float = 50.0     # square interposer edge [mm]
    substrate: float = 70.0      # square substrate edge [mm]
    t_die: float = 0.775         # die / mold thickness [mm]
    t_int: float = 0.100         # interposer thickness [mm]
    t_sub: float = 2.000         # substrate thickness [mm]
    h: float = 0.080             # C4 standoff height [mm]
    pitch: float = 0.150         # C4 pitch [mm]
    d_bump: float = 0.090        # C4 diameter [mm]
    underfill: bool = True
    E_uf: float = UF["E"]
    nu_uf: float = UF["nu"]
    alpha_sub: float = SUB["alpha"]
    E_sub: float = SUB["E"]
    T_ref: float = 125.0         # stress-free temperature [C] (underfill Tg)
    anand: dict = field(default_factory=lambda: dict(ANAND_SAC305))

    @property
    def f(self):
        """Bump area fraction in the C4 layer."""
        return math.pi * (self.d_bump / 2) ** 2 / self.pitch ** 2

    # half-diagonals: distance from the centre to the corner along the diagonal
    @property
    def x_die(self):
        return self.die / math.sqrt(2)

    @property
    def L(self):
        return self.interposer / math.sqrt(2)

    @property
    def L_sub(self):
        return self.substrate / math.sqrt(2)


def laminate(layers):
    """layers: list of (thickness, material dict), bottom to top.
    Returns A, D, zNA (from bottom), NT/dT, MT/dT about NA (biaxial moduli)."""
    z = 0.0
    A = B = 0.0
    rows = []
    for t, m in layers:
        Eb = m["E"] / (1 - m["nu"])
        zc = z + t / 2
        A += Eb * t
        B += Eb * t * zc
        rows.append((t, Eb, m["alpha"], zc))
        z += t
    zna = B / A
    D = sum(Eb * (t ** 3 / 12 + t * (zc - zna) ** 2) for t, Eb, a, zc in rows)
    NT = sum(Eb * a * t for t, Eb, a, zc in rows)
    MT = sum(Eb * a * t * (zc - zna) for t, Eb, a, zc in rows)
    return dict(A=A, D=D, zna=zna, NT=NT, MT=MT, t=z)


def section(pkg: Package, x):
    """Coefficients of the shear-lag equations at position x (array)."""
    x = np.atleast_1d(x)
    sub_mat = dict(SUB, E=pkg.E_sub, alpha=pkg.alpha_sub)
    bot = laminate([(pkg.t_sub, sub_mat)])
    top_die = laminate([(pkg.t_int, SI), (pkg.t_die, SI)])
    top_emc = laminate([(pkg.t_int, SI), (pkg.t_die, EMC)])
    out = {}
    for key in ("A", "D", "zna", "NT", "MT"):
        out[key + "1"] = np.where(x <= pkg.x_die + 1e-12, top_die[key], top_emc[key])
    A1, D1, c1 = out["A1"], out["D1"], out["zna1"]
    A2, D2 = bot["A"], bot["D"]
    c2 = bot["t"] - bot["zna"]                 # substrate top surface to its NA
    e = c1 + pkg.h + c2                        # distance between neutral axes
    epsT1, epsT2 = out["NT1"] / A1, bot["NT"] / A2       # free strain per K
    kT1, kT2 = -out["MT1"] / D1, -bot["MT"] / D2          # free curvature per K
    Dsum = D1 + D2
    # Bond shear strain = slip / h + slope. With Kirchhoff kinematics in each adherend the
    # slip between the bond faces is (u01 - u02) + (c1 + c2) w', so gamma = (u01 - u02 + e w')/h.
    lam = 1 / A1 + 1 / A2 + e ** 2 / Dsum
    F0 = epsT1 - epsT2 + e * (D1 * kT1 + D2 * kT2) / Dsum
    return dict(A1=A1, A2=A2, D1=D1, D2=D2, c1=c1, c2=c2, e=e, lam=lam, F0=F0,
                kT1=kT1, kT2=kT2, Dsum=Dsum)


def make_grid(pkg: Package, n=90, cluster=1.06):
    """Nodes from centre to edge, geometrically clustered toward the free edge and the die edge."""
    # spacing grows away from the edge: dx_min ~ h/4
    dx0 = pkg.h / 4
    xs = [pkg.L]
    dx = dx0
    while xs[-1] - dx > 0:
        xs.append(xs[-1] - dx)
        dx *= cluster
    xs.append(0.0)
    x = np.array(sorted(set(np.round(xs, 9))))
    # make sure the die edge and the critical bump are nodes
    for xp in (pkg.x_die, pkg.L - pkg.pitch / 2):
        if not np.any(np.isclose(x, xp, atol=1e-6)):
            x = np.sort(np.append(x, xp))
    return x


# ----------------------------------------------------------------------------------------
# Anand viscoplasticity in simple shear, implicit (backward Euler) update
# ----------------------------------------------------------------------------------------
def anand_rate(tau, s, T_K, p):
    """Plastic shear strain rate for shear stress tau (vectorised)."""
    sig = math.sqrt(3.0) * np.abs(tau)                       # von Mises equivalent stress
    arg = np.clip(p["xi"] * sig / s, 0, 50)
    epsp = p["A"] * np.exp(-p["Q_R"] / T_K) * np.sinh(arg) ** (1.0 / p["m"])
    return math.sqrt(3.0) * epsp * np.sign(tau)               # gamma_dot_p


def anand_update(gamma_new, gp, s, T_C, dt, p):
    """Return tau, dtau/dgamma, gp_new, s_new for prescribed total shear strain."""
    T_K = T_C + KELVIN
    G = sac_G(T_C)
    tau_tr = G * (gamma_new - gp)
    # solve F(tau) = tau - tau_tr + G dt rate(tau) = 0; root lies between 0 and tau_tr
    sgn = np.sign(tau_tr)
    sgn[sgn == 0] = 1.0
    lo = np.zeros_like(tau_tr)
    hi = np.abs(tau_tr)
    t = hi.copy()
    C = p["A"] * np.exp(-p["Q_R"] / T_K) * math.sqrt(3.0)
    k = p["xi"] * math.sqrt(3.0) / s
    im = 1.0 / p["m"]
    for _ in range(60):
        arg = np.clip(k * t, 0, 50)
        sh = np.sinh(arg)
        r = C * sh ** im
        dr = C * im * sh ** (im - 1) * np.cosh(arg) * k
        F = t - np.abs(tau_tr) + G * dt * r
        lo = np.where(F < 0, t, lo)
        hi_b = np.where(F > 0, t, hi)
        dF = 1.0 + G * dt * dr
        t_new = t - F / dF
        bad = (t_new <= lo) | (t_new >= hi_b) | ~np.isfinite(t_new)
        t_new = np.where(bad, 0.5 * (lo + hi_b), t_new)
        hi = hi_b
        if np.all(np.abs(t_new - t) <= 1e-10 * (1 + np.abs(t))):
            t = t_new
            break
        t = t_new
    arg = np.clip(k * t, 0, 50)
    sh = np.sinh(arg)
    r = C * sh ** im
    dr = C * im * sh ** (im - 1) * np.cosh(arg) * k
    tau = sgn * t
    dgp = sgn * r * dt
    gp_new = gp + dgp
    tangent = G / (1.0 + G * dt * dr)
    # deformation resistance evolution (explicit, with the saturation value as a limit)
    epsp_rate = np.abs(dgp) / (math.sqrt(3.0) * dt)
    Z = np.maximum(epsp_rate, 1e-30) / p["A"] * np.exp(p["Q_R"] / T_K)
    s_star = p["s_hat"] * Z ** p["n"]
    B = 1.0 - s / s_star
    ds = p["h0"] * np.abs(B) ** p["a"] * np.sign(B) * epsp_rate * dt
    s_new = s + ds
    # do not overshoot the saturation value in one step
    s_new = np.where(B > 0, np.minimum(s_new, s_star), np.maximum(s_new, s_star))
    s_new = np.where(epsp_rate > 0, s_new, s)
    return tau, tangent, gp_new, s_new


# ----------------------------------------------------------------------------------------
# Temperature profiles
# ----------------------------------------------------------------------------------------
def profile_cycle(Tmin, Tmax, ramp_s, dwell_s, n_cycles, start=None, dt=5.0, dwell_cold_s=None):
    """Trapezoidal thermal cycle starting at `start` (default Tmax), going down first.
    dwell_s is the hot dwell; dwell_cold_s defaults to the same value."""
    if dwell_cold_s is None:
        dwell_cold_s = dwell_s
    if start is None:
        start = Tmax
    pts_t, pts_T = [0.0], [start]
    t = 0.0
    if start != Tmax:
        # preamble: cool (or heat) from the stress-free state to Tmax at the ramp rate
        t += abs(start - Tmax) / abs(Tmax - Tmin) * ramp_s
        pts_t.append(t)
        pts_T.append(Tmax)
    for _ in range(n_cycles):
        for T_target, dur in ((Tmin, ramp_s), (Tmin, dwell_cold_s), (Tmax, ramp_s), (Tmax, dwell_s)):
            t += dur
            pts_t.append(t)
            pts_T.append(T_target)
    times = np.arange(0.0, t + 1e-9, dt)
    temps = np.interp(times, pts_t, pts_T)
    period = 2 * ramp_s + dwell_s + dwell_cold_s
    return times, temps, period


# ----------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------
def solve_history(pkg: Package, times, temps, x=None, solder=True, record_every=1):
    """March the bond-layer problem through a temperature history.

    Returns a dict with gamma(x,t), N(x,t), tau_solder at the critical bump, hysteresis
    data, and warpage at each recorded step.
    """
    if x is None:
        x = make_grid(pkg)
    n = len(x)
    sec = section(pkg, x)
    dx = np.diff(x)
    p = pkg.anand
    f = pkg.f
    G_uf = pkg.E_uf / (2 * (1 + pkg.nu_uf)) if pkg.underfill else 0.0
    k_uf = (1 - f) * G_uf

    gp = np.zeros(n)
    s = np.full(n, p["s0"])
    gam = np.zeros(n)
    N = np.zeros(n)

    # critical bump: one half-pitch in from the interposer edge
    x_crit = pkg.L - pkg.pitch / 2
    ic = int(np.argmin(np.abs(x - x_crit)))  # exact node by construction

    rec = dict(t=[], T=[], gamma=[], N=[], tau_s_crit=[], gp_crit=[], gamma_crit=[],
               tau_bond_crit=[], warp=[])
    for step in range(1, len(times)):
        T = temps[step]
        dt = times[step] - times[step - 1]
        dT = T - pkg.T_ref
        F = sec["F0"] * dT
        lam = sec["lam"]
        g_try = gam.copy()
        N_try = N.copy()
        for it in range(30):
            if solder:
                tau_s, kt_s, gp_new, s_new = anand_update(g_try, gp, s, T, dt, p)
            else:
                tau_s, kt_s, gp_new, s_new = (sac_G(T) * g_try, np.full(n, sac_G(T)), gp, s)
            tau = k_uf * g_try + f * tau_s
            kt = k_uf + f * kt_s
            # residuals, unknown vector u = [g0, N0, g1, N1, ...]
            R = np.zeros(2 * n)
            R[0] = g_try[0]                                         # gamma(0) = 0
            R[1:2 * n - 1:2] = N_try[1:] - N_try[:-1] + 0.5 * dx * (tau[:-1] + tau[1:])
            R[2:2 * n:2] = pkg.h * (g_try[1:] - g_try[:-1]) - 0.5 * dx * (
                F[:-1] - lam[:-1] * N_try[:-1] + F[1:] - lam[1:] * N_try[1:])
            R[2 * n - 1] = N_try[-1]                                # N(L) = 0
            # reorder equations so the Jacobian is banded: rows grouped by interval
            # Build banded Jacobian (u = 2n unknowns), bandwidth: lower 2, upper 2
            # banded storage ab[2 + i - j, j] = J[i, j]
            J = np.zeros((5, 2 * n))
            kk = np.arange(n - 1)
            rN, rg = 1 + 2 * kk, 2 + 2 * kk          # force / compatibility rows
            gi, Ni, gj, Nj = 2 * kk, 2 * kk + 1, 2 * kk + 2, 2 * kk + 3
            J[2, 0] = 1.0
            J[2 + rN - Ni, Ni] = -1.0
            J[2 + rN - Nj, Nj] = 1.0
            J[2 + rN - gi, gi] = 0.5 * dx * kt[:-1]
            J[2 + rN - gj, gj] = 0.5 * dx * kt[1:]
            J[2 + rg - gi, gi] = -pkg.h
            J[2 + rg - gj, gj] = pkg.h
            J[2 + rg - Ni, Ni] = 0.5 * dx * lam[:-1]
            J[2 + rg - Nj, Nj] = 0.5 * dx * lam[1:]
            J[2, 2 * n - 1] = 1.0
            du = solve_banded((2, 2), J, -R)
            g_try = g_try + du[0::2]
            N_try = N_try + du[1::2]
            if np.max(np.abs(du[0::2])) < 1e-12 and np.max(np.abs(du[1::2])) < 1e-9:
                break
        # converged step: commit
        if solder:
            tau_s, kt_s, gp, s = anand_update(g_try, gp, s, T, dt, p)
        else:
            tau_s = sac_G(T) * g_try
        gam, N = g_try, N_try
        if step % record_every == 0 or step == len(times) - 1:
            rec["t"].append(times[step]); rec["T"].append(T)
            rec["gamma_crit"].append(gam[ic]); rec["gp_crit"].append(gp[ic])
            rec["tau_s_crit"].append(tau_s[ic])
            rec["tau_bond_crit"].append(k_uf * gam[ic] + f * tau_s[ic])
            rec["gamma"].append(gam.copy()); rec["N"].append(N.copy())
            rec["warp"].append(warpage(pkg, x, N, dT, sec)[1][-1])
    for k_ in rec:
        rec[k_] = np.array(rec[k_])
    rec["x"] = x
    rec["ic"] = ic
    return rec


def warpage(pkg: Package, x, N, dT, sec=None, n_over=40):
    """Integrate w'' = kappa from the centre to the substrate corner.
    Returns (x_all, w_all). Positive w = above the centre (corners up)."""
    if sec is None:
        sec = section(pkg, x)
    kap = (sec["D1"] * sec["kT1"] * dT + sec["D2"] * sec["kT2"] * dT - N * sec["e"]) / sec["Dsum"]
    # integrate twice with trapezoids
    slope = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(x) * (kap[:-1] + kap[1:]))])
    w = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(x) * (slope[:-1] + slope[1:]))])
    # substrate overhang: uniform substrate, free of load -> zero curvature, straight line
    xo = np.linspace(x[-1], pkg.L_sub, n_over)[1:]
    wo = w[-1] + slope[-1] * (xo - x[-1])
    return np.concatenate([x, xo]), np.concatenate([w, wo])


# ----------------------------------------------------------------------------------------
# Elastic closed form (constant coefficients) for checks
# ----------------------------------------------------------------------------------------
def elastic_closed_form(pkg: Package, dT, G_bond, x=None):
    """Suhir-type closed form using the coefficients of the die zone everywhere."""
    if x is None:
        x = make_grid(pkg)
    sec = section(pkg, np.array([0.0]))
    lam, F0 = sec["lam"][0], sec["F0"][0]
    k = math.sqrt(G_bond * lam / pkg.h)
    N = F0 * dT / lam * (1 - np.cosh(k * x) / np.cosh(k * pkg.L))
    gamma = F0 * dT * np.sinh(k * x) / np.cosh(k * pkg.L) / (k * pkg.h)
    return x, gamma, N, k


# ----------------------------------------------------------------------------------------
# Damage metrics and life models
# ----------------------------------------------------------------------------------------
def cycle_metrics(rec, period, dt):
    """Per-cycle metrics at the critical bump over the LAST full cycle."""
    t = rec["t"]
    sel = t > t[-1] - period - 1e-9
    tau = rec["tau_s_crit"][sel]
    gp = rec["gp_crit"][sel]
    g = rec["gamma_crit"][sel]
    dgp = np.diff(gp)
    tau_mid = 0.5 * (tau[1:] + tau[:-1])
    W = float(np.sum(tau_mid * dgp))                     # creep strain energy density [MPa]
    eps_acc = float(np.sum(np.abs(dgp)) / math.sqrt(3))  # accumulated equivalent creep strain
    d_gamma = float(g.max() - g.min())                   # total shear strain range
    tau_range = float(tau.max() - tau.min())
    return dict(W=W, eps_acc=eps_acc, d_gamma=d_gamma, tau_range=tau_range,
                tau_max=float(np.abs(tau).max()))


def life_syed(W, eps_acc):
    """Syed (2004) SnAgCu models: N = (0.00165 W)^-1 and N = (0.0513 eps_acc)^-1."""
    return dict(N_energy=1.0 / (0.00165 * W), N_strain=1.0 / (0.0513 * eps_acc))


def life_engelmaier(d_gamma, T_mean_C, dwell_min):
    """Engelmaier: N50 = 0.5 (d_gamma / (2 ef))^(1/c)."""
    ef = 0.325
    c = -0.442 - 6e-4 * T_mean_C + 1.74e-2 * math.log(1 + 360.0 / dwell_min)
    return 0.5 * (d_gamma / (2 * ef)) ** (1.0 / c)


def norris_landzberg_sac(dT_test, dT_field, f_test, f_field, Tmax_test_C, Tmax_field_C):
    """Pan et al. (2006) constants for SAC: AF = (dTt/dTf)^2.65 (ff/ft)^0.136 exp(2185(1/Tf - 1/Tt))."""
    return ((dT_test / dT_field) ** 2.65 * (f_field / f_test) ** 0.136 *
            math.exp(2185.0 * (1 / (Tmax_field_C + KELVIN) - 1 / (Tmax_test_C + KELVIN))))


def run_case(pkg: Package, Tmin=-40.0, Tmax=125.0, ramp_min=15, dwell_min=15, n_cycles=3,
             dt=5.0, x=None, dwell_cold_min=None):
    times, temps, period = profile_cycle(Tmin, Tmax, ramp_min * 60, dwell_min * 60, n_cycles,
                                         start=max(pkg.T_ref, Tmax), dt=dt,
                                         dwell_cold_s=None if dwell_cold_min is None else dwell_cold_min * 60)
    rec = solve_history(pkg, times, temps, x=x)
    m = cycle_metrics(rec, period, dt)
    life = life_syed(m["W"], m["eps_acc"])
    m.update(life)
    m["N_engelmaier"] = life_engelmaier(m["d_gamma"], 0.5 * (Tmin + Tmax), dwell_min)
    # warpage at the cold extreme and at 25 C (on the cool-down of the last cycle)
    T = rec["T"]
    i_cold = int(np.argmin(T))
    m["warp_cold_um"] = 1e3 * float(rec["warp"][i_cold])
    last = rec["t"] > rec["t"][-1] - period
    idx = np.where(last)[0]
    i25 = idx[np.argmin(np.abs(T[idx] - 25.0) + (np.gradient(T)[idx] > 0) * 1e3)]
    m["warp_25_um"] = 1e3 * float(rec["warp"][i25])
    return m, rec


# ----------------------------------------------------------------------------------------
# Weibull utilities
# ----------------------------------------------------------------------------------------
def weibull_mle(t, failed):
    """2-parameter Weibull MLE with right censoring. Returns beta, eta."""
    t = np.asarray(t, float)
    failed = np.asarray(failed, bool)

    def nll(p):
        b, ln_eta = p
        if b <= 0:
            return 1e12
        eta = math.exp(ln_eta)
        z = t / eta
        ll = np.sum(failed * (math.log(b) - math.log(eta) + (b - 1) * np.log(z))) - np.sum(z ** b)
        return -ll
    x0 = [3.0, math.log(np.median(t))]
    r = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
    return float(r.x[0]), float(math.exp(r.x[1]))


def weibull_quantile(beta, eta, q):
    return eta * (-math.log(1 - q)) ** (1 / beta)
