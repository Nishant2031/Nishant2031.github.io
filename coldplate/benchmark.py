"""
Benchmark of the cold-plate correlations against published EXPERIMENTAL data.

Reference: D. B. Tuckerman & R. F. W. Pease, "High-performance heat sinking for VLSI,"
IEEE Electron Device Letters 2(5), 126-129, 1981. Table I.
Three water-cooled silicon microchannel heat sinks, 1 x 1 cm, channels etched in silicon
with an anodically bonded glass cover (so the channel top is adiabatic).

Measured (Table I):
  #  wc[um] ww[um] depth[um]  dP[psi]  Q[cm3/s]  R_max[C/W]  q[W/cm2]
  1    56     44      320       15       4.7       0.110       181
  2    55     45      287       17       6.5       0.113       217
  3    50     50      302       31       8.6       0.090       790
R_max = (max substrate temperature - water inlet temperature) / total power, for a 1 cm^2 source.

The same correlation functions as coldplate.py are used here, with silicon instead of copper
and water instead of glycol. Nothing is tuned to the data.
"""
import numpy as np, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
K_SI = 148.0            # W/m-K, silicon at ~50 C
L = W = 0.010           # m, 1 x 1 cm heat sink
T_IN = 23.0             # C, T&P room-temperature water supply

# --- water properties, linear interpolation of standard table ---
_T = np.array([20, 30, 40, 50, 60, 70, 80, 90.])
_RHO = np.array([998.2, 995.6, 992.2, 988.0, 983.2, 977.8, 971.8, 965.3])
_CP = np.array([4182, 4178, 4179, 4181, 4185, 4190, 4197, 4205.])
_K = np.array([.598, .615, .631, .644, .654, .663, .670, .675])
_MU = np.array([1.002, .798, .653, .547, .467, .404, .355, .315])*1e-3
water = lambda T: dict(rho=np.interp(T, _T, _RHO), cp=np.interp(T, _T, _CP),
                       k=np.interp(T, _T, _K), mu=np.interp(T, _T, _MU))

# --- the same correlations used in coldplate.py ---
def fRe_rect(a):
    return 24*(1-1.3553*a+1.9467*a**2-1.7012*a**3+0.9564*a**4-0.2537*a**5)
def Nu_fd_rect(a):
    return 8.235*(1-2.0421*a+3.0853*a**2-2.4765*a**3+1.0578*a**4-0.1861*a**5)
def K_inf_rect(a):
    return 0.6796+1.2197*a+3.3089*a**2-9.5921*a**3+8.9089*a**4-2.9959*a**5

def predict(wc, ww, H, Q_cm3s, power, t_sub):
    """Return predicted pressure drop and thermal resistance for a T&P heat sink."""
    Q = Q_cm3s*1e-6
    N = int(W//(wc+ww))
    Tm = T_IN                                     # iterate on mean bulk temperature
    for _ in range(30):
        f = water(Tm)
        rho, cp, kf, mu = f["rho"], f["cp"], f["k"], f["mu"]
        mdot = rho*Q
        dT_fluid = power/(mdot*cp)
        Tm_new = T_IN + dT_fluid/2
        if abs(Tm_new-Tm) < 1e-6: Tm = Tm_new; break
        Tm = Tm_new
    V = Q/(N*wc*H); Dh = 2*wc*H/(wc+H)
    Re = rho*V*Dh/mu; Pr = mu*cp/kf
    a = min(wc, H)/max(wc, H)
    xs = L/(Dh*Re*Pr)
    Nu = Nu_fd_rect(a) + 0.024*xs**-1.14/(1+0.0358*Pr**0.17*xs**-0.64)
    h = Nu*kf/Dh
    m = np.sqrt(2*h/(K_SI*ww)); eta = np.tanh(m*H)/(m*H)
    A_eff = N*L*(wc + 2*H*eta)
    R_conv = 1/(h*A_eff)
    R_cal_max = 1/(mdot*cp)                       # outlet: R_max is referenced to the hottest point
    R_cond = t_sub/(K_SI*W*L)
    # hydraulics
    Lp = L/(Dh*Re)
    f_app = np.sqrt((3.44/np.sqrt(Lp))**2 + fRe_rect(a)**2)/Re
    dP_ch = (4*f_app*L/Dh)*0.5*rho*V**2
    sigma = wc/(wc+ww)
    dP_loss = (0.42*(1-sigma**2)+0.5 + (1-sigma)**2)*0.5*rho*V**2
    return dict(N=N, Re=float(Re), Pr=float(Pr), Nu=float(Nu), h=float(h), eta=float(eta), V=float(V),
                Tm=float(Tm), dT_fluid=float(dT_fluid),
                R_conv=float(R_conv), R_cal=float(R_cal_max), R_cond=float(R_cond),
                R_tot=float(R_conv+R_cal_max+R_cond),
                dP_ch=float(dP_ch), dP_tot=float(dP_ch+dP_loss))

CASES = [  # wc, ww, depth (um); dP (psi); Q (cm3/s); R_max (C/W); q (W/cm2)
    dict(n=1, wc=56, ww=44, H=320, dP_psi=15, Q=4.7, R=0.110, q=181),
    dict(n=2, wc=55, ww=45, H=287, dP_psi=17, Q=6.5, R=0.113, q=217),
    dict(n=3, wc=50, ww=50, H=302, dP_psi=31, Q=8.6, R=0.090, q=790),
]

if __name__ == "__main__":
    for t_sub in (100e-6,):
        print(f"\n=== substrate thickness below channels assumed {t_sub*1e6:.0f} um ===")
        rows = []
        for c in CASES:
            p = predict(c["wc"]*1e-6, c["ww"]*1e-6, c["H"]*1e-6, c["Q"], c["q"]*1e4*W*L, t_sub)
            dP_meas = c["dP_psi"]*6894.76
            rows.append(dict(case=c, pred=p, dP_meas=dP_meas,
                             dP_err=100*(p["dP_tot"]-dP_meas)/dP_meas,
                             R_err=100*(p["R_tot"]-c["R"])/c["R"]))
            print(f"#{c['n']}: Re={p['Re']:.0f} Nu={p['Nu']:.2f} h={p['h']:.0f} eta={p['eta']:.3f} "
                  f"| dP pred {p['dP_tot']/1e3:6.1f} kPa vs meas {dP_meas/1e3:6.1f} ({rows[-1]['dP_err']:+5.1f}%) "
                  f"| R pred {p['R_tot']*1e3:5.1f} mK/W vs meas {c['R']*1e3:5.1f} ({rows[-1]['R_err']:+5.1f}%)"
                  f" [conv {p['R_conv']*1e3:.1f} + fluid {p['R_cal']*1e3:.1f} + cond {p['R_cond']*1e3:.1f}]")
    # sensitivity of the R comparison to the one assumed quantity
    print("\nsubstrate thickness sensitivity (case 3):")
    for t in (50e-6, 100e-6, 150e-6):
        p = predict(50e-6, 50e-6, 302e-6, 8.6, 790., t)
        print(f"  {t*1e6:3.0f} um -> R_tot {p['R_tot']*1e3:.1f} mK/W ({100*(p['R_tot']-0.090)/0.090:+.1f}%)")
    json.dump([{k: v for k, v in r.items()} for r in rows],
              open(os.path.join(HERE, "benchmark.json"), "w"), indent=2, default=float)

# ---------------------------------------------------------------------------
# Why the two high-power designs deviate: constant properties.
# The correlations use bulk-temperature viscosity. When a liquid is heated hard,
# near-wall viscosity is much lower, which cuts friction (Deissler, laminar heating):
#     f / f_isothermal = (mu_bulk / mu_wall) ** -0.58
# Wall temperature is not published, so the hottest substrate temperature implied by
# the measured resistance is used as a BOUND, giving the largest possible correction.
# ---------------------------------------------------------------------------
def viscosity_bracket():
    rows = []
    for c in CASES:
        P = c["q"]*1e4*W*L
        p = predict(c["wc"]*1e-6, c["ww"]*1e-6, c["H"]*1e-6, c["Q"], P, 100e-6)
        T_wall_max = T_IN + c["R"]*P                 # hottest substrate, from measured R
        ratio = water(p["Tm"])["mu"]/water(T_wall_max)["mu"]
        dP_corr = p["dP_tot"]/ratio**0.58
        meas = c["dP_psi"]*6894.76
        rows.append(dict(n=c["n"], q=c["q"], T_wall=T_wall_max, mu_ratio=ratio,
                         dP_const=p["dP_tot"], dP_corr=dP_corr, dP_meas=meas,
                         e_const=100*(p["dP_tot"]-meas)/meas, e_corr=100*(dP_corr-meas)/meas))
    return rows

if __name__ == "__main__":
    print("\n=== pressure drop: constant-property vs fully viscosity-corrected (bound) ===")
    print(f"{'#':>2} {'q[W/cm2]':>9} {'Twall[C]':>9} {'mu_b/mu_w':>10} "
          f"{'const[kPa]':>11} {'corr[kPa]':>10} {'meas[kPa]':>10} {'err const':>10} {'err corr':>9}")
    vb = viscosity_bracket()
    for r in vb:
        print(f"{r['n']:>2} {r['q']:>9.0f} {r['T_wall']:>9.1f} {r['mu_ratio']:>10.2f} "
              f"{r['dP_const']/1e3:>11.1f} {r['dP_corr']/1e3:>10.1f} {r['dP_meas']/1e3:>10.1f} "
              f"{r['e_const']:>+9.1f}% {r['e_corr']:>+8.1f}%")
    print("\n=== is the measured data internally consistent? laminar dP scales as V/Dh^2 ===")
    base = None
    for c in CASES:
        N = 100; A = N*c["wc"]*c["H"]*1e-12; V = c["Q"]*1e-6/A
        Dh = 2*c["wc"]*c["H"]/(c["wc"]+c["H"])*1e-6
        s = V/Dh**2
        if base is None: base = s; bm = c["dP_psi"]
        print(f"  #{c['n']}: V={V:.2f} m/s Dh={Dh*1e6:.1f} um | theory dP ratio {s/base:.2f}x"
              f"  measured {c['dP_psi']/bm:.2f}x")
    json.dump(vb, open(os.path.join(HERE, "benchmark_viscosity.json"), "w"), indent=2, default=float)
