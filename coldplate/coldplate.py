"""
Microchannel Cold Plate Design & Optimization for a 1000 W AI Accelerator
Author: Nishant Subedi

Physics:
  - Thermal resistance network: junction -> die -> TIM -> cold-plate base -> fins/convection -> fluid
  - Laminar developing-flow Nusselt number (Shah & London rectangular duct + Stephan developing correction)
  - Fin efficiency of microchannel walls (adiabatic-tip fin)
  - Fluid caloric temperature rise
  - Pressure drop: apparent friction (Shah-London fRe + developing-flow Hagenbach K) + inlet/outlet losses
  - Pumping power = dP * Q
  - 2D finite-difference conduction in the base with a realistic non-uniform die power map (hotspots)
Outputs: design sweep, Pareto front (T_junction vs pumping power), sensitivity, hotspot map.
"""
import numpy as np, matplotlib.pyplot as plt, json, os
from itertools import product
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 140})

# ---------------- Problem definition ----------------
P_TOTAL = 1000.0           # W, accelerator TDP
DIE = 0.030                # m, 30 x 30 mm die (≈ large GPU/ASIC)
PLATE = 0.060              # m, 60 x 60 mm finned region (copper)
T_IN = 35.0                # C, facility coolant inlet (warm-water cooling)
T_J_MAX = 85.0             # C, junction limit
K_CU, K_SI = 390.0, 130.0
T_DIE = 0.00075            # m die thickness
R_TIM = 5e-6               # m^2K/W (high-end TIM / liquid metal ~ 3-8 mm^2K/W)
T_BASE = 0.002             # m base thickness under channels

# Coolant: 25% propylene glycol-water (PG25) @ ~40 C
FLUID = dict(rho=1015.0, cp=3900.0, k=0.50, mu=1.6e-3)

def fRe_rect(a):           # Shah & London fully developed, a = aspect ratio <=1
    return 24*(1-1.3553*a+1.9467*a**2-1.7012*a**3+0.9564*a**4-0.2537*a**5)

def Nu_fd_rect(a):         # Shah & London, 3-wall heated approx (H1, all walls)
    return 8.235*(1-2.0421*a+3.0853*a**2-2.4765*a**3+1.0578*a**4-0.1861*a**5)

def evaluate(w_c, w_w, H, Q_lpm, power=P_TOTAL, fluid=FLUID, r_tim=R_TIM):
    """Return dict of performance for channel width w_c, wall width w_w, channel height H (m),
    flow rate Q_lpm (L/min)."""
    rho, cp, k, mu = fluid["rho"], fluid["cp"], fluid["k"], fluid["mu"]
    W = L = PLATE
    N = int(W // (w_c + w_w))
    Q = Q_lpm / 60000.0
    A_c = w_c * H
    V = Q / (N * A_c)
    Dh = 2 * w_c * H / (w_c + H)
    Re = rho * V * Dh / mu
    Pr = mu * cp / k
    a = min(w_c, H) / max(w_c, H)
    # Developing-flow (simultaneous) Nusselt, Stephan-type correlation
    xs = L / (Dh * Re * Pr)
    Nu_fd = Nu_fd_rect(a)
    Nu = Nu_fd + 0.024 * xs ** -1.14 / (1 + 0.0358 * Pr ** 0.17 * xs ** -0.64)  # Stephan
    h = Nu * k / Dh
    # Fin efficiency of channel walls
    m = np.sqrt(2 * h / (K_CU * w_w))
    eta = np.tanh(m * H) / (m * H)
    A_base = N * w_c * L
    A_fin = N * 2 * H * L * eta
    R_conv = 1.0 / (h * (A_base + A_fin))
    # Caloric resistance (fluid temp rise, referenced to inlet)
    mdot = rho * Q
    R_cal = 1.0 / (2 * mdot * cp)          # mean fluid temp rise
    dT_fluid_out = power / (mdot * cp)
    # Conduction + spreading (die -> plate): Lee/Song-style spreading approx
    A_die = DIE ** 2
    R_die = T_DIE / (K_SI * A_die)
    R_tim = r_tim / A_die
    # Spreading resistance (Song, Lee, Au 1994) for square source on square plate
    a_s, b_s = DIE / np.sqrt(np.pi), PLATE / np.sqrt(np.pi)
    eps = a_s / b_s
    tau = T_BASE / b_s
    Bi = 1.0 / (np.pi * K_CU * b_s * R_conv)
    lam = np.pi + 1 / (np.sqrt(np.pi) * eps)
    phi = (np.tanh(lam * tau) + lam / Bi) / (1 + lam / Bi * np.tanh(lam * tau))
    psi = eps * tau / np.sqrt(np.pi) + (1 - eps) * phi / np.sqrt(np.pi)
    R_sp = psi / (K_CU * a_s * np.sqrt(np.pi))
    R_1d = T_BASE / (K_CU * PLATE ** 2)
    # Physical upper bound: no lateral spreading at all (heat goes straight down under the die)
    R_nospread = T_BASE / (K_CU * A_die) + R_conv * PLATE**2 / A_die
    if R_sp + R_1d + R_conv > R_nospread:
        R_sp = R_nospread - R_1d - R_conv
    R_total = R_die + R_tim + R_sp + R_1d + R_conv + R_cal
    Tj = T_IN + power * R_total
    # Pressure drop
    Lp = L / (Dh * Re)
    fRe_fd = fRe_rect(a)
    K_inf = 0.6796 + 1.2197 * a + 3.3089 * a**2 - 9.5921 * a**3 + 8.9089 * a**4 - 2.9959 * a**5
    fapp_Re = np.sqrt((3.44 / np.sqrt(Lp)) ** 2 + fRe_fd ** 2)  # Shah apparent friction
    f_app = fapp_Re / Re
    dP_ch = (4 * f_app * L / Dh) * 0.5 * rho * V ** 2
    sigma = w_c / (w_c + w_w)          # free-flow area ratio
    Kc, Ke = 0.42 * (1 - sigma**2) + 0.5, (1 - sigma) ** 2
    dP = dP_ch + (Kc + Ke) * 0.5 * rho * V ** 2
    W_pump = dP * Q
    return dict(w_c=w_c*1e6, w_w=w_w*1e6, H=H*1e6, Q_lpm=Q_lpm, N=N, Re=Re, h=h, eta=eta,
                Tj=Tj, dP_kPa=dP/1e3, W_pump=W_pump, dT_fluid=dT_fluid_out,
                R_total=R_total, R=dict(die=R_die, tim=R_tim, spread=R_sp+R_1d,
                conv=R_conv, caloric=R_cal), laminar=Re < 2300)

# ---------------- Design sweep ----------------
w_cs = np.array([100, 150, 200, 300, 400, 600]) * 1e-6
w_ws = np.array([100, 150, 200, 300]) * 1e-6
Hs = np.array([0.5, 1.0, 1.5, 2.0, 3.0]) * 1e-3
Qs = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
res = [evaluate(*d) for d in product(w_cs, w_ws, Hs, Qs)]
res = [r for r in res if r["laminar"] and r["H"]/r["w_c"] <= 15]   # manufacturable AR (skived fins)
Tj = np.array([r["Tj"] for r in res]); Wp = np.array([r["W_pump"] for r in res])
dP = np.array([r["dP_kPa"] for r in res])

# Pareto front: minimize Tj and pumping power
order = np.argsort(Wp); pareto = []; best = np.inf
for i in order:
    if Tj[i] < best: pareto.append(i); best = Tj[i]
pareto = np.array(pareto)

MARGIN = 12.0   # K reserved for hotspots above the uniform-power estimate
feasible = [i for i in range(len(res)) if Tj[i] <= T_J_MAX - MARGIN and dP[i] <= 50]
opt = min(feasible, key=lambda i: Wp[i]) if feasible else int(np.argmin(Tj))
best_T = int(np.argmin(Tj))
O = res[opt]

# Baseline: conventional 1 mm channel / 1 mm fin plate
base = evaluate(1000e-6, 1000e-6, 3e-3, 2.0)

# ---------------- Fig 1: Pareto ----------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
sc = ax.scatter(Wp, Tj, c=dP, s=8, cmap="viridis", norm=plt.matplotlib.colors.LogNorm(), alpha=.6)
ax.plot(Wp[pareto], Tj[pareto], "r-", lw=2, label="Pareto front")
ax.plot(Wp[opt], Tj[opt], "*", ms=16, mfc="gold", mec="k", label="Selected design")
ax.plot(base["W_pump"], base["Tj"], "s", ms=9, mfc="w", mec="k", label="Conventional 1 mm plate")
ax.axhline(T_J_MAX, ls="--", c="gray"); ax.text(1e-3, T_J_MAX+1, "T$_{j,max}$ = 85 °C", color="gray")
ax.set_xscale("log"); ax.set_xlabel("Pumping power [W]"); ax.set_ylabel("Junction temperature [°C]")
ax.set_ylim(45, 160); fig.colorbar(sc, label="Pressure drop [kPa]")
ax.set_title(f"{len(res)} cold-plate designs @ {P_TOTAL:.0f} W"); ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/1_pareto.png"); plt.close()

# ---------------- Fig 2: resistance breakdown ----------------
fig, ax = plt.subplots(figsize=(7.5, 3.9))
labels = ["die", "tim", "spread", "conv", "caloric"]
names = ["Die", "TIM", "Spreading + base", "Convection", "Fluid (caloric)"]
cols = plt.cm.Set2(np.arange(5))
for j, (lab, d) in enumerate([("Conventional", base), ("Optimized", O)]):
    left = 0
    for c, k_, n in zip(cols, labels, names):
        v = d["R"][k_] * 1000
        ax.barh(j, v, left=left, color=c, label=n if j == 0 else None, edgecolor="k")
        left += v
    ax.text(left + .5, j, f"{left:.1f} mK/W", va="center")
ax.set_yticks([0, 1], ["Conventional\n1 mm plate", "Optimized\nmicrochannel"])
ax.set_xlabel("Thermal resistance [K/kW]  (mK/W)"); ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5,-0.28))
ax.set_xlim(0, max(base["R_total"], O["R_total"]) * 1300)
ax.set_title("Junction-to-coolant resistance stack"); fig.tight_layout(); fig.savefig(f"{OUT}/2_resistance_stack.png"); plt.close()

# ---------------- Fig 3: parametric sensitivity ----------------
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
wc0, ww0, H0, Q0 = O["w_c"]*1e-6, O["w_w"]*1e-6, O["H"]*1e-6, O["Q_lpm"]
for Hh in [1e-3, 2e-3, 3e-3]:
    xs = np.linspace(80e-6, 800e-6, 40)
    axs[0].plot(xs*1e6, [evaluate(x, ww0, Hh, Q0)["Tj"] for x in xs], label=f"H={Hh*1e3:.0f} mm")
axs[0].set_xlabel("Channel width [µm]"); axs[0].set_ylabel("T$_j$ [°C]"); axs[0].legend(fontsize=8)
qs = np.linspace(0.3, 5, 40); r_ = [evaluate(wc0, ww0, H0, q) for q in qs]
axs[1].plot(qs, [r["Tj"] for r in r_], "b"); a2 = axs[1].twinx()
a2.plot(qs, [r["dP_kPa"] for r in r_], "r--"); a2.set_ylabel("ΔP [kPa]", color="r")
axs[1].set_xlabel("Flow rate [L/min]"); axs[1].set_ylabel("T$_j$ [°C]", color="b")
tims = np.linspace(2e-6, 30e-6, 30)
axs[2].plot(tims*1e6, [evaluate(wc0, ww0, H0, Q0, r_tim=t)["Tj"] for t in tims], "k")
axs[2].axvline(R_TIM*1e6, ls=":", c="gray")
axs[2].set_xlabel("TIM resistance [mm²K/W]"); axs[2].set_ylabel("T$_j$ [°C]")
axs[2].set_title("TIM matters as much as the cold plate", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/3_sensitivity.png"); plt.close()

# ---------------- Fig 4: 2D hotspot map (finite difference) ----------------
# Base-plane conduction with die power map; under-channel convection as distributed sink h_eff(x)
n = 121; dx = PLATE / (n - 1)
x = np.linspace(0, PLATE, n); X, Y = np.meshgrid(x, x)
q = np.zeros((n, n)); c0, c1 = (PLATE-DIE)/2, (PLATE+DIE)/2
die = (X >= c0) & (X <= c1) & (Y >= c0) & (Y <= c1)
q[die] = 1.0
def blob(xc, yc, s, amp):  # hotspots (tensor cores / HBM PHY)
    return amp * np.exp(-((X-xc)**2+(Y-yc)**2)/(2*s**2))
q += die * (blob(c0+.010, c0+.010, .003, 1.2) + blob(c0+.021, c0+.018, .0035, 1.0) + blob(c0+.015, c0+.026, .0025, 1.5))
q *= P_TOTAL / (q.sum() * dx * dx)                   # W/m^2
# Effective h from the 1D model referenced to base area
h_eff = 1.0 / (O["R"]["conv"] * PLATE**2)
mdot_cp = FLUID["rho"] * O["Q_lpm"]/60000 * FLUID["cp"]
t_eff = T_BASE                                       # lateral conduction in the base only (conservative)
G = K_CU * t_eff
A = lil_matrix((n*n, n*n)); b = np.zeros(n*n)
idx = lambda i, j: i*n + j
# fluid temperature rises along flow direction x (march, linear approx using power fraction)
cumP = np.cumsum(q.sum(axis=0)) * dx * dx
Tf_x = T_IN + cumP / mdot_cp
for i in range(n):
    for j in range(n):
        p = idx(i, j); diag = h_eff * dx * dx
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ii, jj = i+di, j+dj
            if 0 <= ii < n and 0 <= jj < n:
                A[p, idx(ii, jj)] = -G; diag += G
        A[p, p] = diag
        b[p] = q[i, j]*dx*dx + h_eff*dx*dx*Tf_x[j]
Tb = spsolve(A.tocsr(), b).reshape(n, n)
# add die+TIM local 1D rise for junction map
Tjmap = Tb + q * (T_DIE/K_SI + R_TIM)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
im0 = axs[0].imshow(q/1e4, extent=[0,60,0,60], origin="lower", cmap="inferno")
axs[0].set_title("Die power map [W/cm²]"); fig.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(np.where(die, Tjmap, np.nan), extent=[0,60,0,60], origin="lower", cmap="turbo")
axs[1].contour(X*1e3, Y*1e3, Tb, 12, colors="k", linewidths=.4)
axs[1].set_title(f"Junction temperature [°C]  (max {np.nanmax(Tjmap[die]):.1f} °C)")
fig.colorbar(im1, ax=axs[1])
for a_ in axs: a_.set_xlabel("x [mm]  (flow →)"); a_.set_ylabel("y [mm]")
fig.tight_layout(); fig.savefig(f"{OUT}/4_hotspot_map.png"); plt.close()

summary = dict(n_designs=len(res), selected=O, best_Tj=res[best_T], baseline=base,
               hotspot_Tj_max=float(np.nanmax(Tjmap[die])), hotspot_Tj_mean=float(np.nanmean(Tjmap[die])),
               avg_heat_flux_W_cm2=P_TOTAL/(DIE*100)**2, peak_heat_flux_W_cm2=float(q.max()/1e4))
json.dump(summary, open(os.path.join(os.path.dirname(OUT), "results.json"), "w"), indent=2, default=float)
print(json.dumps(summary, indent=2, default=float))
