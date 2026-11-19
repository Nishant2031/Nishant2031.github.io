"""
Runs every study for the package reliability write-up and saves results.json + figures.

  1. Baseline (underfilled C4) and no-underfill cases, JEDEC JESD22-A104 condition G
     (-40/+125 C, 15 min ramps and dwells, 1 cycle per hour)
  2. Design study: substrate CTE, underfill modulus, interposer size, C4 standoff height
  3. Monte Carlo over manufacturing scatter + simulated 77-part test-to-failure, Weibull fit
  4. Field life on a data-centre power cycle that uses the junction temperatures from my
     cold plate project, compared with Norris-Landzberg acceleration
"""
import json
import math
import time
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from package import (Package, run_case, make_grid, warpage, section, weibull_mle,
                     weibull_quantile, norris_landzberg_sac, life_syed)

EMBER, STEEL, GREY, GOLD = "#c2410c", "#1d4e89", "#5a626d", "#d99a2b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "savefig.dpi": 170, "figure.dpi": 100})

COND_G = dict(Tmin=-40.0, Tmax=125.0, ramp_min=15, dwell_min=15)
RNG = np.random.default_rng(2027)


def short(m):
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}


def main():
    t_start = time.time()
    R = {}
    base = Package()
    R["package"] = dict(die_mm=base.die, interposer_mm=base.interposer, substrate_mm=base.substrate,
                        t_die=base.t_die, t_int=base.t_int, t_sub=base.t_sub, standoff_mm=base.h,
                        pitch_mm=base.pitch, bump_d_mm=base.d_bump, bump_area_fraction=base.f,
                        T_ref_C=base.T_ref, corner_DNP_mm=base.L)

    # ------------------------------------------------------------------ 1. baseline
    m_uf, rec_uf = run_case(base, n_cycles=3, dt=5, **COND_G)
    noUF = replace(base, underfill=False, T_ref=217.0)
    m_no, rec_no = run_case(noUF, n_cycles=3, dt=5, **COND_G)
    R["baseline_underfill"] = short(m_uf)
    R["no_underfill"] = short(m_no)
    print("baseline", short(m_uf))
    print("no UF   ", short(m_no))

    # ------------------------------------------------------------------ 2. design study
    studies = {
        "alpha_sub_ppm": [(v, dict(alpha_sub=v * 1e-6)) for v in (10, 12, 14, 15, 17)],
        "E_uf_GPa": [(v, dict(E_uf=v * 1e3)) for v in (3, 6, 9, 12, 15)],
        "interposer_mm": [(v, dict(interposer=v, substrate=v + 20)) for v in (33, 40, 50, 60, 70)],
        "standoff_um": [(v, dict(h=v * 1e-3)) for v in (50, 65, 80, 100)],
    }
    R["design_study"] = {}
    for key, rows in studies.items():
        out = []
        for v, kw in rows:
            m, _ = run_case(replace(base, **kw), n_cycles=2, dt=10, **COND_G)
            out.append(dict(value=v, N_energy=m["N_energy"], N_strain=m["N_strain"],
                            warp_25_um=m["warp_25_um"], warp_cold_um=m["warp_cold_um"],
                            d_gamma=m["d_gamma"], W=m["W"]))
            print(key, v, round(m["N_energy"]), round(m["warp_25_um"]))
        R["design_study"][key] = out

    # ------------------------------------------------------------------ 3. Monte Carlo
    n_mc = 120
    samples = []
    for i in range(n_mc):
        kw = dict(h=RNG.normal(0.080, 0.004), d_bump=RNG.normal(0.090, 0.003),
                  alpha_sub=RNG.normal(15e-6, 0.5e-6), E_uf=RNG.normal(9e3, 500.0),
                  E_sub=RNG.normal(28e3, 1500.0))
        m, _ = run_case(replace(base, **kw), n_cycles=2, dt=10, **COND_G)
        samples.append(dict(kw, N=m["N_energy"], warp_25_um=m["warp_25_um"]))
    N_mc = np.array([s["N"] for s in samples])
    # bias correction: production runs use 3 cycles at dt=5 s; MC uses 2 cycles at dt=10 s
    m_fast, _ = run_case(base, n_cycles=2, dt=10, **COND_G)
    bias = m_uf["N_energy"] / m_fast["N_energy"]
    N_mc = N_mc * bias
    beta_int = 6.0      # intrinsic (microstructural) scatter of solder fatigue, Weibull slope
    med_factor = (math.log(2)) ** (1 / beta_int)          # Weibull with median 1
    # "true" population: geometry scatter x intrinsic scatter
    big = RNG.choice(N_mc, 200000) * RNG.weibull(beta_int, 200000) / med_factor
    true_B1, true_B10, true_B50 = np.percentile(big, [1, 10, 50])
    # simulated qualification-style test: 77 parts, stopped at 3000 cycles
    n_parts, t_stop = 77, 3000.0
    parts = RNG.choice(N_mc, n_parts) * RNG.weibull(beta_int, n_parts) / med_factor
    failed = parts <= t_stop
    t_obs = np.where(failed, parts, t_stop)
    beta_hat, eta_hat = weibull_mle(t_obs, failed)
    B1, B10 = weibull_quantile(beta_hat, eta_hat, 0.01), weibull_quantile(beta_hat, eta_hat, 0.10)
    # bootstrap confidence on B1 / B10
    boot = []
    for _ in range(400):
        idx = RNG.integers(0, n_parts, n_parts)
        if failed[idx].sum() < 3:
            continue
        b, e = weibull_mle(t_obs[idx], failed[idx])
        boot.append((weibull_quantile(b, e, 0.01), weibull_quantile(b, e, 0.10), b))
    boot = np.array(boot)
    ci = lambda col: [float(np.percentile(boot[:, col], 5)), float(np.percentile(boot[:, col], 95))]
    R["monte_carlo"] = dict(
        n_samples=n_mc, bias_factor=bias,
        model_life_spread=dict(min=float(N_mc.min()), median=float(np.median(N_mc)), max=float(N_mc.max()),
                               cov_pct=100 * float(N_mc.std() / N_mc.mean())),
        intrinsic_beta=beta_int,
        population=dict(B1=float(true_B1), B10=float(true_B10), B50=float(true_B50)),
        test=dict(n_parts=n_parts, stop_cycles=t_stop, failures=int(failed.sum()),
                  beta=beta_hat, eta=eta_hat, B1=B1, B10=B10, B1_90ci=ci(0), B10_90ci=ci(1), beta_90ci=ci(2)))
    print("MC", json.dumps(R["monte_carlo"], indent=1))

    # sensitivity of the MC: rank correlation of each input with life
    from scipy.stats import spearmanr
    sens = {}
    for k in ("h", "d_bump", "alpha_sub", "E_uf", "E_sub"):
        sens[k] = float(spearmanr([s[k] for s in samples], N_mc).correlation)
    R["monte_carlo"]["spearman"] = sens

    # ------------------------------------------------------------------ 4. field life
    # Cold plate project: 1 kW at 69 C junction, 35 C coolant. Idle ~ 38 C.
    field = dict(Tmin=38.0, Tmax=69.0, ramp_min=5, dwell_min=240, dwell_cold_min=110)  # 4 cycles/day
    m_field, rec_field = run_case(base, n_cycles=3, dt=10, **field)
    period_test_h = (2 * 15 + 15 + 15) / 60
    period_field_h = (2 * 5 + 240 + 110) / 60
    AF_model = m_field["N_energy"] / m_uf["N_energy"]
    AF_NL = norris_landzberg_sac(165.0, 31.0, 24 / period_test_h, 24 / period_field_h, 125.0, 69.0)
    cyc_per_year = 365 * 24 / period_field_h
    R["field"] = dict(
        profile=field, cycles_per_day=24 / period_field_h, metrics=short(m_field),
        AF_model=AF_model, AF_norris_landzberg=AF_NL,
        B1_field_years_NL=B1 * AF_NL / cyc_per_year,
        B1_field_years_model=B1 * AF_model / cyc_per_year,
        B1_field_years_NL_24_per_day=B1 * norris_landzberg_sac(165.0, 31.0, 24 / period_test_h, 24.0, 125.0, 69.0) / (365 * 24))
    print("field", json.dumps(R["field"], indent=1))

    # ------------------------------------------------------------------ figures
    make_figures(base, noUF, rec_uf, rec_no, m_uf, m_no, R, big, t_obs, failed, beta_hat, eta_hat)
    R["runtime_s"] = time.time() - t_start
    with open("results.json", "w") as fh:
        json.dump(R, fh, indent=1)
    print("done in %.0f s" % R["runtime_s"])


def last_cycle(rec, period_s=3600.0):
    sel = rec["t"] > rec["t"][-1] - period_s - 1e-9
    return sel


def make_figures(base, noUF, rec_uf, rec_no, m_uf, m_no, R, big, t_obs, failed, beta_hat, eta_hat):
    # ---- Figure 1: bond shear near the corner + hysteresis loops
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for rec, pkg, col, lab in ((rec_uf, base, STEEL, "with underfill"), (rec_no, noUF, EMBER, "no underfill")):
        i_cold = int(np.argmin(rec["T"]))
        x = rec["x"]
        ax[0].plot(pkg.L - x, 100 * np.abs(rec["gamma"][i_cold]), color=col, lw=2, label=lab + " (−40 °C)")
    ax[0].axvline(base.pitch / 2, color=GREY, ls=":", lw=1)
    ax[0].annotate("corner bump", xy=(base.pitch / 2, 3.0), xytext=(0.45, 3.6), color=GREY, fontsize=8.5,
                   arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax[0].set_xlim(0, 3.0)
    ax[0].set_xlabel("distance in from the interposer corner [mm]")
    ax[0].set_ylabel("C4 shear strain [%]")
    ax[0].set_title("Shear concentrates in the last millimetre")
    ax[0].legend(frameon=False, fontsize=8.5)
    for rec, col, lab, m in ((rec_uf, STEEL, "with underfill", m_uf), (rec_no, EMBER, "no underfill", m_no)):
        sel = last_cycle(rec)
        ax[1].plot(100 * rec["gamma_crit"][sel], rec["tau_s_crit"][sel], color=col, lw=2,
                   label=f"{lab}: ΔW = {m['W']:.2f} MPa/cycle")
    ax[1].set_xlabel("corner-bump shear strain [%]")
    ax[1].set_ylabel("solder shear stress [MPa]")
    ax[1].set_title("Corner-bump hysteresis, one −40/125 °C cycle")
    ax[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    ax[1].axhline(0, color=GREY, lw=0.6); ax[1].axvline(0, color=GREY, lw=0.6)
    fig.tight_layout()
    fig.savefig("figures/1_bond_shear_hysteresis.png")
    plt.close(fig)

    # ---- Figure 2: warpage profiles through the cycle
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    sec = section(base, rec_uf["x"])
    for T_target, col in ((-40, STEEL), (25, GOLD), (125, EMBER)):
        sel = np.where(last_cycle(rec_uf))[0]
        i = sel[np.argmin(np.abs(rec_uf["T"][sel] - T_target))]
        xw, w = warpage(base, rec_uf["x"], rec_uf["N"][i], rec_uf["T"][i] - base.T_ref, sec)
        ax.plot(xw, 1e3 * w, color=col, lw=2, label=f"{T_target:+d} °C")
    for xv, lab in ((base.x_die, "die corner"), (base.L, "interposer corner")):
        ax.axvline(xv, color=GREY, ls=":", lw=1)
        ax.text(xv - 0.6, -60, lab, rotation=90, color=GREY, fontsize=8, ha="right", va="top")
    ax.set_xlabel("distance from package centre along the diagonal [mm]")
    ax.set_ylabel("bottom-surface height [µm]")
    ax.set_title("Package warpage (bare, no lid or stiffener)")
    ax.legend(frameon=False, title="temperature", fontsize=8.5, title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig("figures/2_warpage.png")
    plt.close(fig)

    # ---- Figure 3: design study
    ds = R["design_study"]
    fig, axs = plt.subplots(1, 4, figsize=(12.6, 3.4))
    specs = (("alpha_sub_ppm", "substrate CTE [ppm/K]"), ("E_uf_GPa", "underfill modulus [GPa]"),
             ("interposer_mm", "interposer edge [mm]"), ("standoff_um", "C4 standoff height [µm]"))
    for a, (key, lab) in zip(axs, specs):
        v = [r["value"] for r in ds[key]]
        N = [r["N_energy"] for r in ds[key]]
        w = [abs(r["warp_25_um"]) for r in ds[key]]
        a.plot(v, N, "o-", color=STEEL, lw=2, ms=4)
        a.set_xlabel(lab)
        a.set_yscale("log")
        a.set_ylim(1000, 12000)
        a.tick_params(axis="y", colors=STEEL)
        b = a.twinx()
        b.plot(v, w, "s--", color=EMBER, lw=1.6, ms=3.5)
        b.set_ylim(0, 1150)
        b.tick_params(axis="y", colors=EMBER)
        b.spines["top"].set_visible(False)
        if key == "alpha_sub_ppm":
            a.set_ylabel("C4 life, cycles to failure", color=STEEL)
        if key == "standoff_um":
            b.set_ylabel("warpage at 25 °C [µm]", color=EMBER)
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.suptitle("Design study: life (blue, log scale) and warpage (orange)", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/3_design_study.png")
    plt.close(fig)

    # ---- Figure 4: Weibull plot of the simulated test
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    order = np.argsort(t_obs)
    t_sorted, f_sorted = t_obs[order], failed[order]
    n = len(t_obs)
    # Johnson's adjusted ranks for censored data, Benard median ranks
    ranks, prev, rev = [], 0.0, n
    for i in range(n):
        if f_sorted[i]:
            inc = (n + 1 - prev) / (1 + (n - i))
            prev = prev + inc
            ranks.append((t_sorted[i], (prev - 0.3) / (n + 0.4)))
    tr = np.array([r[0] for r in ranks]); Fr = np.array([r[1] for r in ranks])
    yfun = lambda F: np.log(-np.log(1 - F))
    ax.plot(tr, yfun(Fr), "o", color=STEEL, ms=4, label=f"simulated failures ({failed.sum()} of {n})")
    tt = np.logspace(np.log10(600), np.log10(12000), 200)
    ax.plot(tt, beta_hat * np.log(tt / eta_hat), color=STEEL, lw=1.8,
            label=f"censored MLE fit: β = {beta_hat:.1f}, η = {eta_hat:,.0f}")
    q = np.linspace(0.001, 0.999, 400)
    ax.plot(np.percentile(big, 100 * q), yfun(q), color=EMBER, lw=1.4, ls="--", label="true population (model)")
    ax.axvline(3000, color=GREY, ls=":", lw=1)
    ax.text(3080, yfun(0.002), "test stopped\n3,000 cycles", color=GREY, fontsize=8)
    ax.set_xscale("log")
    ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99]
    ax.set_yticks(yfun(np.array(ticks)))
    ax.set_yticklabels([f"{100 * t:g}%" for t in ticks])
    ax.set_ylim(yfun(0.0008), yfun(0.995))
    ax.set_xlim(600, 12000)
    ax.set_xlabel("cycles to failure, −40/125 °C")
    ax.set_ylabel("fraction failed")
    ax.set_title("Simulated 77-part thermal-cycling test")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/4_weibull.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
