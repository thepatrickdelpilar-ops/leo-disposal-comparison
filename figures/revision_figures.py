"""Render every manuscript figure from the JSON logs in validation/logs/rev2."""
import sys
import os
import json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M
from _style import apply, C_FCC, C_IADC

LOG = os.path.join(ROOT, "validation", "logs", "rev2")
OUT = os.path.join(ROOT, "figures", "outputs")
RULES = (("FCC", C_FCC, "o", "-", "FCC 5-year"), ("IADC", C_IADC, "s", "--", "IADC 25-year"))
CEIL = 106.0


def load(name):
    return json.load(open(os.path.join(LOG, name + ".json")))


def pct(v):
    return np.where(np.array(v, dtype=float) > 1.0, CEIL, np.array(v, dtype=float) * 100)


def ceiling(ax, x0, label=True):
    ax.axhspan(100, 112, color="#888888", alpha=0.15, lw=0)
    ax.axhline(100, color="#888888", lw=0.6)
    if label:
        ax.text(x0, 106, "unachievable", fontsize=6.5, color="#555555", va="center")
    ax.set_ylim(-4, 112)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_inputs():
    alts = sorted(M.POPULATION)
    hs = np.arange(500, 901, 10)
    life = load("msis_lifetimes")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    ax = axes[0, 0]
    ax.bar(alts, [M.POPULATION[a] for a in alts], width=38, color="#4477aa", alpha=0.85)
    ax.set_xlabel("Altitude (km)"); ax.set_ylabel("Tracked objects"); ax.set_title("(A) Grounded shell population", fontsize=9)
    ax = axes[0, 1]
    sh = [a for a in alts]
    lo = [life["msis"]["cycle_mean"]["0.02"][str(a)] for a in sh]
    hi = [life["msis"]["low"]["0.01"][str(a)] for a in sh]
    ax.fill_between(sh, lo, hi, color="#bbbbbb", alpha=0.5, lw=0, label="NRLMSISE-00 envelope")
    ax.semilogy(sh, [life["msis"]["cycle_mean"]["0.01"][str(a)] for a in sh], color="#555555", ls=":", lw=1.2,
                label="NRLMSISE-00, cycle mean,\n$A/m$ = 0.01 m$^2$/kg")
    ax.semilogy(hs, [M.decay_lifetime(h) for h in hs], color="#4477aa", lw=1.6, label="Model fit, Eq. (2)")
    ax.set_xlabel("Altitude (km)"); ax.set_ylabel("Orbital lifetime (yr)"); ax.set_title("(B) Decay lifetime", fontsize=9)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    ax = axes[1, 0]
    ax.plot(hs, [M.clustering_factor(h) for h in hs], color="#4477aa")
    ax.set_xlabel("Altitude (km)"); ax.set_ylabel("Clustering factor"); ax.set_title("(C) Clustering profile", fontsize=9)
    ax = axes[1, 1]
    fr = np.array([M.CLASS_FRACTIONS[a] for a in alts])
    ax.stackplot(alts, fr[:, 0], fr[:, 1], fr[:, 2], labels=["Active", "Rocket body", "Debris"],
                 colors=["#4477aa", "#ee6677", "#bbbbbb"], alpha=0.9)
    ax.set_xlabel("Altitude (km)"); ax.set_ylabel("Class fraction"); ax.set_title("(D) Population composition", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="center right")
    fig.tight_layout()
    save(fig, "Figure1")


def fig_headline():
    h = load("headline")["exact"]
    ctl = load("control")["collisionless"]
    alts = np.array(sorted(int(a) for a in h["FCC"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, key, title in ((axes[0], "exceed", "(A) Exceedance criterion"),
                           (axes[1], "selfsust", "(B) Self-sustaining criterion")):
        for r, col, mk, ls, lab in RULES:
            v = np.array([pct(h[r][str(a)][key]) for a in alts])
            med, lo, hi = np.median(v, axis=1), np.percentile(v, 16, axis=1), np.percentile(v, 84, axis=1)
            ax.fill_between(alts, lo, hi, color=col, alpha=0.35, lw=0.6, edgecolor=col)
            fin = med <= 100
            ax.plot(alts[fin], med[fin], color=col, marker=mk, ls=ls, ms=3.2, lw=1.3, label=lab)
            ax.plot(alts[~fin], med[~fin], color=col, marker="x", ls="none", ms=5, mew=1.3)
            if key == "exceed":
                c = pct([ctl[r][str(a)]["exceed"] for a in alts])
                ax.plot(alts, c, color=col, ls=":", lw=0.9, alpha=0.9,
                        label=lab.split()[0] + ", no collisions")
        ceiling(ax, 505)
        ax.set_xlabel("Altitude (km)"); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Critical compliance (%)")
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower right", bbox_to_anchor=(1.0, 0.03))
    axes[1].legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(0.0, 0.7))
    fig.tight_layout()
    save(fig, "Figure2")


def fig_boundary():
    b = load("boundary")["exact"]
    alts = np.array(sorted(int(a) for a in b["FCC"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, key, title in ((axes[0], "exceed", "(A) Exceedance criterion"),
                           (axes[1], "selfsust", "(B) Self-sustaining criterion")):
        for r, col, mk, ls, lab in RULES:
            ax.plot(alts, pct([b[r][str(a)][key] for a in alts]), color=col, marker=mk, ls=ls, ms=3, label=lab)
        ceiling(ax, 602)
        ax.set_xlabel("Altitude (km)"); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Critical compliance (%)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7, bbox_to_anchor=(0.0, 0.9))
    fig.tight_layout()
    save(fig, "Figure3")


def fig_heatmaps():
    h = load("headline")["heatmap"]
    comp = np.array(h["compliance"]) * 100
    alts = np.array(h["altitudes"])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, sharey=True)
    for j, (r, title) in enumerate((("FCC", "FCC 5-year"), ("IADC", "IADC 25-year"))):
        for i, (key, name) in enumerate((("exceed", "exceedance"), ("selfsust", "self-sustaining"))):
            ax = axes[i, j]
            z = np.array(h[r][key])
            im = ax.pcolormesh(comp, alts, z, cmap="cividis", vmin=0, vmax=1, shading="auto")
            if z.min() < 0.5 < z.max():
                ax.contour(comp, alts, z, levels=[0.5], colors="w", linewidths=1.3)
            ax.set_title(f"{title}, {name}", fontsize=8.5)
            ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("Compliance (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Altitude (km)")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Probability", fontsize=8)
    save(fig, "Figure4")


def fig_floor():
    f = load("floor")["800"]
    s = np.array(sorted(float(k) for k in f["FCC"]))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for r, col, mk, ls, lab in RULES:
        for key, alpha, suffix, lw in (("exceed", 1.0, ", exceedance", 1.4), ("selfsust", 0.45, ", self-sustaining", 1.0)):
            v = pct([f[r][f"{x:.2f}"][key] for x in s])
            fin = v <= 100
            ax.plot(s[fin] * 100, v[fin], color=col, marker=mk, ls=ls, ms=3.5, lw=lw, alpha=alpha, label=lab + suffix)
    ceiling(ax, 51)
    ax.set_xlabel("Disposal success rate (%)"); ax.set_ylabel("Intended compliance required (%)")
    ax.set_ylim(40, 112); ax.set_xlim(49, 101)
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    save(fig, "Figure5")


def fig_avoid():
    a = load("avoid")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
    for ax, name, title in ((axes[0], "800", "(A) 800 km, catalog mix\n(12% active)"),
                            (axes[1], "650", "(B) 650 km, catalog mix\n(42% active)"),
                            (axes[2], "800_active_dominated", "(C) 800 km, hypothetical\n(90% active)")):
        al = sorted(float(k) for k in a[name]["FCC"])
        for r, col, mk, ls, lab in RULES:
            ax.plot(np.array(al) * 100, pct([a[name][r][str(x) if str(x) in a[name][r] else str(int(x))]["exceed"] for x in al]),
                    color=col, marker=mk, ls=ls, ms=3.2, label=lab)
        ceiling(ax, 2, label=False)
        ax.set_title(title, fontsize=8); ax.set_xlabel("Avoidance effectiveness (%)")
    axes[0].set_ylabel("Critical compliance (%)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "Figure6")


def fig_matrix():
    m = load("matrix")["matrix"]
    gaps = np.array([(row["IADC"]["exceed"] - row["FCC"]["exceed"]) * 100 for row in m if row["IADC"]["exceed"] <= 1])
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(gaps, bins=np.arange(-1.25, gaps.max() + 3.75, 2.5), color=C_FCC, alpha=0.85, edgecolor="white", lw=0.4)
    ax.axvline(0, color="#333333", lw=1.0)
    ax.axvline(np.median(gaps), color=C_IADC, ls="--", lw=1.2, label=f"median {np.median(gaps):.1f} points")
    ax.set_xlabel("IADC minus FCC critical compliance (points), finite cases")
    ax.set_ylabel("Parameter combinations")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "Figure7")


def fig_sensitivity():
    o = load("matrix")["oat"]
    params = [("clustering_multiplier", "Clustering"), ("population_multiplier", "Population"),
              ("decay_multiplier", "Decay lifetime"), ("launch_multiplier", "Launch rate")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharey=True)
    for ax, (p, label) in zip(axes.flat, params):
        lv = sorted(float(k) for k in o[p])
        for r, col, mk, ls, lab in RULES:
            ax.plot(lv, pct([o[p][str(x)][r]["exceed"] for x in lv]), color=col, marker=mk, ls=ls, ms=4, label=lab)
        ceiling(ax, 0.76, label=False)
        ax.set_xlabel(f"{label} multiplier")
    axes[0, 0].set_ylabel("Critical compliance (%)"); axes[1, 0].set_ylabel("Critical compliance (%)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "Figure8")


def fig_convergence():
    c = load("misc")["convergence"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for alt, ls in ((650, "-"), (800, "--")):
        n = sorted(int(k) for k in c[str(alt)]["FCC"])
        for r, col, mk, _, lab in RULES:
            ax.plot(n, pct([c[str(alt)][r][str(x)]["exceed"] for x in n]), color=col, ls=ls, marker=mk, ms=3,
                    label=f"{lab.split()[0]}, {alt} km")
    ax.set_xscale("log"); ax.set_xlabel("Monte Carlo runs"); ax.set_ylabel("Critical compliance (%)")
    ax.set_ylim(0, 112); ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
    save(fig, "Figure9")


def first_nonzero(d, key):
    alts = sorted(int(a) for a in d)
    for a in alts:
        if d[str(a)][key] > 0:
            return a
    return None


def fig_boundary_variants():
    sol, ms, kap, pw = load("solar"), load("msis"), load("kappa"), load("pairwise_recal")
    cp, ld = load("coupled"), load("launchdist")

    def first(d, key):
        xs = [first_nonzero(d[r], key) for r in ("FCC", "IADC")]
        xs = [x for x in xs if x is not None]
        return min(xs) if xs else None

    def shell_first(v, key):
        xs = [a for r in ("FCC", "IADC") for a, t in zip(M.SHELLS, v[r][key]) if t > 0]
        return min(xs) if xs else None

    def launch_first(key):
        xs = [a for r in ("FCC", "IADC") for a in M.SHELLS if np.median(ld["thresholds"][r][str(a)][key]) > 0]
        return min(xs) if xs else None

    rows = [
        ("Baseline", lambda k: first(sol["nominal"], k)),
        ("Static solar maximum ($\\varphi$ = 0.3)", lambda k: first(sol["solar_max_static"], k)),
        ("Static solar minimum ($\\varphi$ = 3)", lambda k: first(sol["solar_min_static"], k)),
        ("11-year solar cycle, factor 3", lambda k: first(sol["cycle"], k)),
        ("11-year solar cycle, factor 5", lambda k: first(sol["cycle_amp5"], k)),
        ("NRLMSISE-00 lifetimes, $A/m$ = 0.005 m$^2$/kg", lambda k: first(ms["thresholds"]["0.005"], k)),
        ("NRLMSISE-00 lifetimes, $A/m$ = 0.01 m$^2$/kg", lambda k: first(ms["thresholds"]["0.01"], k)),
        ("NRLMSISE-00 lifetimes, $A/m$ = 0.02 m$^2$/kg", lambda k: first(ms["thresholds"]["0.02"], k)),
        ("NRLMSISE-00, $A/m$ = 0.01, recalibrated $\\kappa$", lambda k: first(ms["recalibrated"]["thresholds"], k)),
        ("$\\kappa$ = 0.35", lambda k: first(kap["0.35"]["sweep"], k)),
        ("$\\kappa$ = 1.4", lambda k: first(kap["1.4"]["sweep"], k)),
        ("Pairwise yields, recalibrated $\\kappa$", lambda k: first(pw["pairwise_yields"]["sweep"], k)),
        ("Class cross-sections, recalibrated $\\kappa$", lambda k: first(pw["class_cross_sections"]["sweep"], k)),
        ("Coupled shells, drag transport*", lambda k: shell_first(cp["transport"], k)),
        ("Coupled shells, transport + fragment spread*", lambda k: shell_first(cp["transport_spread_0.3"], k)),
        ("Observed launch distribution*", lambda k: launch_first(k)),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for y, (name, fn) in enumerate(rows):
        xe, xs = fn("exceed"), fn("selfsust")
        if xe is not None and xs is not None:
            ax.plot([xe, xs], [y, y], color="#aaaaaa", lw=1.0, zorder=1)
        if xe is not None:
            ax.plot(xe, y, marker="o", color="#222222", ms=5.5, ls="none", zorder=3)
        else:
            ax.text(905, y, "none", fontsize=6.5, va="center", ha="left", color="#222222")
        if xs is not None:
            ax.plot(xs, y, marker="D", color=C_IADC, ms=5.5, fillstyle="none", mew=1.3, ls="none", zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Lowest altitude at which either rule requires nonzero compliance (km)")
    ax.set_xlim(540, 940)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color="#222222", ls="none", label="Exceedance criterion"),
                       Line2D([], [], marker="D", color=C_IADC, ls="none", fillstyle="none", mew=1.3,
                              label="Self-sustaining criterion")],
              frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2)
    save(fig, "Figure10")


ALL = [fig_inputs, fig_headline, fig_boundary, fig_heatmaps, fig_floor, fig_avoid, fig_matrix,
       fig_sensitivity, fig_convergence, fig_boundary_variants]

if __name__ == "__main__":
    apply()
    want = sys.argv[1:]
    for f in ALL:
        if not want or f.__name__ in want:
            f()
