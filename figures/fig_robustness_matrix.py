import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools
from model import critical_compliance
from _style import apply, C_FCC, C_IADC

RUNS = 1000
ALTITUDE = 800
LEVELS = {
    "launch_multiplier": [0.5, 1.0, 2.0],
    "population_multiplier": [0.75, 1.0, 1.25],
    "clustering_multiplier": [0.75, 1.0, 1.25],
    "fragment_multiplier": [0.75, 1.0, 1.25],
}


def main():
    apply()
    keys = list(LEVELS)
    combos = list(itertools.product(*(LEVELS[k] for k in keys)))
    gaps, preserved = [], 0
    for combo in combos:
        kw = dict(zip(keys, combo))
        f = critical_compliance(5, ALTITUDE, RUNS, np.random.default_rng(42), **kw)
        i = critical_compliance(25, ALTITUDE, RUNS, np.random.default_rng(42), **kw)
        gaps.append((min(i, 1.05) - min(f, 1.05)) * 100)
        preserved += f <= i
    gaps = np.array(gaps)
    frac = preserved / len(combos)
    print(f"ordering preserved: {preserved}/{len(combos)} ({frac:.0%})")
    print(f"gap median {np.median(gaps):.0f}, range [{gaps.min():.0f}, {gaps.max():.0f}]")
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.hist(gaps, bins=np.arange(-2.5, gaps.max() + 2.5, 2.5), color=C_FCC, alpha=0.8,
            edgecolor="white", lw=0.4)
    ax.axvline(0, color="#333333", lw=1.0)
    ax.axvline(np.median(gaps), color=C_IADC, ls="--", lw=1.2, label=f"median {np.median(gaps):.0f} pts")
    ax.set_xlabel("IADC - FCC compliance gap (pts)"); ax.set_ylabel("Parameter combinations")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.95, f"{frac:.0%} preserve\nFCC <= IADC", transform=ax.transAxes,
            va="top", fontsize=8, color="#333333")
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/fig_robustness_matrix.pdf", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/fig_robustness_matrix.png", dpi=600, bbox_inches="tight")
    print("saved outputs/fig_robustness_matrix.pdf/.png")


if __name__ == "__main__":
    main()
