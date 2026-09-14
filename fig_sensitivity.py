import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import critical_compliance
from _style import apply, C_FCC, C_IADC

RUNS = 1000
ALTITUDE = 800
PARAMS = [("clustering_multiplier", "Clustering"), ("population_multiplier", "Population"),
          ("decay_multiplier", "Decay lifetime"), ("launch_multiplier", "Launch rate")]
LEVELS = [0.75, 1.0, 1.25]


def main():
    apply()
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6), sharey=True)
    for ax, (param, label) in zip(axes.flat, PARAMS):
        fcc = [min(critical_compliance(5, ALTITUDE, RUNS, np.random.default_rng(42), **{param: m}), 1.05) * 100 for m in LEVELS]
        iadc = [min(critical_compliance(25, ALTITUDE, RUNS, np.random.default_rng(42), **{param: m}), 1.05) * 100 for m in LEVELS]
        print(f"{label}: FCC {fcc}  IADC {iadc}")
        ax.plot(LEVELS, fcc, color=C_FCC, marker="o", ms=4, label="FCC 5-year")
        ax.plot(LEVELS, iadc, color=C_IADC, marker="s", ls="--", ms=4, label="IADC 25-year")
        ax.set_xlabel(f"{label} multiplier"); ax.set_ylim(0, 112)
    axes[0, 0].set_ylabel("Critical compliance (%)")
    axes[1, 0].set_ylabel("Critical compliance (%)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/fig_sensitivity.pdf", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/fig_sensitivity.png", dpi=600, bbox_inches="tight")
    print("saved outputs/fig_sensitivity.pdf/.png")


if __name__ == "__main__":
    main()
