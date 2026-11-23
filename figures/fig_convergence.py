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

RUN_COUNTS = [100, 250, 500, 1000, 2000]
STYLES = {650: "-", 800: "--"}


def main():
    apply()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for alt in (650, 800):
        fcc = [min(critical_compliance(5, alt, r, np.random.default_rng(42)), 1.05) * 100 for r in RUN_COUNTS]
        iadc = [min(critical_compliance(25, alt, r, np.random.default_rng(42)), 1.05) * 100 for r in RUN_COUNTS]
        print(f"{alt} km: FCC {fcc}  IADC {iadc}")
        ax.plot(RUN_COUNTS, fcc, color=C_FCC, ls=STYLES[alt], marker="o", ms=3, label=f"FCC {alt} km")
        ax.plot(RUN_COUNTS, iadc, color=C_IADC, ls=STYLES[alt], marker="s", ms=3, label=f"IADC {alt} km")
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo runs"); ax.set_ylabel("Critical compliance (%)")
    ax.set_ylim(0, 112); ax.legend(frameon=False, fontsize=6.5, ncol=2)
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/fig_convergence.pdf", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/fig_convergence.png", dpi=600, bbox_inches="tight")
    print("saved outputs/fig_convergence.pdf/.png")


if __name__ == "__main__":
    main()
