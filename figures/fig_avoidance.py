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

RUNS = 2000
ALTITUDE = 800
ALPHAS = [0, 0.2, 0.4, 0.6, 0.8, 0.95]


def main():
    apply()
    fcc = [critical_compliance(5, ALTITUDE, RUNS, np.random.default_rng(42), avoidance=a) * 100 for a in ALPHAS]
    iadc = [critical_compliance(25, ALTITUDE, RUNS, np.random.default_rng(42), avoidance=a) * 100 for a in ALPHAS]
    for a, f, i in zip(ALPHAS, fcc, iadc):
        print(f"avoidance {a:.0%}: FCC {f:.0f}%  IADC {i:.0f}%")
    x = np.array(ALPHAS) * 100
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(x, np.clip(fcc, 0, 105), color=C_FCC, marker="o", ms=3.5, label="FCC 5-year")
    ax.plot(x, np.clip(iadc, 0, 105), color=C_IADC, marker="s", ls="--", ms=3.5, label="IADC 25-year")
    ax.set_xlabel("Avoidance effectiveness (%)"); ax.set_ylabel("Critical compliance (%)")
    ax.set_ylim(0, 110); ax.legend(frameon=False, loc="lower left")
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/fig_avoidance.pdf", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/fig_avoidance.png", dpi=600, bbox_inches="tight")
    print("saved outputs/fig_avoidance.pdf/.png")


if __name__ == "__main__":
    main()
