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
ALTITUDES = np.arange(620, 731, 5)


def main():
    apply()
    fcc = [critical_compliance(5, a, RUNS, np.random.default_rng(42)) * 100 for a in ALTITUDES]
    iadc = [critical_compliance(25, a, RUNS, np.random.default_rng(42)) * 100 for a in ALTITUDES]
    for a, f, i in zip(ALTITUDES, fcc, iadc):
        print(f"{a} km: FCC {f:.0f}%  IADC {i:.0f}%")
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(ALTITUDES, np.clip(fcc, 0, 105), color=C_FCC, marker="o", ms=3, label="FCC 5-year")
    ax.plot(ALTITUDES, np.clip(iadc, 0, 105), color=C_IADC, marker="s", ls="--", ms=3, label="IADC 25-year")
    ax.axvline(650, color="#888888", lw=0.7, ls=":")
    ax.set_xlabel("Altitude (km)"); ax.set_ylabel("Critical compliance (%)")
    ax.set_ylim(-5, 110); ax.legend(frameon=False, loc="upper left")
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/fig_boundary.pdf", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/fig_boundary.png", dpi=600, bbox_inches="tight")
    print("saved outputs/fig_boundary.pdf/.png")


if __name__ == "__main__":
    main()
