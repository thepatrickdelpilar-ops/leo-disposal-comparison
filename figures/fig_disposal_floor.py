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
SUCCESS = np.clip(np.arange(0.55, 1.001, 0.05), 0, 1.0)


def main():
    apply()
    fcc, iadc = [], []
    for s in SUCCESS:
        f = critical_compliance(5, ALTITUDE, RUNS, np.random.default_rng(42), disposal_success=s) * 100
        i = critical_compliance(25, ALTITUDE, RUNS, np.random.default_rng(42), disposal_success=s) * 100
        fcc.append(f); iadc.append(i)
        print(f"success {s:.0%}: FCC {f:.0f}  IADC {i:.0f}")
    x = SUCCESS * 100
    fp = np.where(np.array(fcc) > 100, np.nan, fcc)
    ip = np.where(np.array(iadc) > 100, np.nan, iadc)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axhspan(100, 115, color="#999999", alpha=0.12)
    ax.text(57, 107, "unachievable", fontsize=7, color="#666666", va="center")
    ax.plot(x, ip, color=C_IADC, marker="s", ls="--", ms=4, label="IADC 25-year")
    ax.plot(x, fp, color=C_FCC, marker="o", ms=4, label="FCC 5-year")
    ax.axhline(100, color="#999999", lw=0.8)
    ax.set_xlabel("Disposal success rate (%)"); ax.set_ylabel("Intended compliance required (%)")
    ax.set_ylim(0, 115); ax.set_xlim(54, 101)
    ax.legend(frameon=False, loc="lower left")
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/fig_disposal_floor.pdf", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/fig_disposal_floor.png", dpi=600, bbox_inches="tight")
    print("saved outputs/fig_disposal_floor.pdf/.png")


if __name__ == "__main__":
    main()
