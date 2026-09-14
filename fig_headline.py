import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import critical_compliance_multiseed
from _style import apply, C_FCC, C_IADC

N_SEEDS = 15
RUNS = 1000
ALTITUDES = np.arange(500, 901, 25)


def main():
    apply()
    fm, fl, fh, im, il, ih = [], [], [], [], [], []
    for a in ALTITUDES:
        vf = critical_compliance_multiseed(5, a, RUNS, N_SEEDS) * 100
        vi = critical_compliance_multiseed(25, a, RUNS, N_SEEDS) * 100
        fm.append(np.median(vf)); fl.append(np.percentile(vf, 16)); fh.append(np.percentile(vf, 84))
        im.append(np.median(vi)); il.append(np.percentile(vi, 16)); ih.append(np.percentile(vi, 84))
        print(f"{a} km: FCC {np.median(vf):.0f}%  IADC {np.median(vi):.0f}%")
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.fill_between(ALTITUDES, np.clip(fl, 0, 100), np.clip(fh, 0, 100), color=C_FCC, alpha=0.18, lw=0)
    ax.fill_between(ALTITUDES, np.clip(il, 0, 100), np.clip(ih, 0, 100), color=C_IADC, alpha=0.18, lw=0)
    ax.plot(ALTITUDES, fm, color=C_FCC, marker="o", ms=3, label="FCC 5-year")
    ax.plot(ALTITUDES, im, color=C_IADC, marker="s", ls="--", ms=3, label="IADC 25-year")
    ax.set_xlabel("Altitude (km)"); ax.set_ylabel("Critical compliance (%)")
    ax.set_ylim(-5, 108); ax.legend(frameon=False, loc="lower right")
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/fig_headline.pdf", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/fig_headline.png", dpi=600, bbox_inches="tight")
    print("saved outputs/fig_headline.pdf/.png")


if __name__ == "__main__":
    main()
