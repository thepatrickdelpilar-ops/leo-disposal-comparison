import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import cascade_probability
from _style import apply

RUNS = 500
ALTITUDES = np.arange(500, 901, 50)
COMPLIANCES = np.arange(0, 1.001, 0.1)


def main():
    apply()
    H = {}
    for dy in (5, 25):
        grid = np.zeros((len(ALTITUDES), len(COMPLIANCES)))
        for i, a in enumerate(ALTITUDES):
            for j, c in enumerate(COMPLIANCES):
                grid[i, j] = cascade_probability(c, dy, a, RUNS, np.random.default_rng(42))
        H[dy] = grid
        print(f"rule {dy}y grid done")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True)
    for ax, dy, title in [(axes[0], 5, "FCC 5-year"), (axes[1], 25, "IADC 25-year")]:
        im = ax.pcolormesh(COMPLIANCES * 100, ALTITUDES, H[dy], cmap="RdYlBu_r", vmin=0, vmax=1, shading="auto")
        ax.contour(COMPLIANCES * 100, ALTITUDES, H[dy], levels=[0.5], colors="k", linewidths=1.2)
        ax.set_xlabel("Compliance (%)"); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Altitude (km)")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Cascade probability", fontsize=8)
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/fig_heatmaps.pdf", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/fig_heatmaps.png", dpi=600, bbox_inches="tight")
    print("saved outputs/fig_heatmaps.pdf/.png")


if __name__ == "__main__":
    main()
