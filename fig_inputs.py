import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import POPULATION, CLASS_FRACTIONS, decay_rate, clustering_factor
from _style import apply


def main():
    apply()
    alts = sorted(POPULATION)
    hs = np.arange(500, 901, 10)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6))
    axes[0, 0].bar(alts, [POPULATION[a] for a in alts], width=38, color="#4477aa", alpha=0.85)
    axes[0, 0].set_xlabel("Altitude (km)"); axes[0, 0].set_ylabel("Tracked objects")
    axes[0, 0].set_title("Grounded shell population", fontsize=9)
    axes[0, 1].semilogy(hs, [1 / decay_rate(h) for h in hs], color="#4477aa")
    axes[0, 1].set_xlabel("Altitude (km)"); axes[0, 1].set_ylabel("Orbital lifetime (yr)")
    axes[0, 1].set_title("Decay lifetime", fontsize=9)
    axes[1, 0].plot(hs, [clustering_factor(h) for h in hs], color="#4477aa")
    axes[1, 0].set_xlabel("Altitude (km)"); axes[1, 0].set_ylabel("clustering factor")
    axes[1, 0].set_title("Calibrated clustering", fontsize=9)
    fr = np.array([CLASS_FRACTIONS[a] for a in alts])
    axes[1, 1].stackplot(alts, fr[:, 0], fr[:, 1], fr[:, 2],
                         labels=["Active", "Rocket body", "Debris"],
                         colors=["#4477aa", "#ee6677", "#bbbbbb"], alpha=0.9)
    axes[1, 1].set_xlabel("Altitude (km)"); axes[1, 1].set_ylabel("Class fraction")
    axes[1, 1].set_title("Population composition", fontsize=9)
    axes[1, 1].legend(frameon=False, fontsize=6.5, loc="center right")
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/fig_inputs.pdf", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/fig_inputs.png", dpi=600, bbox_inches="tight")
    print("saved outputs/fig_inputs.pdf/.png")


if __name__ == "__main__":
    main()
