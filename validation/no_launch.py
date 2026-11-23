import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model import population_growth

RUNS = 500


def main():
    print("No-launch 200-year growth factor and mean collisions by altitude:")
    for alt in (550, 650, 750, 850, 900):
        g, c = population_growth(0.0, 25, alt, RUNS, np.random.default_rng(42),
                                 launch_multiplier=0.0)
        print(f"  {alt} km: growth {g:5.2f}x   collisions {c:7.1f}")
    print("Expected: populations decay away at low altitude and persist toward "
          "the 800-900 km band, where drag is negligible; the baseline is "
          "sub-critical, consistent with the lower-bound framing.")


if __name__ == "__main__":
    main()
