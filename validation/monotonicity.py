import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model import cascade_probability

RUNS = 2000
ALT = 800
TOL = 0.05


def check(name, a, b):
    ok = a <= b + TOL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {a:.2f} <= {b:.2f}")
    return ok


def main():
    p = lambda c, dy, **kw: cascade_probability(c, dy, ALT, RUNS, np.random.default_rng(42), **kw)
    results = [
        check("higher compliance not worse", p(0.95, 25), p(0.3, 25)),
        check("higher disposal success not worse",
              p(0.7, 25, disposal_success=1.0), p(0.7, 25, disposal_success=0.6)),
        check("higher launch not safer",
              p(0.7, 25, launch_multiplier=0.4), p(0.7, 25, launch_multiplier=2.0)),
        check("higher population not safer",
              p(0.7, 25, population_multiplier=0.75), p(0.7, 25, population_multiplier=1.25)),
        check("higher clustering not safer",
              p(0.7, 25, clustering_multiplier=0.75), p(0.7, 25, clustering_multiplier=1.25)),
        check("FCC not stricter than IADC", p(0.7, 5), p(0.7, 25)),
        check("zero launch not worse",
              p(0.7, 25, launch_multiplier=0.0), p(0.7, 25, launch_multiplier=1.0)),
        check("full compliance safest", p(1.0, 25), p(0.0, 25)),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
