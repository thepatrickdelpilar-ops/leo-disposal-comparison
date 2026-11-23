import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model import critical_both, simulate

RUNS = 2000
ALT = 800
STEP = 0.025


def th(T, **kw):
    return critical_both(T, ALT, RUNS, np.random.default_rng(42), **kw)


def check(name, a, b, results):
    ok = a <= b + 1e-9
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {a:.3f} <= {b:.3f}")


def main():
    for T, rule in ((5, "FCC 5-year"), (25, "IADC 25-year")):
        base = th(T)
        for k, crit in enumerate(("exceed", "selfsust")):
            print(f"{rule}, {crit} criterion (baseline threshold {base[k]:.3f})")
            res = []
            c = min(base[k], 1.0)
            lo = simulate(max(c - 2 * STEP, 0), T, ALT, RUNS, np.random.default_rng(42))[crit]
            hi = simulate(min(c + 2 * STEP, 1), T, ALT, RUNS, np.random.default_rng(42))[crit]
            check("probability falls as compliance rises", hi, lo, res)
            check("higher disposal success needs no more compliance", th(T, disposal_success=1.0)[k], th(T, disposal_success=0.9)[k], res)
            check("more launches need no less compliance", th(T, launch_multiplier=0.75)[k], th(T, launch_multiplier=1.25)[k], res)
            check("larger population needs no less compliance", th(T, population_multiplier=0.75)[k], th(T, population_multiplier=1.25)[k], res)
            check("more clustering needs no less compliance", th(T, clustering_multiplier=0.75)[k], th(T, clustering_multiplier=1.25)[k], res)
            check("FCC needs no more compliance than IADC", th(5)[k], th(25)[k], res)
            check("zero launches never flagged", simulate(0.0, T, ALT, RUNS, np.random.default_rng(42), launch_multiplier=0.0)[crit], 0.0, res)
            check("full compliance no worse than none", simulate(1.0, T, ALT, RUNS, np.random.default_rng(42))[crit],
                  simulate(0.0, T, ALT, RUNS, np.random.default_rng(42))[crit], res)
            print(f"  {sum(res)}/{len(res)} checks passed\n")


if __name__ == "__main__":
    main()
