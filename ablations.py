import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import model as M
from model import cascade_probability, critical_compliance, _nearest, decay_rate, collision_coefficient

RUNS = 1000
OUT = "outputs/ablations.json"


def cascade_probability_hard(compliance, deorbit_years, altitude_km, runs, rng,
                             disposal_success=1.0, active_decays=True):
    """Variant with an explicit cohort queue: compliant objects are removed
    exactly `deorbit_years` after retirement instead of at rate 1/T."""
    n0 = _nearest(altitude_km, M.POPULATION)
    fa, fr, fd = _nearest(altitude_km, M.CLASS_FRACTIONS)
    R = runs
    active = np.full(R, n0 * fa)
    rb = np.full(R, n0 * fr)
    debris = np.full(R, n0 * fd)
    cohorts = np.zeros((deorbit_years, R))
    abandoned = np.zeros(R)
    dr = decay_rate(altitude_km)
    coeff = collision_coefficient(altitude_km)
    threshold = n0 * M.CASCADE_MULTIPLE
    launch = M.LAUNCH_RATE
    active_dr = dr if active_decays else 0.0
    ya, yr, yd = (M.FRAGMENT_YIELDS["active"], M.FRAGMENT_YIELDS["rocketbody"],
                  M.FRAGMENT_YIELDS["debris"])
    cascaded = np.zeros(R, dtype=bool)
    for _ in range(M.HORIZON_YEARS):
        compliant = cohorts.sum(axis=0)
        total = active + compliant + abandoned + rb + debris
        cascaded |= total > threshold
        if not (~cascaded).any():
            break
        launched = rng.poisson(launch, R)
        retiring = np.minimum(rng.poisson(launch, R), active.astype(int))
        intending = rng.binomial(retiring, compliance)
        succeeding = rng.binomial(intending, disposal_success)
        newly_abandoned = retiring - succeeding
        collisions = rng.poisson(np.minimum(coeff * total ** 2, 1e7))
        mean_yield = np.divide(ya * active + yr * rb + yd * (debris + abandoned + compliant),
                               total, out=np.full(R, float(ya)), where=total > 0)
        active = np.maximum(active + launched - succeeding - newly_abandoned - active * active_dr, 0)
        rb = np.maximum(rb - rb * dr, 0)
        debris = np.maximum(debris - debris * dr + collisions * mean_yield, 0)
        abandoned = np.maximum(abandoned + newly_abandoned - abandoned * dr, 0)
        cohorts = cohorts * (1 - dr)
        cohorts = np.roll(cohorts, 1, axis=0)
        cohorts[0] = succeeding
    return cascaded.mean()


def critical_hard(deorbit_years, altitude_km, runs, rng, cutoff=0.5, step=0.025):
    for c in np.clip(np.arange(0, 1.0001, step), 0, 1.0):
        if cascade_probability_hard(c, deorbit_years, altitude_km, runs, rng) < cutoff:
            return float(c)
    return 1.05


def first_nonzero_altitude(deorbit_years, runs, **kw):
    for a in range(500, 901, 25):
        if critical_compliance(deorbit_years, a, runs, np.random.default_rng(42), **kw) > 0:
            return a
    return None


def fmt(v):
    return "unachievable" if v > 1 else f"{v * 100:.1f}%"


def main():
    os.makedirs("outputs", exist_ok=True)
    res = {}

    print("Solar bracket: first altitude requiring nonzero compliance (25 km grid)")
    res["solar_bracket"] = {}
    for phi, label in [(0.3, "solar_max"), (1.0, "nominal"), (3.0, "solar_min")]:
        f = first_nonzero_altitude(5, RUNS, decay_multiplier=phi)
        i = first_nonzero_altitude(25, RUNS, decay_multiplier=phi)
        res["solar_bracket"][label] = {"phi": phi, "fcc_first_nonzero_km": f, "iadc_first_nonzero_km": i}
        print(f"  phi={phi}: FCC {f} km, IADC {i} km")

    print("\nActive-decay ablation (decay on vs off for the active pool)")
    res["active_decay_ablation"] = {}
    for alt in (650, 800):
        for dy, rule in ((5, "FCC"), (25, "IADC")):
            on = critical_compliance(dy, alt, RUNS, np.random.default_rng(42), active_decays=True)
            off = critical_compliance(dy, alt, RUNS, np.random.default_rng(42), active_decays=False)
            res["active_decay_ablation"][f"{alt}_{rule}"] = {"on": on, "off": off}
            print(f"  {alt} km {rule}: on {fmt(on)}, off {fmt(off)}")

    print("\nHard-deadline ablation (exact removal after T years vs rate 1/T)")
    res["hard_deadline"] = {}
    for alt in (650, 800):
        for dy, rule in ((5, "FCC"), (25, "IADC")):
            rate = critical_compliance(dy, alt, RUNS, np.random.default_rng(42))
            hard = critical_hard(dy, alt, RUNS, np.random.default_rng(42))
            res["hard_deadline"][f"{alt}_{rule}"] = {"rate_1_over_T": rate, "hard_deadline": hard}
            print(f"  {alt} km {rule}: rate-based {fmt(rate)}, hard deadline {fmt(hard)}")

    print("\nCutoff and cascade-multiple robustness at 800 km")
    res["cutoff_multiple"] = {}
    for q in (0.3, 0.5, 0.7):
        f = critical_compliance(5, 800, RUNS, np.random.default_rng(42), cutoff=q)
        i = critical_compliance(25, 800, RUNS, np.random.default_rng(42), cutoff=q)
        res["cutoff_multiple"][f"cutoff_{q}"] = {"fcc": f, "iadc": i}
        print(f"  cutoff q={q}: FCC {fmt(f)}, IADC {fmt(i)}")
    base_mult = M.CASCADE_MULTIPLE
    for mult in (5, 10, 20):
        M.CASCADE_MULTIPLE = mult
        f = critical_compliance(5, 800, RUNS, np.random.default_rng(42))
        i = critical_compliance(25, 800, RUNS, np.random.default_rng(42))
        res["cutoff_multiple"][f"multiple_{mult}"] = {"fcc": f, "iadc": i}
        print(f"  multiple {mult}x: FCC {fmt(f)}, IADC {fmt(i)}")
    M.CASCADE_MULTIPLE = base_mult

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
