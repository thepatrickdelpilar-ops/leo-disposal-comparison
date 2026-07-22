"""
Reduced-order stochastic source-sink model comparing the FCC 5-year and
IADC 25-year post-mission disposal rules in LEO.

Populations are grounded in a CelesTrak SATCAT snapshot (2026-07-21,
currently-on-orbit objects). Fragment yields derive from the NASA Standard
Breakup Model. The collision calibration constant is set so the modeled
IADC threshold reproduces the ~90% congested-LEO stability benchmark.
All randomness is seeded for exact reproducibility.
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0
SHELL_THICKNESS_KM = 100.0
CROSS_SECTION_KM2 = 10.0 / 1e6
REL_VELOCITY_KM_YR = 9.6 * 3.15e7
HORIZON_YEARS = 200
LAUNCH_RATE = 250
CASCADE_MULTIPLE = 10
DECAY_REF_LIFETIME_YR = 6.3
DECAY_REF_ALTITUDE_KM = 500.0
DECAY_SCALE_HEIGHT_KM = 70.0
CLUSTER_FLOOR = 0.0645
CLUSTER_AMPLITUDE = 0.0944
CLUSTER_SCALE_KM = 47.5
COLLISION_CALIBRATION = 0.7
FRAGMENT_YIELDS = {"active": 330, "rocketbody": 1240, "debris": 170}

# CelesTrak SATCAT snapshot 2026-07-21 (on-orbit; see data/)
POPULATION = {500: 5212, 550: 2121, 600: 1044, 650: 910, 700: 961,
              750: 1179, 800: 1551, 850: 1446, 900: 903}
CLASS_FRACTIONS = {
    500: (0.9762, 0.0067, 0.0171), 550: (0.9052, 0.0189, 0.0759),
    600: (0.6437, 0.0805, 0.2759), 650: (0.4187, 0.0363, 0.5451),
    700: (0.1113, 0.0499, 0.8387), 750: (0.0416, 0.0755, 0.8830),
    800: (0.1161, 0.0426, 0.8414), 850: (0.0450, 0.0512, 0.9039),
    900: (0.0687, 0.0310, 0.9003),
}


def _nearest(a, t):
    return t[min(t, key=lambda k: abs(k - a))]


def decay_rate(a, m=1.0):
    life = DECAY_REF_LIFETIME_YR * np.exp((a - DECAY_REF_ALTITUDE_KM) / DECAY_SCALE_HEIGHT_KM) * m
    return 1.0 / life


def clustering_factor(a, m=1.0):
    c = np.clip(a, 675, 950)
    return (CLUSTER_FLOOR + CLUSTER_AMPLITUDE * np.exp(-(c - 675.0) / CLUSTER_SCALE_KM)) * m


def collision_coefficient(a, m=1.0):
    V = 4 * np.pi * (EARTH_RADIUS_KM + a) ** 2 * SHELL_THICKNESS_KM
    return (CROSS_SECTION_KM2 * REL_VELOCITY_KM_YR) / (2 * V) * clustering_factor(a, m) * COLLISION_CALIBRATION


def cascade_probability(compliance, deorbit_years, altitude_km, runs, rng,
                        disposal_success=1.0, avoidance=0.0, launch_multiplier=1.0,
                        population_multiplier=1.0, clustering_multiplier=1.0,
                        decay_multiplier=1.0, fragment_multiplier=1.0,
                        active_decays=True):
    """Fraction of `runs` realizations that cascade (vectorized across runs)."""
    n0 = _nearest(altitude_km, POPULATION) * population_multiplier
    fa, fr, fd = _nearest(altitude_km, CLASS_FRACTIONS)
    R = runs
    active = np.full(R, n0 * fa)
    rb = np.full(R, n0 * fr)
    debris = np.full(R, n0 * fd)
    compliant = np.zeros(R)
    abandoned = np.zeros(R)

    dr = decay_rate(altitude_km, decay_multiplier)
    clearance = max(1.0 / deorbit_years, dr)
    coeff = collision_coefficient(altitude_km, clustering_multiplier)
    threshold = n0 * CASCADE_MULTIPLE
    launch = LAUNCH_RATE * launch_multiplier
    active_dr = dr if active_decays else 0.0
    ya = FRAGMENT_YIELDS["active"] * fragment_multiplier
    yr = FRAGMENT_YIELDS["rocketbody"] * fragment_multiplier
    yd = FRAGMENT_YIELDS["debris"] * fragment_multiplier

    cascaded = np.zeros(R, dtype=bool)
    for _ in range(HORIZON_YEARS):
        total = active + compliant + abandoned + rb + debris
        cascaded |= total > threshold
        live = ~cascaded
        if not live.any():
            break
        launched = rng.poisson(launch, R)
        retiring = np.minimum(rng.poisson(launch, R), active.astype(int))
        intending = rng.binomial(retiring, compliance)
        succeeding = rng.binomial(intending, disposal_success)
        newly_abandoned = retiring - succeeding
        rate = coeff * total ** 2
        if avoidance > 0:
            p_act = np.divide(active, total, out=np.zeros_like(active), where=total > 0)
            rate = rate * (1 - avoidance * (1 - (1 - p_act) ** 2))
        collisions = rng.poisson(np.minimum(rate, 1e7))
        mean_yield = np.divide(ya * active + yr * rb + yd * (debris + abandoned + compliant),
                               total, out=np.full(R, float(ya)), where=total > 0)
        active = np.maximum(active + launched - succeeding - newly_abandoned - active * active_dr, 0)
        rb = np.maximum(rb - rb * dr, 0)
        debris = np.maximum(debris - debris * dr + collisions * mean_yield, 0)
        compliant = np.maximum(compliant + succeeding - compliant * clearance, 0)
        abandoned = np.maximum(abandoned + newly_abandoned - abandoned * dr, 0)
    return cascaded.mean()


def critical_compliance(deorbit_years, altitude_km, runs, rng,
                        cutoff=0.5, step=0.025, **kw):
    """Lowest compliance with cascade probability below `cutoff`; >1 if none."""
    for c in np.clip(np.arange(0, 1.0001, step), 0, 1.0):
        if cascade_probability(c, deorbit_years, altitude_km, runs, rng, **kw) < cutoff:
            return c
    return 1.05


def critical_compliance_multiseed(deorbit_years, altitude_km, runs,
                                  n_seeds=15, base_seed=42, **kw):
    streams = np.random.SeedSequence(base_seed).spawn(n_seeds)
    return np.array([critical_compliance(deorbit_years, altitude_km, runs,
                                         np.random.default_rng(s), **kw)
                     for s in streams])


def population_growth(compliance, deorbit_years, altitude_km, runs, rng,
                      launch_multiplier=1.0, **kw):
    """Mean final/initial population ratio and mean collision count over the horizon."""
    n0 = _nearest(altitude_km, POPULATION)
    fa, fr, fd = _nearest(altitude_km, CLASS_FRACTIONS)
    R = runs
    active = np.full(R, n0 * fa)
    rb = np.full(R, n0 * fr)
    debris = np.full(R, n0 * fd)
    compliant = np.zeros(R)
    abandoned = np.zeros(R)
    dr = decay_rate(altitude_km)
    clearance = max(1.0 / deorbit_years, dr)
    coeff = collision_coefficient(altitude_km)
    launch = LAUNCH_RATE * launch_multiplier
    ya, yr, yd = (FRAGMENT_YIELDS["active"], FRAGMENT_YIELDS["rocketbody"],
                  FRAGMENT_YIELDS["debris"])
    total_collisions = np.zeros(R)
    for _ in range(HORIZON_YEARS):
        total = active + compliant + abandoned + rb + debris
        launched = rng.poisson(launch, R)
        retiring = np.minimum(rng.poisson(launch, R), active.astype(int))
        intending = rng.binomial(retiring, compliance)
        newly_abandoned = retiring - intending
        collisions = rng.poisson(np.minimum(coeff * total ** 2, 1e7))
        total_collisions += collisions
        mean_yield = np.divide(ya * active + yr * rb + yd * (debris + abandoned + compliant),
                               total, out=np.full(R, float(ya)), where=total > 0)
        active = np.maximum(active + launched - intending - newly_abandoned - active * dr, 0)
        rb = np.maximum(rb - rb * dr, 0)
        debris = np.maximum(debris - debris * dr + collisions * mean_yield, 0)
        compliant = np.maximum(compliant + intending - compliant * clearance, 0)
        abandoned = np.maximum(abandoned + newly_abandoned - abandoned * dr, 0)
    final = active + compliant + abandoned + rb + debris
    return (final / (n0 if n0 > 0 else 1)).mean(), total_collisions.mean()
