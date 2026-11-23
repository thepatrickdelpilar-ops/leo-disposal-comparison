"""
Reduced-order stochastic source-sink model comparing the FCC 5-year and
IADC 25-year post-mission disposal rules in LEO.

Populations are grounded in a CelesTrak SATCAT snapshot (2026-07-21,
currently-on-orbit objects). Fragment yields derive from the NASA Standard
Breakup Model. The collision calibration constant is set so the modeled
IADC threshold reproduces the ~90% congested-LEO stability benchmark.
All randomness is seeded for exact reproducibility.

Revision 2: compliant objects are removed exactly T years after retirement
(cohort queue) by default, and every run is scored against two criteria:
population exceedance (N > mu * N0) and self-sustaining collisional growth
(expected fragment production above natural-decay removal).
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
CLASS_MASS_KG = {"active": 250.0, "retired": 250.0, "rocketbody": 1500.0, "debris": 100.0}
CLASS_RADIUS_M = {"active": 1.5, "retired": 1.5, "rocketbody": 2.5, "debris": 0.3}
SOLAR_PERIOD_YR = 11.0
SOLAR_AMPLITUDE = 3.0
UNTRACKED_RATIO = 30.0

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
SHELLS = sorted(POPULATION)


def _nearest(a, t):
    return t[min(t, key=lambda k: abs(k - a))]


def decay_lifetime(a, m=1.0):
    return DECAY_REF_LIFETIME_YR * np.exp((a - DECAY_REF_ALTITUDE_KM) / DECAY_SCALE_HEIGHT_KM) * m


def decay_rate(a, m=1.0):
    return 1.0 / decay_lifetime(a, m)


def clustering_factor(a, m=1.0):
    c = np.clip(a, 675, 950)
    return (CLUSTER_FLOOR + CLUSTER_AMPLITUDE * np.exp(-(c - 675.0) / CLUSTER_SCALE_KM)) * m


def collision_coefficient(a, m=1.0):
    """Collisions per year per squared object count (dimension 1/yr)."""
    V = 4 * np.pi * (EARTH_RADIUS_KM + a) ** 2 * SHELL_THICKNESS_KM
    return (CROSS_SECTION_KM2 * REL_VELOCITY_KM_YR) / (2 * V) * clustering_factor(a, m) * COLLISION_CALIBRATION


def sbm_fragments(mass_kg, lc_m=0.10):
    """NASA Standard Breakup Model, catastrophic collision: fragments larger than lc_m."""
    return 0.1 * mass_kg ** 0.75 * lc_m ** -1.71


def solar_mean_rate_factor(amplitude=SOLAR_AMPLITUDE):
    """Cycle-averaged decay-rate multiplier for the periodic solar model."""
    x = np.linspace(0, 2 * np.pi, 20001)[:-1]
    return float(np.mean(amplitude ** (-np.sin(x))))


_CLASSES = ("active", "retired", "rocketbody", "debris")
_PAIR_YIELD = np.array([[sbm_fragments(CLASS_MASS_KG[i] + CLASS_MASS_KG[j]) for j in _CLASSES]
                        for i in _CLASSES])
_PAIR_WEIGHT = np.array([[np.pi * (CLASS_RADIUS_M[i] + CLASS_RADIUS_M[j]) ** 2 / 10.0 for j in _CLASSES]
                         for i in _CLASSES])


def _pair_norm():
    n0 = POPULATION[800]
    fa, fr, fd = CLASS_FRACTIONS[800]
    v = np.array([fa, 0.0, fr, fd]) * n0
    return n0 ** 2 / float(v @ _PAIR_WEIGHT @ v)


def simulate(compliance, deorbit_years, altitude_km, runs, rng,
             deadline="exact", disposal_success=1.0, avoidance=0.0,
             launch_multiplier=1.0, launch_rate=None,
             population_multiplier=1.0, clustering_multiplier=1.0,
             decay_multiplier=1.0, fragment_multiplier=1.0,
             active_decays=True, solar_cycle=False, collisions=True,
             pairwise=False, class_cross_sections=False, parent_removal=False,
             success_concentration=None, untracked=False,
             class_fractions=None, multiple=None, population=None,
             external_debris_inflow=0.0):
    """Run `runs` realizations of one shell; return both criteria and diagnostics."""
    n0 = (population if population is not None else _nearest(altitude_km, POPULATION)) * population_multiplier
    fa, fr, fd = class_fractions if class_fractions is not None else _nearest(altitude_km, CLASS_FRACTIONS)
    R = runs
    T = int(deorbit_years)
    active = np.full(R, n0 * fa)
    rb = np.full(R, n0 * fr)
    debris = np.full(R, n0 * fd)
    abandoned = np.zeros(R)
    exact = deadline == "exact"
    cohorts = np.zeros((T, R)) if exact else None
    pool = np.zeros(R)
    frag = np.zeros(R)
    unt = np.full(R, UNTRACKED_RATIO * n0 * fd) if untracked else None

    dr0 = decay_rate(altitude_km, decay_multiplier)
    if solar_cycle:
        amp = SOLAR_AMPLITUDE if solar_cycle is True else float(solar_cycle)
        phase = rng.uniform(0, SOLAR_PERIOD_YR, R)
        dr_ref = dr0 * solar_mean_rate_factor(amp)
    else:
        dr_ref = dr0
    coeff = collision_coefficient(altitude_km, clustering_multiplier) if collisions else 0.0
    mult = CASCADE_MULTIPLE if multiple is None else multiple
    threshold = n0 * mult
    launch = (LAUNCH_RATE if launch_rate is None else launch_rate) * launch_multiplier
    ya = FRAGMENT_YIELDS["active"] * fragment_multiplier
    yr = FRAGMENT_YIELDS["rocketbody"] * fragment_multiplier
    yd = FRAGMENT_YIELDS["debris"] * fragment_multiplier
    pair_y = _PAIR_YIELD * fragment_multiplier
    norm = _pair_norm() if class_cross_sections else 1.0
    W = _PAIR_WEIGHT if class_cross_sections else np.ones((4, 4))

    exceeded = np.zeros(R, dtype=bool)
    selfsust = np.zeros(R, dtype=bool)
    share_at_cross = np.full(R, np.nan)
    total_collisions = np.zeros(R)

    for t in range(HORIZON_YEARS):
        if exact:
            pool = cohorts.sum(axis=0)
        total = active + pool + abandoned + rb + debris
        newly = (total > threshold) & ~exceeded
        share_at_cross[newly] = frag[newly] / total[newly]
        exceeded |= newly

        dr = dr0 / amp ** np.sin(2 * np.pi * (t + phase) / SOLAR_PERIOD_YR) if solar_cycle else dr0
        clearance = np.maximum(1.0 / T, dr)
        active_dr = dr if active_decays else 0.0

        if pairwise or class_cross_sections:
            v = np.stack([active, pool + abandoned, rb, debris])
            wv = np.einsum("ij,jr->ir", W, v)
            pair_sum = (v * wv).sum(axis=0)
            rate = coeff * norm * pair_sum
            safe = np.where(pair_sum > 0, pair_sum, 1.0)
            if pairwise:
                mean_yield = (v * np.einsum("ij,jr->ir", W * pair_y, v)).sum(axis=0) / safe
            else:
                share = v * wv / safe
                mean_yield = share[0] * ya + share[2] * yr + (share[1] + share[3]) * yd
            part = v * wv / safe
        else:
            rate = coeff * total ** 2
            mean_yield = np.divide(ya * active + yr * rb + yd * (debris + abandoned + pool),
                                   total, out=np.full(R, float(ya)), where=total > 0)
            part = None
        if avoidance > 0:
            if part is not None:
                p_inv = (2 * v[0] * wv[0] - W[0, 0] * v[0] ** 2) / safe
                rate = rate * (1 - avoidance * p_inv)
            else:
                p_act = np.divide(active, total, out=np.zeros_like(active), where=total > 0)
                rate = rate * (1 - avoidance * (1 - (1 - p_act) ** 2))
        expected_fragments = rate * mean_yield
        if untracked:
            disable_p = np.minimum(2 * coeff * unt, 1.0)
        selfsust |= expected_fragments > dr_ref * total

        launched = rng.poisson(launch, R)
        retiring = np.minimum(rng.poisson(launch, R), active.astype(int))
        intending = rng.binomial(retiring, compliance)
        if success_concentration is None:
            s = disposal_success
        else:
            a_ = disposal_success * success_concentration
            b_ = (1 - disposal_success) * success_concentration
            s = rng.beta(a_, b_, R) if 0 < disposal_success < 1 else disposal_success
        succeeding = rng.binomial(intending, s)
        newly_abandoned = retiring - succeeding
        collisions_n = rng.poisson(np.minimum(rate, 1e7))
        total_collisions += collisions_n
        new_frag = collisions_n * mean_yield

        loss = np.zeros((4, R))
        if parent_removal:
            if part is None:
                tot_safe = np.where(total > 0, total, 1.0)
                part_simple = np.stack([active, pool + abandoned, rb, debris]) / tot_safe
                loss = 2 * collisions_n * part_simple
            else:
                loss = 2 * collisions_n * part
        disabled = 0.0
        if untracked:
            disabled = rng.binomial(np.maximum(active - retiring, 0).astype(int), disable_p)

        ret_pool = pool + abandoned
        ret_safe = np.where(ret_pool > 0, ret_pool, 1.0)
        active = np.maximum(active + launched - retiring - active * active_dr - loss[0] - disabled, 0)
        rb = np.maximum(rb - rb * dr - loss[2], 0)
        debris = np.maximum(debris - debris * dr + new_frag - loss[3] + external_debris_inflow, 0)
        frag = np.maximum(frag - frag * dr + new_frag, 0)
        abandoned = np.maximum(abandoned + newly_abandoned + disabled - abandoned * dr
                               - loss[1] * abandoned / ret_safe, 0)
        if exact:
            keep = 1 - dr - (loss[1] / ret_safe if parent_removal else 0.0)
            cohorts = np.maximum(cohorts * keep, 0)
            cohorts = np.roll(cohorts, 1, axis=0)
            cohorts[0] = succeeding
        else:
            pool = np.maximum(pool + succeeding - pool * clearance - loss[1] * pool / ret_safe, 0)
        if untracked:
            unt = np.maximum(unt - unt * dr + UNTRACKED_RATIO * new_frag, 0)
        if exceeded.all() and selfsust.all():
            break

    if exact:
        pool = cohorts.sum(axis=0)
    final = active + pool + abandoned + rb + debris
    return {
        "exceed": float(exceeded.mean()),
        "selfsust": float(selfsust.mean()),
        "fragment_share_at_crossing": float(np.nanmean(share_at_cross)) if exceeded.any() else float("nan"),
        "growth": float((final / n0).mean()) if n0 > 0 else float("nan"),
        "collisions": float(total_collisions.mean()),
    }


def cascade_probability(compliance, deorbit_years, altitude_km, runs, rng,
                        criterion="exceed", **kw):
    """Fraction of `runs` realizations flagged under `criterion`."""
    return simulate(compliance, deorbit_years, altitude_km, runs, rng, **kw)[criterion]


GRID = np.clip(np.arange(0, 1.0001, 0.025), 0, 1.0)


def probability_curve(deorbit_years, altitude_km, runs, rng, grid=GRID, **kw):
    """Both probabilities at every compliance level on `grid`."""
    out = [simulate(c, deorbit_years, altitude_km, runs, rng, **kw) for c in grid]
    return (np.array([o["exceed"] for o in out]), np.array([o["selfsust"] for o in out]))


def threshold_from_curve(p, grid=GRID, cutoff=0.5):
    """Lowest compliance with probability below `cutoff`; 1.05 if none."""
    below = np.nonzero(p < cutoff)[0]
    return float(grid[below[0]]) if below.size else 1.05


def critical_compliance(deorbit_years, altitude_km, runs, rng,
                        cutoff=0.5, step=0.025, criterion="exceed", **kw):
    """Lowest compliance with probability below `cutoff`; >1 if none."""
    for c in np.clip(np.arange(0, 1.0001, step), 0, 1.0):
        if cascade_probability(c, deorbit_years, altitude_km, runs, rng, criterion=criterion, **kw) < cutoff:
            return float(c)
    return 1.05


def critical_both(deorbit_years, altitude_km, runs, rng, cutoff=0.5, grid=GRID, **kw):
    """Thresholds under both criteria: the lowest grid level with probability below
    `cutoff`, located by bisection and confirmed against the two levels beneath it."""
    cache = {}

    def p(i, key):
        if i not in cache:
            cache[i] = simulate(float(grid[i]), deorbit_years, altitude_km, runs, rng, **kw)
        return cache[i][key]

    def find(key):
        n = len(grid)
        if p(n - 1, key) >= cutoff:
            return 1.05
        if p(0, key) < cutoff:
            return float(grid[0])
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if p(mid, key) < cutoff:
                hi = mid
            else:
                lo = mid
        for j in (hi - 1, hi - 2):
            if j >= 0 and p(j, key) < cutoff:
                hi = j
        return float(grid[hi])

    return find("exceed"), find("selfsust")


def critical_compliance_multiseed(deorbit_years, altitude_km, runs,
                                  n_seeds=15, base_seed=42, **kw):
    streams = np.random.SeedSequence(base_seed).spawn(n_seeds)
    return np.array([critical_compliance(deorbit_years, altitude_km, runs,
                                         np.random.default_rng(s), **kw)
                     for s in streams])


def critical_population(altitude_km, mean_yield, decay_multiplier=1.0, clustering_multiplier=1.0):
    """Population above which expected fragment production exceeds drag removal."""
    return 1.0 / (collision_coefficient(altitude_km, clustering_multiplier) * mean_yield
                  * decay_lifetime(altitude_km, decay_multiplier))


def population_growth(compliance, deorbit_years, altitude_km, runs, rng,
                      launch_multiplier=1.0, **kw):
    """Mean final/initial population ratio and mean collision count over the horizon."""
    o = simulate(compliance, deorbit_years, altitude_km, runs, rng,
                 launch_multiplier=launch_multiplier, **kw)
    return o["growth"], o["collisions"]
