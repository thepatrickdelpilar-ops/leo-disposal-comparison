"""Coupled nine-shell variant: objects leaving a shell by drag enter the shell
below, and a share of each collision's fragments is deposited in neighbouring
shells. Used only as a sensitivity test of the independent-shell assumption."""

import numpy as np
import model as M

SHELLS = np.array(M.SHELLS, dtype=float)
SPACING_KM = 50.0


def simulate(compliance, deorbit_years, runs, rng, frag_spread=0.0,
             decay_transport=True, launch_rates=None, multiple=None):
    S, R, T = len(SHELLS), runs, int(deorbit_years)
    n0 = np.array([M.POPULATION[int(a)] for a in SHELLS], dtype=float)
    fr = np.array([M.CLASS_FRACTIONS[int(a)] for a in SHELLS])
    active = np.repeat((n0 * fr[:, 0])[:, None], R, axis=1)
    rb = np.repeat((n0 * fr[:, 1])[:, None], R, axis=1)
    debris = np.repeat((n0 * fr[:, 2])[:, None], R, axis=1)
    abandoned = np.zeros((S, R))
    cohorts = np.zeros((T, S, R))
    life = np.array([M.decay_lifetime(a) for a in SHELLS])
    dr_life = (1.0 / life)[:, None]
    residence = 1.0 - np.exp(-SPACING_KM / M.DECAY_SCALE_HEIGHT_KM)
    dr = dr_life / residence if decay_transport else dr_life
    coeff = np.array([M.collision_coefficient(a) for a in SHELLS])[:, None]
    launch = np.full(S, float(M.LAUNCH_RATE)) if launch_rates is None else np.asarray(launch_rates, dtype=float)
    lam = np.repeat(launch[:, None], R, axis=1)
    threshold = (n0 * (M.CASCADE_MULTIPLE if multiple is None else multiple))[:, None]
    ya, yr, yd = (M.FRAGMENT_YIELDS[k] for k in ("active", "rocketbody", "debris"))
    kernel = np.zeros((S, S))
    for i in range(S):
        kernel[i, i] += 1 - frag_spread
        for off, w in ((1, 1 / 3), (-1, 1 / 3), (2, 1 / 6), (-2, 1 / 6)):
            if 0 <= i + off < S:
                kernel[i + off, i] += frag_spread * w
    exceeded = np.zeros((S, R), dtype=bool)
    selfsust = np.zeros((S, R), dtype=bool)
    for _ in range(M.HORIZON_YEARS):
        pool = cohorts.sum(axis=0)
        total = active + pool + abandoned + rb + debris
        exceeded |= total > threshold
        rate = coeff * total ** 2
        mean_yield = np.divide(ya * active + yr * rb + yd * (debris + abandoned + pool), total,
                               out=np.full((S, R), float(ya)), where=total > 0)
        selfsust |= rate * mean_yield > dr_life * total
        launched = rng.poisson(lam)
        retiring = np.minimum(rng.poisson(lam), active.astype(int))
        succeeding = rng.binomial(retiring, compliance)
        newly_abandoned = retiring - succeeding
        new_frag = kernel @ (rng.poisson(np.minimum(rate, 1e7)) * mean_yield)
        out_d, out_r, out_b = debris * dr, rb * dr, abandoned * dr

        def inflow(x):
            if not decay_transport:
                return 0.0
            y = np.zeros_like(x)
            y[:-1] = x[1:]
            return y

        active = np.maximum(active + launched - retiring - active * dr_life, 0)
        rb = np.maximum(rb - out_r + inflow(out_r), 0)
        debris = np.maximum(debris - out_d + inflow(out_d) + new_frag, 0)
        abandoned = np.maximum(abandoned + newly_abandoned - out_b + inflow(out_b), 0)
        cohorts = cohorts * (1 - dr_life)
        cohorts = np.roll(cohorts, 1, axis=0)
        cohorts[0] = succeeding
        if exceeded.all():
            break
    return exceeded.mean(axis=1), selfsust.mean(axis=1)


def thresholds(deorbit_years, runs, rng, **kw):
    """Per-shell critical compliance when every shell shares the same compliance."""
    pe, ps = [], []
    for c in M.GRID:
        e, s = simulate(c, deorbit_years, runs, rng, **kw)
        pe.append(e); ps.append(s)
    pe, ps = np.array(pe), np.array(ps)
    return ([M.threshold_from_curve(pe[:, i]) for i in range(len(SHELLS))],
            [M.threshold_from_curve(ps[:, i]) for i in range(len(SHELLS))])
