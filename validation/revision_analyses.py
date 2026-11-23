"""Analyses added in revision 2. Usage: python validation/revision_analyses.py <section> [...]
Each section writes validation/logs/rev2/<section>.json."""
import sys
import os
import json
import itertools
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import model as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "rev2")
ALT25 = [int(a) for a in np.arange(500, 901, 25)]
RULES = {"FCC": 5, "IADC": 25}
RUNS = 1000


def rng(seed=42):
    return np.random.default_rng(seed)


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(obj, fh, indent=1)
    print("wrote", name, flush=True)


def both(T, a, runs=RUNS, seed=42, **kw):
    e, s = M.critical_both(T, a, runs, rng(seed), **kw)
    return {"exceed": e, "selfsust": s}


def sweep(alts=ALT25, runs=RUNS, seed=42, **kw):
    return {r: {str(a): both(T, a, runs, seed, **kw) for a in alts} for r, T in RULES.items()}


def headline():
    res = {}
    for dl, n in (("exact", 15), ("rate", 5)):
        streams = np.random.SeedSequence(42).spawn(n)
        res[dl] = {}
        for r, T in RULES.items():
            res[dl][r] = {}
            for a in ALT25:
                te, ts = [], []
                for s in streams:
                    e, sst = M.critical_both(T, a, RUNS, np.random.default_rng(s), deadline=dl)
                    te.append(e); ts.append(sst)
                res[dl][r][str(a)] = {"exceed": te, "selfsust": ts}
                print(dl, r, a, np.median(te), np.median(ts), flush=True)
        save("headline", res)
    comp = np.round(np.arange(0, 1.001, 0.1), 1)
    heat = {"compliance": comp.tolist(), "altitudes": list(M.SHELLS)}
    for r, T in RULES.items():
        pe, ps = [], []
        for a in M.SHELLS:
            e, s2 = M.probability_curve(T, a, 500, rng(), grid=comp)
            pe.append(e.tolist()); ps.append(s2.tolist())
        heat[r] = {"exceed": pe, "selfsust": ps}
    res["heatmap"] = heat
    save("headline", res)


def control():
    res = {"collisionless": sweep(collisions=False)}
    share = {}
    base = sweep(alts=list(M.SHELLS))
    for r, T in RULES.items():
        share[r] = {}
        for a in M.SHELLS:
            th = base[r][str(a)]["exceed"]
            c = max(min(th, 1.0) - 0.025, 0.0)
            o = M.simulate(c, T, a, 2000, rng())
            o0 = M.simulate(0.0, T, a, 2000, rng())
            share[r][str(a)] = {"threshold": th, "c_probe": c, "share_below_threshold": o["fragment_share_at_crossing"],
                                "p_exceed_probe": o["exceed"], "share_zero_compliance": o0["fragment_share_at_crossing"],
                                "growth_zero": o0["growth"], "collisions_zero": o0["collisions"]}
    res["fragment_share"] = share
    res["critical_population_multiple"] = {}
    for a in M.SHELLS:
        fa, fr, fd = M.CLASS_FRACTIONS[a]
        y = 330 * fa + 1240 * fr + 170 * fd
        res["critical_population_multiple"][str(a)] = {
            "initial_mix": M.critical_population(a, y) / M.POPULATION[a],
            "debris_yield": M.critical_population(a, 170) / M.POPULATION[a]}
    save("control", res)


def boundary():
    alts = [int(a) for a in np.arange(600, 731, 5)]
    save("boundary", {"exact": sweep(alts=alts, runs=2000), "rate": sweep(alts=alts, runs=2000, deadline="rate")})


def solar():
    res = {}
    for label, kw in (("solar_max_static", dict(decay_multiplier=0.3)), ("nominal", {}),
                      ("solar_min_static", dict(decay_multiplier=3.0)), ("cycle", dict(solar_cycle=True)),
                      ("cycle_amp5", dict(solar_cycle=5.0))):
        res[label] = sweep(**kw)
        print(label, flush=True)
    res["cycle_mean_rate_factor"] = M.solar_mean_rate_factor()
    alts = [int(a) for a in np.arange(560, 731, 5)]
    res["cycle_fine"] = sweep(alts=alts, runs=2000, solar_cycle=True)
    save("solar", res)


def msis_lifetimes():
    import pymsis
    MU = 3.986004418e14
    hs = np.arange(200.0, 901.0, 2.5)
    lats = np.array([-75, -45, -15, 15, 45, 75.0]); lons = np.arange(0, 360, 45.0)
    w = np.cos(np.radians(lats))[None, :, None]

    def rho(f107, ap):
        tot = 0
        dates = [np.datetime64("2026-03-21T00:00"), np.datetime64("2026-06-21T00:00"),
                 np.datetime64("2026-09-21T00:00"), np.datetime64("2026-12-21T00:00")]
        for d in dates:
            o = pymsis.calculate(np.array([d]), lons, lats, hs, [f107], [f107], [[ap] * 7], version=0)
            tot = tot + (o[0, :, :, :, 0] * w).sum(axis=(0, 1)) / (w.sum() * len(lons))
        return tot / len(dates)

    levels = {"low": (70, 5), "moderate": (140, 12), "high": (210, 20)}
    dens = {k: rho(*v) for k, v in levels.items()}
    ph = np.linspace(0, 2 * np.pi, 23)[:-1]
    cyc = np.mean([rho(140 + 70 * np.sin(p), 12) for p in ph], axis=0)
    dens["cycle_mean"] = cyc
    YR = 3.15576e7
    res = {"altitudes": [int(a) for a in M.SHELLS] + [625, 675], "fit": {}, "msis": {}, "density_kg_m3": {}}
    for a in res["altitudes"]:
        res["fit"][str(a)] = float(M.decay_lifetime(a))
    for k, d in dens.items():
        res["density_kg_m3"][k] = {str(int(a)): float(d[np.argmin(abs(hs - a))]) for a in M.SHELLS}
        res["msis"][k] = {}
        for am in (0.005, 0.01, 0.02, 0.05, 0.1):
            B = 2.2 * am
            integrand = 1.0 / (d * B * np.sqrt(MU * (6371e3 + hs * 1e3)))
            cum = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(hs) * 1e3)]) / YR
            res["msis"][k][str(am)] = {str(a): float(np.interp(a, hs, cum)) for a in res["altitudes"]}
    save("msis_lifetimes", res)
    return res


def msis():
    msis_lifetimes()
    full = json.load(open(os.path.join(OUT, "msis_lifetimes.json")))
    res = {"thresholds": {}}

    def mult(am, a):
        tab = full["msis"]["cycle_mean"][am]
        known = sorted(int(k) for k in tab)
        tau = float(np.exp(np.interp(a, known, [np.log(tab[str(k)]) for k in known])))
        return tau, tau / M.decay_lifetime(a)

    for am in ("0.005", "0.01", "0.02"):
        res["thresholds"][am] = {r: {} for r in RULES}
        for a in ALT25:
            tau, m = mult(am, a)
            for r, T in RULES.items():
                res["thresholds"][am][r][str(a)] = dict(both(T, a, decay_multiplier=m), lifetime=tau, multiplier=m)
        print("msis", am, flush=True)
    am = "0.01"
    _, m800 = mult(am, 800)
    cal = None
    for cm in np.arange(1.0, 12.01, 0.25):
        th = M.critical_compliance(25, 800, RUNS, rng(), decay_multiplier=m800, clustering_multiplier=float(cm))
        if th >= 0.90:
            cal = float(cm)
            break
    res["recalibrated"] = {"area_to_mass": am, "kappa": cal * M.COLLISION_CALIBRATION, "thresholds": {r: {} for r in RULES}}
    for a in ALT25:
        _, m = mult(am, a)
        for r, T in RULES.items():
            res["recalibrated"]["thresholds"][r][str(a)] = both(T, a, decay_multiplier=m, clustering_multiplier=cal)
    save("msis", res)


def coupled():
    import coupled as C
    active = {a: M.POPULATION[a] * M.CLASS_FRACTIONS[a][0] for a in M.SHELLS}
    tot = sum(active.values())
    obs = [2250 * active[a] / tot for a in M.SHELLS]
    variants = {
        "independent": dict(decay_transport=False),
        "transport": dict(),
        "spread_0.3": dict(decay_transport=False, frag_spread=0.3),
        "transport_spread_0.3": dict(frag_spread=0.3),
        "transport_spread_0.6": dict(frag_spread=0.6),
        "observed_launch_independent": dict(decay_transport=False, launch_rates=obs),
        "observed_launch_transport_spread_0.3": dict(frag_spread=0.3, launch_rates=obs),
    }
    res = {"shells": M.SHELLS}
    for name, kw in variants.items():
        res[name] = {}
        for r, T in RULES.items():
            e, s = C.thresholds(T, RUNS, rng(), **kw)
            res[name][r] = {"exceed": e, "selfsust": s}
        print(name, res[name], flush=True)
    save("coupled", res)


def launchdist():
    active = {a: M.POPULATION[a] * M.CLASS_FRACTIONS[a][0] for a in M.SHELLS}
    tot = sum(active.values())
    rates = {a: 2250 * active[a] / tot for a in M.SHELLS}
    res = {"rates": {str(a): rates[a] for a in M.SHELLS}, "thresholds": {r: {} for r in RULES}}
    streams = np.random.SeedSequence(42).spawn(5)
    for r, T in RULES.items():
        for a in M.SHELLS:
            te, ts = [], []
            for s in streams:
                e, sst = M.critical_both(T, a, RUNS, np.random.default_rng(s), launch_rate=rates[a])
                te.append(e); ts.append(sst)
            res["thresholds"][r][str(a)] = {"exceed": te, "selfsust": ts}
            print(r, a, te, ts, flush=True)
    save("launchdist", res)


def floor_at(a, runs=2000, **kw):
    out = {r: {} for r in RULES}
    for s in np.clip(np.arange(0.5, 1.001, 0.05), 0, 1):
        for r, T in RULES.items():
            out[r][f"{s:.2f}"] = both(T, a, runs, disposal_success=float(s), **kw)
    return out


def min_success(fl, crit="exceed"):
    ok = [float(s) for s, v in fl.items() if v[crit] <= 1.0]
    return min(ok) if ok else None


def kappa():
    res = {}
    for k in (0.35, 0.5, 0.7, 1.0, 1.4):
        cm = k / M.COLLISION_CALIBRATION
        sw = sweep(clustering_multiplier=cm)
        fl = floor_at(800, runs=1000, clustering_multiplier=cm)
        res[str(k)] = {"sweep": sw, "floor_800": fl,
                       "min_success_800": {r: min_success(fl[r]) for r in RULES}}
        print("kappa", k, flush=True)
    save("kappa", res)


def pairwise():
    variants = {
        "baseline": {},
        "parent_removal": dict(parent_removal=True),
        "pairwise_yields": dict(pairwise=True, parent_removal=True),
        "class_cross_sections": dict(class_cross_sections=True),
        "pairwise_and_cross_sections": dict(pairwise=True, class_cross_sections=True, parent_removal=True),
    }
    res = {"pair_yield": M._PAIR_YIELD.tolist(), "pair_weight": M._PAIR_WEIGHT.tolist(), "norm": M._pair_norm()}
    for name, kw in variants.items():
        res[name] = sweep(**kw)
        print(name, flush=True)
    save("pairwise", res)


def floor():
    res = {"800": floor_at(800), "700": floor_at(700, runs=1000), "900": floor_at(900, runs=1000),
           "800_beta20": floor_at(800, success_concentration=20.0),
           "800_beta5": floor_at(800, success_concentration=5.0)}
    save("floor", res)


def avoid():
    alphas = [0, 0.2, 0.4, 0.6, 0.8, 0.95]
    cases = {"800": dict(a=800), "650": dict(a=650), "550": dict(a=550),
             "800_active_dominated": dict(a=800, class_fractions=(0.90, 0.02, 0.08)),
             "800_half_active": dict(a=800, class_fractions=(0.50, 0.04, 0.46))}
    res = {}
    for name, cs in cases.items():
        a = cs.pop("a")
        res[name] = {r: {str(al): both(T, a, 2000, avoidance=al, **cs) for al in alphas} for r, T in RULES.items()}
        print(name, flush=True)
    save("avoid", res)


def matrix():
    levels = {"launch_multiplier": [0.5, 1.0, 2.0], "population_multiplier": [0.75, 1.0, 1.25],
              "clustering_multiplier": [0.75, 1.0, 1.25], "fragment_multiplier": [0.75, 1.0, 1.25]}
    keys = list(levels)
    rows = []
    for combo in itertools.product(*(levels[k] for k in keys)):
        kw = dict(zip(keys, combo))
        rows.append({"params": kw, "FCC": both(5, 800, **kw), "IADC": both(25, 800, **kw)})
    oat = {}
    for p in ("clustering_multiplier", "population_multiplier", "decay_multiplier", "launch_multiplier", "fragment_multiplier"):
        oat[p] = {str(m): {r: both(T, 800, **{p: m}) for r, T in RULES.items()} for m in (0.75, 1.0, 1.25)}
    save("matrix", {"matrix": rows, "oat": oat})


def misc():
    res = {}
    res["convergence"] = {str(a): {r: {str(n): both(T, a, n) for n in (100, 200, 500, 1000, 2000)}
                                   for r, T in RULES.items()} for a in (650, 800)}
    cm = {}
    for a in (650, 800):
        cm[str(a)] = {}
        for q in (0.3, 0.5, 0.7):
            cm[str(a)][f"cutoff_{q}"] = {r: dict(zip(("exceed", "selfsust"), M.critical_both(T, a, RUNS, rng(), cutoff=q)))
                                          for r, T in RULES.items()}
        for mu in (5, 10, 20):
            cm[str(a)][f"multiple_{mu}"] = {r: both(T, a, multiple=mu) for r, T in RULES.items()}
    res["cutoff_multiple"] = cm
    res["active_decay"] = {str(a): {r: {"on": both(T, a), "off": both(T, a, active_decays=False)}
                                    for r, T in RULES.items()} for a in (650, 800)}
    res["deadline_repr"] = {str(a): {r: {"exact": both(T, a), "rate": both(T, a, deadline="rate")}
                                     for r, T in RULES.items()} for a in (650, 700, 800, 900)}
    res["disposal_time"] = {str(a): {str(T): both(T, a) for T in (1, 2, 5, 10, 15, 25)} for a in (650, 700, 800, 900)}
    res["untracked"] = {str(a): {r: {"off": both(T, a), "on": both(T, a, untracked=True)}
                                 for r, T in RULES.items()} for a in (650, 800)}
    rates = {str(a): float(M.collision_coefficient(a) * M.POPULATION[a] ** 2) for a in M.SHELLS}
    nl = {str(a): M.simulate(0.0, 25, a, 2000, rng(), launch_multiplier=0.0) for a in M.SHELLS}
    res["collision_rates"] = {"initial_per_year": rates, "initial_total_per_year": sum(rates.values()),
                              "no_launch": nl, "no_launch_total_collisions": sum(v["collisions"] for v in nl.values())}
    save("misc", res)


def stats():
    streams = np.random.SeedSequence(42).spawn(15)
    res = {}
    for a in (650, 800):
        res[str(a)] = {}
        for r, T in RULES.items():
            base, _ = M.critical_both(T, a, RUNS, rng())
            grid = np.round(np.arange(max(base - 0.12, 0), min(base + 0.08, 1.0) + 1e-9, 0.01), 2)
            pe_all, th, interp = [], [], []
            for s in streams:
                pe, _ = M.probability_curve(T, a, RUNS, np.random.default_rng(s), grid=grid)
                pe_all.append(pe)
                i = np.nonzero(pe < 0.5)[0]
                th.append(float(grid[i[0]]) if i.size else float("nan"))
                if i.size and i[0] > 0:
                    k = i[0]
                    interp.append(float(grid[k - 1] + (pe[k - 1] - 0.5) / (pe[k - 1] - pe[k]) * 0.01))
                else:
                    interp.append(float("nan"))
            pe_all = np.array(pe_all)
            rev = int(sum(((np.diff(p) > 0) & (p[:-1] < 0.5) & (p[1:] >= 0.5)).sum() for p in pe_all))
            res[str(a)][r] = {"grid": grid.tolist(), "p_mean": pe_all.mean(axis=0).tolist(),
                              "p_sd": pe_all.std(axis=0, ddof=1).tolist(),
                              "threshold_1pt": th, "threshold_interp": interp, "reversals_across_cutoff": rev}
            print(a, r, np.nanmedian(th), np.nanmedian(interp), np.nanstd(interp), rev, flush=True)
    save("stats", res)


def traffic():
    res = {}
    for a in (700, 800, 900):
        res[str(a)] = {str(L): {r: both(T, a, launch_rate=L) for r, T in RULES.items()}
                       for L in (10, 25, 50, 75, 100, 150, 200, 250, 375, 500)}
        print(a, res[str(a)], flush=True)
    save("traffic", res)


def pairwise_recal():
    variants = {
        "pairwise_yields": dict(pairwise=True, parent_removal=True),
        "class_cross_sections": dict(class_cross_sections=True),
        "pairwise_and_cross_sections": dict(pairwise=True, class_cross_sections=True, parent_removal=True),
    }
    res = {}
    for name, kw in variants.items():
        cal = None
        for cm in np.round(np.arange(1.0, 0.049, -0.05), 2):
            th = M.critical_both(25, 800, RUNS, rng(), clustering_multiplier=float(cm), **kw)[0]
            if th <= 0.90:
                cal = float(cm)
                break
        res[name] = {"kappa": cal * M.COLLISION_CALIBRATION, "sweep": sweep(clustering_multiplier=cal, **kw)}
        print(name, cal, flush=True)
    save("pairwise_recal", res)


def flat_clustering():
    g800 = float(M.clustering_factor(800))
    floor0, amp0 = M.CLUSTER_FLOOR, M.CLUSTER_AMPLITUDE
    M.CLUSTER_FLOOR, M.CLUSTER_AMPLITUDE = g800, 0.0
    res = {"gamma": g800, "sweep": sweep()}
    M.CLUSTER_FLOOR, M.CLUSTER_AMPLITUDE = floor0, amp0
    res["profile"] = {str(a): float(M.clustering_factor(a)) for a in M.SHELLS}
    save("flat_clustering", res)


SECTIONS = dict(headline=headline, control=control, boundary=boundary, solar=solar, msis=msis,
                launchdist=launchdist, kappa=kappa, pairwise=pairwise, floor=floor, avoid=avoid,
                matrix=matrix, misc=misc, stats=stats, coupled=coupled, traffic=traffic, pairwise_recal=pairwise_recal, flat_clustering=flat_clustering)

if __name__ == "__main__":
    for name in sys.argv[1:]:
        t = time.time()
        SECTIONS[name]()
        print(f"{name} done in {time.time() - t:.0f}s", flush=True)
