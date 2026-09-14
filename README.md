# Comparing the FCC 5-Year and IADC 25-Year Post-Mission Disposal Rules

Code and data accompanying *Comparing the FCC 5-Year and IADC 25-Year
Post-Mission Disposal Rules: A Reduced-Order Stochastic Model of
Altitude-Dependent Cascade Risk in LEO* by Patrick Alexander H. Del Pilar.

## Description

This repository holds everything needed to reproduce the paper: a reduced-order
stochastic source-sink model of a single LEO altitude shell, the CelesTrak
catalog snapshot that grounds its starting populations, a validation suite, and
one script per paper figure. Given an altitude, a disposal rule, and an operator
compliance level, the model estimates the probability of a collisional cascade
over a 200-year horizon and reports the critical compliance threshold at which
that probability falls below 50%. Anyone downloading this package can rerun
every figure in the paper, check the model's physical consistency for
themselves, or change the assumptions and see what happens.

## Size

Approximately 2 MB in total: about 60 KB of code, documentation, and the frozen
catalog snapshot (`data/celestrak_snapshot_20260721.json`, ~3 KB), plus the
pre-generated paper figures in `paper-figures/` (~1.9 MB of PNG and PDF files).
No single file exceeds 1 MB.

## Platform

Any platform that runs Python 3.9 or later: Linux, macOS, or Windows. No GPU,
no special hardware, no network access needed (the data snapshot is frozen and
included). Everything runs comfortably on an ordinary laptop.

## Environment

- Python 3.9+ (developed and tested on Python 3.12, Linux)
- NumPy >= 1.22 and Matplotlib >= 3.5, installed via `requirements.txt`
- No compilers, system libraries, or other dependencies

## Major Components

- `model.py` — the entire model, defined once. Grounded population tables, the
  collision calibration constant, and the core functions
  (`cascade_probability`, `critical_compliance`,
  `critical_compliance_multiseed`). Every other script imports from here.
- `data/celestrak_snapshot_20260721.json` — frozen CelesTrak SATCAT snapshot
  (accessed 2026-07-21; 26,628 on-orbit LEO objects) used to ground the shell
  populations and class fractions.
- `data/acquire_snapshot.py` — the acquisition script that produced the
  snapshot: grouped SATCAT pull, on-orbit filter, shell binning. Requires
  network access to celestrak.org; rerunning it produces a snapshot for the
  current date rather than the frozen one.
- `figures/` — one script per paper figure (headline comparison, redundancy
  boundary, disposal-reliability floor, robustness matrix, sensitivity panels,
  avoidance sweep, cascade-probability heatmaps, convergence, model inputs),
  plus shared plot styling in `_style.py`.
- `validation/` — `monotonicity.py` (eight physical-consistency checks),
  `no_launch.py` (independent no-launch growth validation), and `ablations.py`
  (solar bracket, active-decay ablation, hard-deadline ablation, and cascade
  criterion robustness). `validation/logs/` holds the console output and JSON
  results of every run used in the paper, including the full 15-seed headline
  thresholds.
- `paper-figures/` — the generated figures exactly as they appear in the paper.
- `outputs/` — created at run time for freshly generated figures (not tracked).

## Setup Instructions

1. Download or clone this repository and open a terminal in its root folder.
2. Check your Python version: `python --version` (3.9 or later).
3. Optionally create a virtual environment:
   `python -m venv venv`, then `source venv/bin/activate`
   (on Windows: `venv\Scripts\activate`).
4. Install the two dependencies: `pip install -r requirements.txt`.

That is the whole setup. The data snapshot ships with the repository, and the
`outputs/` folder is created automatically the first time a figure script runs.

## Run Instructions

All commands run from the repository root.

Validation suite:

```
python validation/monotonicity.py
python validation/no_launch.py
python validation/ablations.py
```

Any figure, for example:

```
python figures/fig_headline.py
python figures/fig_heatmaps.py
```

Each figure script prints its numerical results to the console and writes a
600-dpi PDF and PNG to `outputs/`. Run counts and the base seed are declared as
constants at the top of each script; results are fully determined by the seed
and therefore reproduce exactly. Regenerating the data snapshot (optional,
needs internet): `python data/acquire_snapshot.py`.

## Output Description

`monotonicity.py` prints eight PASS/FAIL checks and should end with
`8/8 checks passed`. `no_launch.py` prints cascade probabilities at five
altitudes; expect near zero at low altitude and rising values toward the
800-900 km legacy-debris band. Each figure script prints a small table of
thresholds and saves its figure. The headline figure shows the FCC threshold
sitting 10-15 percentage points below the IADC threshold across the
rule-binding band, both rules requiring zero compliance below roughly 625 km,
and the IADC rule becoming unachievable at the highest altitudes. The
disposal-floor figure shows both rules failing below roughly 75-80% disposal
success. The robustness matrix reports the FCC-below-IADC ordering preserved in
100% of the 81 tested parameter combinations, with finite-gap statistics
reported separately from the 26 combinations where the IADC rule is
unachievable. Thresholds sit on a 2.5-point grid, so a rerun on a different
platform may differ by one grid step at a few altitudes.

## Contact Information

Patrick Alexander H. Del Pilar
Xavier School, San Juan, Metro Manila, Philippines
Academic e-mail: pahdelpilar27@xs.edu.ph
Personal e-mail: thepatrickdelpilar@gmail.com

## Citation

A BibTeX entry will be added on publication.

## License

MIT License. See `LICENSE`.
