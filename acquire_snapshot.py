"""
CelesTrak SATCAT acquisition: grouped pull, on-orbit filter, shell binning.
Produces the POPULATION and CLASS_FRACTIONS tables in model.py and writes
the dated JSON snapshot. Requires network access to celestrak.org.
"""

import requests
import json
import io
import csv
from datetime import date

BASE = "https://celestrak.org/satcat/records.php"
GROUPS = ["active", "cosmos-2251-debris", "iridium-33-debris",
          "fengyun-1c-debris", "cosmos-1408-debris", "last-30-days"]
NAME_SEARCHES = ["R/B", "DEB"]
SHELLS = list(range(500, 901, 50))
ACTIVE_STATUS = {"+", "P", "B", "S"}


def fetch(params):
    r = requests.get(BASE, params={**params, "FORMAT": "csv"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    return list(csv.DictReader(io.StringIO(text))) if text else []


def mean_altitude(row):
    return (float(row["APOGEE"]) + float(row["PERIGEE"])) / 2.0


def classify(row):
    otype = row["OBJECT_TYPE"].strip()
    if otype == "R/B":
        return "rocketbody"
    if otype == "DEB":
        return "debris"
    if otype == "PAY" and row.get("OPS_STATUS_CODE", "").strip() in ACTIVE_STATUS:
        return "active"
    return "debris"


def on_orbit(row):
    if row.get("DECAY_DATE", "").strip():
        return False
    if row.get("OPS_STATUS_CODE", "").strip() == "D":
        return False
    return bool(row.get("APOGEE", "").strip()) and bool(row.get("PERIGEE", "").strip())


def main():
    records = {}
    for g in GROUPS:
        for rec in fetch({"GROUP": g}):
            records[rec["NORAD_CAT_ID"]] = rec
    for name in NAME_SEARCHES:
        for rec in fetch({"NAME": name}):
            records[rec["NORAD_CAT_ID"]] = rec
    records = list(records.values())

    population = {s: 0 for s in SHELLS}
    counts = {s: {"active": 0, "rocketbody": 0, "debris": 0} for s in SHELLS}
    leo_total = 0
    for row in records:
        if not on_orbit(row):
            continue
        try:
            alt = mean_altitude(row)
        except ValueError:
            continue
        if not (200 <= alt <= 2000):
            continue
        leo_total += 1
        if 475 <= alt <= 925:
            s = min(SHELLS, key=lambda x: abs(x - alt))
            population[s] += 1
            counts[s][classify(row)] += 1

    snapshot_date = date.today().isoformat()
    shells = []
    for s in SHELLS:
        t = population[s]
        if t == 0:
            continue
        shells.append({
            "altitude_km": s, "total": t,
            "fraction_active": round(counts[s]["active"] / t, 4),
            "fraction_rocketbody": round(counts[s]["rocketbody"] / t, 4),
            "fraction_debris": round(counts[s]["debris"] / t, 4),
        })
    snapshot = {
        "source": "CelesTrak Satellite Catalog (SATCAT), https://celestrak.org",
        "accessed": snapshot_date,
        "method": "grouped records.php pull; on-orbit filter (no decay date, "
                  "not status D); dead payloads classified as debris",
        "unique_objects_pulled": len(records),
        "total_leo_objects": leo_total,
        "shell_width_km": 100,
        "shells": shells,
    }
    fname = f"celestrak_snapshot_{snapshot_date.replace('-', '')}.json"
    with open(fname, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"pulled {len(records)} unique, {leo_total} LEO on-orbit; wrote {fname}")


if __name__ == "__main__":
    main()
