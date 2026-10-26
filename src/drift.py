"""Drift check. Compares the last 7 days with the training window.
PSI above 0.2 raises an alert, which should trigger a retrain.

Run with --simulate to fake a 25% price rise and see the alert fire."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SILVER_HOMES = Path("data/silver/listings.parquet")
SILVER_EVENTS = Path("data/silver/events.parquet")
REPORTS = Path("reports")
ALERT = 0.2
TRAIN_END = 45
WINDOW = 7
FEATURES = ["price", "rooms", "distance", "bathroom", "car"]


def psi(ref, cur, bins=10):
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-6, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def run(simulate=False):
    homes = pd.read_parquet(SILVER_HOMES)
    events = pd.read_parquet(SILVER_EVENTS)
    last = int(events.day.max())

    ref = events[events.day <= TRAIN_END].merge(homes, on="listing_id")
    cur = events[events.day > last - WINDOW].merge(homes, on="listing_id")
    if simulate:
        cur = cur.assign(price=cur.price * 1.25)

    rows = [{"check": f"{f} of homes shown", "psi": psi(ref[f], cur[f])} for f in FEATURES]
    rows.append({"check": "buyer skip rate",
                 "psi": psi(ref.groupby("buyer_id").is_skip.mean(),
                            cur.groupby("buyer_id").is_skip.mean())})
    table = pd.DataFrame(rows)
    table["psi"] = table.psi.round(3)
    table["alert"] = table.psi > ALERT

    REPORTS.mkdir(exist_ok=True)
    report = {"reference_days": f"1 to {TRAIN_END}",
              "current_days": f"{last - WINDOW + 1} to {last}",
              "simulated_shift": simulate,
              "alert": bool(table.alert.any()),
              "checks": table.to_dict(orient="records")}
    (REPORTS / "drift.json").write_text(json.dumps(report, indent=2))

    print(f"Reference: days 1 to {TRAIN_END}   Current: days {last - WINDOW + 1} to {last}"
          + ("   (simulated 25% price rise)" if simulate else ""))
    print(table.to_string(index=False))
    if table.alert.any():
        print("\nALERT: drift above 0.2. Retrain now.")
        sys.exit(1)
    print("\nNo drift. Nightly retrain is enough.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", action="store_true")
    run(p.parse_args().simulate)