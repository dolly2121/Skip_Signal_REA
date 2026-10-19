"""Nightly batch job.

1. Rebuild Bronze, Silver and Gold from the latest raw data and events
2. Retrain the dealbreaker model and content model on all events
3. Write the serving files the API loads at start up
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src import bronze, gold, silver
from src.features import buyer_profile
from src.models import dealbreaker as dbm
from src.models.baselines import ContentRanker

SILVER_HOMES = Path("data/silver/listings.parquet")
SILVER_EVENTS = Path("data/silver/events.parquet")
CONFIG = Path("config/dealbreakers.yaml")
SERVING = Path("artifacts/serving")
LISTING_COLS = ["listing_id", "suburb", "address", "type", "price", "rooms",
                "bathroom", "car", "landsize", "distance", "year_built", "region"]


def run():
    t0 = time.perf_counter()

    print("1/4 Rebuilding data layers")
    bronze.run()
    silver.run()
    gold.run()

    homes = pd.read_parquet(SILVER_HOMES)
    events = pd.read_parquet(SILVER_EVENTS)
    as_of = int(events.day.max())
    with open(CONFIG) as f:
        labels = {k: v["label"] for k, v in yaml.safe_load(f)["categories"].items()}

    print("2/4 Retraining dealbreaker model on all events")
    ev, flags, cats = dbm.prepare(events, homes, as_of)
    db = dbm.assign_levels(dbm.detect_logreg(ev, flags, cats, C=dbm.LR_C),
                           dbm.LR_SOFT_T, dbm.LR_HARD_T)
    db["label"] = db.category.map(labels)

    print("3/4 Retraining content model on all events")
    content = ContentRanker().fit(events, homes)

    print("4/4 Writing serving files")
    SERVING.mkdir(parents=True, exist_ok=True)
    homes[LISTING_COLS + flags].to_parquet(SERVING / "listings.parquet", index=False)

    vec = content.vec.copy()
    vec.columns = [f"v{i}" for i in range(vec.shape[1])]
    vec.reset_index().to_parquet(SERVING / "home_vectors.parquet", index=False)

    prof = content.profile.copy()
    prof.columns = [f"v{i}" for i in range(prof.shape[1])]
    prof.index.name = "buyer_id"
    budgets = buyer_profile(events, homes, as_of)[["buyer_id", "budget"]]
    prof = prof.reset_index().merge(budgets, on="buyer_id").dropna(subset=["budget"])
    prof.to_parquet(SERVING / "buyer_profiles.parquet", index=False)

    db.to_parquet(SERVING / "dealbreakers.parquet", index=False)

    recent = events[events.day > as_of - 15]
    strength = {"leave": -1, "skip": 0, "view": 1, "save": 2, "enquire": 3}
    shown = (recent.assign(s=recent.action.map(strength))
                   .sort_values("s", kind="stable")
                   .groupby(["buyer_id", "listing_id"], as_index=False)
                   .last()[["buyer_id", "listing_id", "action"]])
    shown.to_parquet(SERVING / "recent_shown.parquet", index=False)

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_up_to_day": as_of,
        "listings": int(len(homes)),
        "buyers": int(len(prof)),
        "dealbreakers": int(len(db)),
        "hard_dealbreakers": int((db.level == "hard").sum()),
        "thresholds": {"C": dbm.LR_C, "soft": dbm.LR_SOFT_T, "hard": dbm.LR_HARD_T},
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    (SERVING / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    run()