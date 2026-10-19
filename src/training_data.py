"""Training examples for the dwell time and next action models.

Leakage rule: features only use events before the target window.
  Fit set:  history days 1 to 30, targets days 31 to 45
  Test set: history days 1 to 45, targets days 46 to 60
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import buyer_profile
from src.models.dealbreaker import (prepare, detect_logreg, assign_levels,
                                    LR_C, LR_SOFT_T, LR_HARD_T)

SILVER_HOMES = Path("data/silver/listings.parquet")
GOLD = Path("data/gold")
FIT_CUTOFF = 30

FEATURES = [
    "log_price", "rooms", "bathroom", "car", "log_land", "distance", "age",
    "year_built_missing", "type_h", "type_u", "type_t",
    "log_budget", "usual_rooms", "skip_rate", "n_sessions",
    "price_gap", "room_gap", "abs_room_gap",
    "n_hard_hits", "n_soft_hits",
]
CLASSES = ["skip", "view", "save", "enquire", "leave"]


def dealbreaker_levels(history, homes, as_of):
    ev, flags, cats = prepare(history, homes, as_of)
    db = assign_levels(detect_logreg(ev, flags, cats, C=LR_C), LR_SOFT_T, LR_HARD_T)
    db = db.assign(v=db.level.map({"soft": 1, "hard": 2}))
    wide = (db.pivot(index="buyer_id", columns="category", values="v")
              .reindex(columns=cats).fillna(0))
    return wide.add_prefix("lvl_").reset_index(), cats


def build_examples(target, history, homes, as_of):
    prof = buyer_profile(history, homes, as_of)
    levels, cats = dealbreaker_levels(history, homes, as_of)

    X = (target[["buyer_id", "listing_id", "day", "action", "dwell_s"]]
         .merge(homes, on="listing_id")
         .merge(prof, on="buyer_id")
         .merge(levels, on="buyer_id", how="left"))

    lvl_cols = [f"lvl_{c}" for c in cats]
    X[lvl_cols] = X[lvl_cols].fillna(0)
    hits = np.stack([X[f"db_{c}"].to_numpy() * X[f"lvl_{c}"].to_numpy() for c in cats], axis=1)
    X["n_hard_hits"] = (hits == 2).sum(axis=1)
    X["n_soft_hits"] = (hits == 1).sum(axis=1)

    X["log_price"] = np.log(X.price)
    X["log_land"] = np.log1p(X.landsize)
    X["log_budget"] = np.log(X.budget)
    X["price_gap"] = (X.price - X.budget) / X.budget
    X["room_gap"] = X.rooms - X.usual_rooms
    X["abs_room_gap"] = X.room_gap.abs()
    return X[["buyer_id", "listing_id", "day", "action", "dwell_s"] + FEATURES]


def run():
    homes = pd.read_parquet(SILVER_HOMES)
    train = pd.read_parquet(GOLD / "train_events.parquet")
    test = pd.read_parquet(GOLD / "test_events.parquet")

    fit = build_examples(train[train.day > FIT_CUTOFF],
                         train[train.day <= FIT_CUTOFF], homes, FIT_CUTOFF)
    tst = build_examples(test, train, homes, int(train.day.max()))

    fit.to_parquet(GOLD / "examples_fit.parquet", index=False)
    tst.to_parquet(GOLD / "examples_test.parquet", index=False)

    print(f"Fit examples: {len(fit):,}   Test examples: {len(tst):,}")
    print("\nAverage dealbreaker hits by action (test):")
    print(tst.groupby("action")[["n_hard_hits", "n_soft_hits"]].mean().round(3))


if __name__ == "__main__":
    run()