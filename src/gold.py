"""Silver events + Gold layer: time based train and test tables."""
from pathlib import Path
import pandas as pd

SILVER_HOMES = Path("data/silver/listings.parquet")
BRONZE_EVENTS = Path("data/bronze/events.parquet")
SILVER_EVENTS = Path("data/silver/events.parquet")
GOLD = Path("data/gold")
SPLIT_DAY = 45


def clean_events(events, homes):
    events = events.drop_duplicates()
    events = events[events.listing_id.isin(homes.listing_id)]
    events = events.assign(
        is_skip=(events.action == "skip").astype(int),
        is_positive=events.action.isin(["save", "enquire"]).astype(int),
        quick_exit=(events.dwell_s < 5).astype(int),
    )
    return events.sort_values(["buyer_id", "event_time"]).reset_index(drop=True)


def run():
    homes = pd.read_parquet(SILVER_HOMES)
    events = clean_events(pd.read_parquet(BRONZE_EVENTS), homes)
    events.to_parquet(SILVER_EVENTS, index=False)

    train = events[events.day <= SPLIT_DAY]
    test = events[events.day > SPLIT_DAY]
    test = test.assign(has_history=test.buyer_id.isin(set(train.buyer_id)).astype(int))

    # What each buyer actually liked in the test window (for scoring)
    test_positives = (test[test.is_positive == 1][["buyer_id", "listing_id"]]
                      .drop_duplicates())

    GOLD.mkdir(parents=True, exist_ok=True)
    train.to_parquet(GOLD / "train_events.parquet", index=False)
    test.to_parquet(GOLD / "test_events.parquet", index=False)
    test_positives.to_parquet(GOLD / "test_positives.parquet", index=False)

    print(f"Train: {len(train):,} events, days {train.day.min()} to {train.day.max()}")
    print(f"Test:  {len(test):,} events, days {test.day.min()} to {test.day.max()}")
    print(f"Test buyers: {test.buyer_id.nunique():,}  "
          f"with no train history: {(test.groupby('buyer_id').has_history.max() == 0).sum():,}")
    print(f"Test positives: {len(test_positives):,} (buyer, home) pairs")
    assert train.day.max() < test.day.min(), "Leak: train overlaps test"


if __name__ == "__main__":
    run()