"""Read the mistakes: hidden homes the buyer actually saved or enquired on.
Uses the answer file, so this is analysis only, never a feature."""
from pathlib import Path

import pandas as pd

from src.evaluate import build_test_candidates

GOLD = Path("data/gold")
SILVER_HOMES = Path("data/silver/listings.parquet")
ANSWERS_DB = Path("data/answers/dealbreakers.parquet")


def run():
    homes = pd.read_parquet(SILVER_HOMES)
    cands = build_test_candidates()
    pred = pd.read_parquet(GOLD / "dealbreakers_pred.parquet")
    truth = pd.read_parquet(ANSWERS_DB)
    hard = pred[pred.level == "hard"]

    flags = [c for c in homes.columns if c.startswith("db_")]
    long = homes.melt(id_vars="listing_id", value_vars=flags,
                      var_name="category", value_name="hit")
    long = long[long.hit == 1].assign(category=lambda d: d.category.str.removeprefix("db_"))

    hid = (cands.merge(hard, on="buyer_id")
                .merge(long[["listing_id", "category"]], on=["listing_id", "category"])
                .merge(truth, on=["buyer_id", "category"], how="left"))
    hid["real"] = hid.strength.notna()
    hid["wrong"] = hid.gain > 0

    wrong = hid[hid.wrong]
    print(f"Hidden (buyer, home, category) rows: {len(hid):,}")
    print(f"Wrongly hidden: {len(wrong):,} ({hid.wrong.mean() * 100:.1f}%)")
    print(f"  of those, dealbreaker was real (noise save): {wrong.real.mean() * 100:.1f}%")
    print(f"  of those, dealbreaker was false: {(~wrong.real).mean() * 100:.1f}%")

    print("\nBy category:")
    print(hid.groupby("category").agg(
        hidden=("wrong", "size"),
        wrongly_hidden_pct=("wrong", lambda s: round(s.mean() * 100, 1)),
        false_db_pct=("real", lambda s: round((~s).mean() * 100, 1)),
    ).to_string())

    hid["sessions"] = pd.cut(hid.n_sessions, [0, 4, 7, 100], labels=["3-4", "5-7", "8+"])
    hid["confidence"] = pd.cut(hid.score, [0, 4, 6, 1e9], labels=["3-4", "4-6", "6+"])
    for col in ["sessions", "confidence"]:
        print(f"\nBy {col}:")
        print(hid.groupby(col, observed=True).agg(
            hidden=("wrong", "size"),
            wrongly_hidden_pct=("wrong", lambda s: round(s.mean() * 100, 1)),
            false_db_pct=("real", lambda s: round((~s).mean() * 100, 1)),
        ).to_string())

    worst = wrong[~wrong.real].sort_values("score", ascending=False).head(20)
    print("\n20 worst: false dealbreaker, home hidden, buyer liked it")
    print(worst[["buyer_id", "listing_id", "category", "score", "n_skips",
                 "n_sessions", "action"]].round(2).to_string(index=False))


if __name__ == "__main__":
    run()