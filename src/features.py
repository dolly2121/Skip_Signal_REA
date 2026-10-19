"""Buyer features: skip lift (smoothed, recency weighted) and buyer profile.

Only uses events up to as_of_day, so nothing from the future leaks in.
"""
from pathlib import Path
import pandas as pd

SILVER_HOMES = Path("data/silver/listings.parquet")
GOLD = Path("data/gold")
ANSWERS_DB = Path("data/answers/dealbreakers.parquet")


def category_stats(events, homes, as_of_day, half_life=14, prior_k=5,
                   use_recency=True, use_smoothing=True, use_lift=True):
    flags = [c for c in homes.columns if c.startswith("db_")]
    ev = events[(events.day <= as_of_day) & (events.action != "leave")]
    ev = ev[["buyer_id", "session_id", "day", "listing_id", "is_skip", "quick_exit"]]
    ev = ev.merge(homes[["listing_id"] + flags], on="listing_id")
    ev["w"] = 0.5 ** ((as_of_day - ev.day) / half_life) if use_recency else 1.0

    long = ev.melt(id_vars=["buyer_id", "w", "is_skip", "quick_exit"],
                   value_vars=flags, var_name="category", value_name="hit")
    long["category"] = long["category"].str.removeprefix("db_")
    long["ws"] = long.w * long.is_skip
    long["wq"] = long.w * long.quick_exit

    keys = ["buyer_id", "category"]
    inside = long[long.hit == 1].groupby(keys).agg(
        n_in=("w", "size"), w_in=("w", "sum"), s_in=("ws", "sum"), q_in=("wq", "sum"))
    outside = long[long.hit == 0].groupby(keys).agg(
        n_out=("w", "size"), w_out=("w", "sum"), s_out=("ws", "sum"))
    df = inside.join(outside, how="outer").fillna(0).reset_index()

    # Beta prior: start every buyer from the average across all buyers
    by_cat = df.groupby("category")[["s_in", "w_in", "s_out", "w_out"]].sum()
    df["p_in"] = df.category.map(by_cat.s_in / by_cat.w_in)
    df["p_out"] = df.category.map(by_cat.s_out / by_cat.w_out)
    k = prior_k if use_smoothing else 0

    df["rate_in"] = ((df.s_in + k * df.p_in) / (df.w_in + k)).fillna(df.p_in)
    df["rate_out"] = ((df.s_out + k * df.p_out) / (df.w_out + k)).fillna(df.p_out)
    df["quick_exit_in"] = (df.q_in / df.w_in).fillna(0)

    if use_lift:
        df["score"] = df.rate_in / df.rate_out.clip(lower=0.01)
    else:
        df["score"] = df.rate_in

    sessions = ev.groupby("buyer_id").session_id.nunique().rename("n_sessions")
    df = df.merge(sessions, on="buyer_id", how="left")
    return df[["buyer_id", "category", "n_in", "rate_in", "rate_out",
               "quick_exit_in", "score", "n_sessions"]]


def buyer_profile(events, homes, as_of_day):
    ev = events[events.day <= as_of_day]
    liked = ev[ev.action.isin(["view", "save", "enquire"])]
    liked = liked.merge(homes[["listing_id", "price", "rooms", "type", "region"]],
                        on="listing_id")
    prof = liked.groupby("buyer_id").agg(
        budget=("price", "median"),
        usual_rooms=("rooms", "median"),
        usual_type=("type", lambda s: s.mode().iat[0]),
        usual_region=("region", lambda s: s.mode().iat[0]),
    )
    overall = ev.groupby("buyer_id").agg(
        skip_rate=("is_skip", "mean"),
        n_sessions=("session_id", "nunique"),
    )
    return prof.join(overall, how="outer").reset_index()


def check(stats):
    """Sanity check only. Uses the answer file, so never import into models."""
    truth = pd.read_parquet(ANSWERS_DB)
    m = stats.merge(truth, on=["buyer_id", "category"], how="left")
    m["strength"] = m["strength"].fillna("not a dealbreaker")
    out = m.groupby("strength").agg(
        median_lift=("score", "median"),
        pct_lift_over_2=("score", lambda s: round((s > 2).mean() * 100, 1)),
    )
    print("\nSkip lift by true strength:")
    print(out.round(2))


def run():
    homes = pd.read_parquet(SILVER_HOMES)
    train = pd.read_parquet(GOLD / "train_events.parquet")
    as_of = int(train.day.max())

    stats = category_stats(train, homes, as_of)
    profile = buyer_profile(train, homes, as_of)

    stats.to_parquet(GOLD / "buyer_category_features.parquet", index=False)
    profile.to_parquet(GOLD / "buyer_profile.parquet", index=False)

    print(f"Category features: {len(stats):,} rows  Profiles: {len(profile):,} buyers")
    check(stats)


if __name__ == "__main__":
    run()