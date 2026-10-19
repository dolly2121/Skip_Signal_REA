"""Metrics and test harness.

Each test buyer's candidates are the homes they were shown on days 46 to 60.
A model re-ranks those. We only judge homes the buyer actually reacted to.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

SILVER_HOMES = Path("data/silver/listings.parquet")
GOLD = Path("data/gold")
REPORTS = Path("reports")
ANSWERS_DB = Path("data/answers/dealbreakers.parquet")
K = 10
GAIN = {"enquire": 2, "save": 1}
STRENGTH = {"leave": -1, "skip": 0, "view": 1, "save": 2, "enquire": 3}


def build_test_candidates():
    test = pd.read_parquet(GOLD / "test_events.parquet")
    test = test[test.has_history == 1].copy()
    test["strength"] = test.action.map(STRENGTH)
    # Strongest reaction per buyer and home
    cand = (test.sort_values("strength", kind="stable")
                .groupby(["buyer_id", "listing_id"], as_index=False)
                .last()[["buyer_id", "listing_id", "action"]])
    cand["gain"] = cand.action.map(GAIN).fillna(0)
    cand["skipped"] = (cand.action == "skip").astype(int)
    return cand


def order_by(cands, scores):
    ids = cands.listing_id.to_numpy()
    return list(ids[np.argsort(-np.asarray(scores), kind="stable")])


def ndcg(gains_ranked, gains_all, k=K):
    disc = 1 / np.log2(np.arange(2, k + 2))
    top = np.asarray(gains_ranked[:k], dtype=float)
    dcg = (top * disc[:len(top)]).sum()
    ideal = np.sort(np.asarray(gains_all, dtype=float))[::-1][:k]
    idcg = (ideal * disc[:len(ideal)]).sum()
    return dcg / idcg if idcg > 0 else np.nan


def evaluate_model(model, cands):
    rows, hidden_n, hidden_pos = [], 0, 0
    t0 = time.perf_counter()
    for bid, g in cands.groupby("buyer_id"):
        order, hidden = model.rank(bid, g)
        g = g.set_index("listing_id")
        top = order[:K]
        gains = g.loc[top, "gain"].to_numpy()
        n_pos = int((g.gain > 0).sum())
        rows.append({
            "p": (gains > 0).sum() / len(top) if top else 0.0,
            "r": (gains > 0).sum() / n_pos if n_pos else np.nan,
            "ndcg": ndcg(gains, g.gain.to_numpy()),
            "wasted": g.loc[top, "skipped"].mean() if top else 0.0,
        })
        if hidden:
            hidden_n += len(hidden)
            hidden_pos += int((g.loc[list(hidden), "gain"] > 0).sum())
    latency = (time.perf_counter() - t0) / len(rows) * 1000
    m = pd.DataFrame(rows)
    return {
        "model": model.name,
        "precision@10": round(m.p.mean(), 4),
        "recall@10": round(m.r.mean(), 4),
        "ndcg@10": round(m.ndcg.mean(), 4),
        "wasted_views_%": round(m.wasted.mean() * 100, 1),
        "wrongly_hidden_%": round(hidden_pos / hidden_n * 100, 1) if hidden_n else np.nan,
        "latency_ms": round(latency, 2),
        "n_buyers": len(m),
    }


def save_results(new_rows):
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "results.csv"
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.model.isin(new.model)], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def run():
    from src.models.baselines import (RandomRanker, PopularityRanker,
                                      ContentRanker, CFRanker)
    from src.models.two_tower import TwoTowerRanker
    from src.rank import DealbreakerRanker

    homes = pd.read_parquet(SILVER_HOMES)
    train = pd.read_parquet(GOLD / "train_events.parquet")
    cands = build_test_candidates()

    print(f"Test buyers: {cands.buyer_id.nunique():,}  "
          f"avg candidates: {cands.groupby('buyer_id').size().mean():.1f}  "
          f"positive rate: {(cands.gain > 0).mean() * 100:.1f}%  "
          f"skip rate: {cands.skipped.mean() * 100:.1f}%\n")

    # Oracle: true dealbreakers from the answer file. Evaluation only.
    truth = pd.read_parquet(ANSWERS_DB)
    oracle_file = GOLD / "dealbreakers_oracle.parquet"
    truth.rename(columns={"strength": "level"}).to_parquet(oracle_file, index=False)

    models = [
        RandomRanker(), PopularityRanker(), ContentRanker(), CFRanker(),
        TwoTowerRanker(),
        DealbreakerRanker(ContentRanker()),
        DealbreakerRanker(TwoTowerRanker()),
        DealbreakerRanker(ContentRanker(), db_file=oracle_file,
                          name="Content + true dealbreakers (oracle)"),
    ]
    results = [evaluate_model(m.fit(train, homes), cands) for m in models]
    table = save_results(results)
    print(table.to_string(index=False))


if __name__ == "__main__":
    run()