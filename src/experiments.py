"""Proof: segments, bootstrap confidence, simulated A/B test and ablation."""
import numpy as np
import pandas as pd

from src.evaluate import (GOLD, K, REPORTS, SILVER_HOMES, build_test_candidates,
                          evaluate_model, ndcg)
from src.models import dealbreaker as dbm
from src.models.baselines import ContentRanker
from src.rank import DealbreakerRanker

N_BOOT = 1000
SEED = 42


def per_buyer(model, cands, seen_listings):
    rows = []
    for bid, g in cands.groupby("buyer_id"):
        order, _ = model.rank(bid, g)
        g = g.set_index("listing_id")
        t = g.loc[order[:K]]
        n = len(t)
        rows.append({
            "buyer_id": bid,
            "ndcg": ndcg(t.gain.to_numpy(), g.gain.to_numpy()),
            "precision": (t.gain > 0).mean() if n else 0.0,
            "wasted": t.skipped.mean() if n else 0.0,
            "enquiry": (t.action == "enquire").mean() if n else 0.0,
            "new_listing_share": (~t.index.isin(seen_listings)).mean() if n else 0.0,
        })
    return pd.DataFrame(rows).set_index("buyer_id")


def segments(base, ours, train):
    sess = train.groupby("buyer_id").session_id.nunique().reindex(base.index).fillna(0)
    seg = pd.Series(np.where(sess < 5, "new (<5 sessions)", "regular (5+)"), index=base.index)
    rows = []
    for name in sorted(seg.unique()):
        idx = seg[seg == name].index
        rows.append({
            "segment": name, "buyers": len(idx),
            "ndcg_base": round(base.loc[idx, "ndcg"].mean(), 4),
            "ndcg_ours": round(ours.loc[idx, "ndcg"].mean(), 4),
            "wasted_base_%": round(base.loc[idx, "wasted"].mean() * 100, 1),
            "wasted_ours_%": round(ours.loc[idx, "wasted"].mean() * 100, 1),
        })
    rows.append({
        "segment": "brand new listings (share of top 10)", "buyers": len(base),
        "ndcg_base": round(base.new_listing_share.mean() * 100, 1),
        "ndcg_ours": round(ours.new_listing_share.mean() * 100, 1),
        "wasted_base_%": np.nan, "wasted_ours_%": np.nan,
    })
    return pd.DataFrame(rows)


def bootstrap(base, ours):
    rng = np.random.default_rng(SEED)
    rows = []
    for metric in ["ndcg", "precision", "wasted"]:
        d = (ours[metric] - base[metric]).dropna().to_numpy()
        boots = [rng.choice(d, len(d)).mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"metric": metric, "mean_change": round(d.mean(), 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "real": "yes" if lo > 0 or hi < 0 else "no"})
    return pd.DataFrame(rows)


def ab_test(base, ours):
    rng = np.random.default_rng(SEED)
    buyers = base.index.to_numpy()
    in_b = rng.random(len(buyers)) < 0.5
    A, B = base.loc[buyers[~in_b]], ours.loc[buyers[in_b]]
    rows = []
    for metric in ["wasted", "enquiry", "precision"]:
        a, b = A[metric].to_numpy(), B[metric].to_numpy()
        boots = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
                 for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"metric": metric, "group_A_normal": round(a.mean(), 4),
                     "group_B_dealbreaker": round(b.mean(), 4),
                     "change": round(b.mean() - a.mean(), 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "real": "yes" if lo > 0 or hi < 0 else "no"})
    return pd.DataFrame(rows), len(A), len(B)


def detect_variant(train, homes, name, use_recency=True, C=dbm.LR_C,
                   method="logreg", min_sessions=dbm.MIN_SESSIONS_HARD):
    ev, flags, cats = dbm.prepare(train, homes, int(train.day.max()), use_recency=use_recency)
    if method == "peeled":
        pred = dbm.detect_peeled(ev, flags, cats)
        soft_t, hard_t = dbm.SOFT_T, dbm.HARD_T
    else:
        pred = dbm.detect_logreg(ev, flags, cats, C=C)
        soft_t, hard_t = dbm.LR_SOFT_T, dbm.LR_HARD_T
    pred = pred[pred.score >= soft_t].copy()
    hard = (pred.score >= hard_t) & (pred.n_sessions >= min_sessions)
    pred["level"] = np.where(hard, "hard", "soft")
    path = GOLD / f"db_ablation_{name}.parquet"
    pred.to_parquet(path, index=False)
    return path


def ablation(train, homes, cands):
    full = GOLD / "dealbreakers_pred.parquet"
    print("  building ablation variants (about 3 minutes)...")
    variants = [
        ("Full system", {}),
        ("No hard removal", {"use_hard": False}),
        ("No soft push down", {"use_soft": False}),
        ("No explore slot", {"use_explore": False}),
        ("No enquiry boost", {"use_boost": False}),
        ("No recency weighting",
         {"db_file": detect_variant(train, homes, "no_recency", use_recency=False)}),
        ("No regularisation (C=100)",
         {"db_file": detect_variant(train, homes, "no_reg", C=100)}),
        ("No 3 session rule",
         {"db_file": detect_variant(train, homes, "no_min_sessions", min_sessions=0)}),
        ("Skip lift instead of logreg",
         {"db_file": detect_variant(train, homes, "peeled", method="peeled")}),
    ]
    rows = []
    for name, kw in variants:
        kw.setdefault("db_file", full)
        m = DealbreakerRanker(ContentRanker(), name=name, **kw).fit(train, homes)
        rows.append(evaluate_model(m, cands))
    cols = ["model", "precision@10", "ndcg@10", "wasted_views_%", "wrongly_hidden_%"]
    return pd.DataFrame(rows)[cols].rename(columns={"model": "variant"})


def run():
    homes = pd.read_parquet(SILVER_HOMES)
    train = pd.read_parquet(GOLD / "train_events.parquet")
    cands = build_test_candidates()
    seen = set(train.listing_id)
    REPORTS.mkdir(exist_ok=True)

    base = per_buyer(ContentRanker().fit(train, homes), cands, seen)
    ours = per_buyer(DealbreakerRanker(ContentRanker()).fit(train, homes), cands, seen)

    seg = segments(base, ours, train)
    boot = bootstrap(base, ours)
    ab, n_a, n_b = ab_test(base, ours)

    print("1. Segments (Content vs Content + dealbreakers)")
    print(seg.to_string(index=False))
    print("\n2. Bootstrap, 95% confidence on the change (ours minus base)")
    print(boot.to_string(index=False))
    print(f"\n3. Simulated A/B test (A: {n_a} buyers normal, B: {n_b} buyers dealbreaker)")
    print(ab.to_string(index=False))

    abl = ablation(train, homes, cands)
    print("\n4. Ablation (remove one idea at a time)")
    print(abl.to_string(index=False))

    seg.to_csv(REPORTS / "segments.csv", index=False)
    boot.to_csv(REPORTS / "bootstrap.csv", index=False)
    ab.to_csv(REPORTS / "ab_test.csv", index=False)
    abl.to_csv(REPORTS / "ablation.csv", index=False)


if __name__ == "__main__":
    run()
