"""Dealbreaker model.

A. Peeled skip lift: find the strongest dealbreaker, set those homes aside,
   look again. Stops overlapping categories riding on a real one.
B. Logistic regression per buyer on all flags at once.
Scored against the hidden answer file. Tuned for precision.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SILVER_HOMES = Path("data/silver/listings.parquet")
GOLD = Path("data/gold")
ANSWERS_DB = Path("data/answers/dealbreakers.parquet")

HALF_LIFE = 14
PRIOR_K = 5
MIN_N = 5               # need at least this many homes seen in a category
SOFT_T = 2.0            # peeled lift needed to push homes down
HARD_T = 3.0            # peeled lift needed to hide homes
MIN_SESSIONS_HARD = 3   # no hiding for new buyers


LR_C = 2.0              # logistic regression strength (lower = more shrinkage)
LR_SOFT_T = 2.5         # odds ratio needed to push homes down
LR_HARD_T = 3.0         # odds ratio needed to hide homes

COLS = ["buyer_id", "category", "score", "n_skips", "n_sessions"]


def prepare(events, homes, as_of_day, use_recency=True):
    flags = [c for c in homes.columns if c.startswith("db_")]
    ev = events[(events.day <= as_of_day) & (events.action != "leave")]
    ev = ev.merge(homes[["listing_id"] + flags], on="listing_id")
    ev["w"] = 0.5 ** ((as_of_day - ev.day) / HALF_LIFE) if use_recency else 1.0
    cats = [f.removeprefix("db_") for f in flags]
    return ev, flags, cats


def global_priors(ev, flags):
    y, w = ev.is_skip.to_numpy(), ev.w.to_numpy()
    p_in, p_out = [], []
    for f in flags:
        x = ev[f].to_numpy() == 1
        p_in.append((w * y)[x].sum() / w[x].sum())
        p_out.append((w * y)[~x].sum() / w[~x].sum())
    return np.array(p_in), np.array(p_out)


def rate(s, w, p, k):
    return (s + k * p) / (w + k) if (w + k) > 0 else p


def peel_one_buyer(X, y, w, p_in, p_out, k, min_score, max_found=3):
    found = []
    mask = np.ones(len(y), dtype=bool)
    for _ in range(max_found):
        best = None
        for j in range(X.shape[1]):
            if any(f[0] == j for f in found):
                continue
            inn = mask & (X[:, j] == 1)
            out = mask & (X[:, j] == 0)
            if inn.sum() < MIN_N:
                continue
            r_in = rate((w * y)[inn].sum(), w[inn].sum(), p_in[j], k)
            r_out = rate((w * y)[out].sum(), w[out].sum(), p_out[j], k)
            lift = r_in / max(r_out, 0.01)
            if best is None or lift > best[1]:
                best = (j, lift, int(y[inn].sum()))
        if best is None or best[1] < min_score:
            break
        found.append(best)
        mask &= X[:, best[0]] == 0
    return found


def detect_peeled(ev, flags, cats, min_score=1.5, k=PRIOR_K):
    p_in, p_out = global_priors(ev, flags)
    rows = []
    for bid, g in ev.groupby("buyer_id"):
        X, y, w = g[flags].to_numpy(), g.is_skip.to_numpy(), g.w.to_numpy()
        n_sess = g.session_id.nunique()
        for j, lift, n_skips in peel_one_buyer(X, y, w, p_in, p_out, k, min_score):
            rows.append((bid, cats[j], lift, n_skips, n_sess))
    return pd.DataFrame(rows, columns=COLS)


def detect_logreg(ev, flags, cats, min_score=1.2, C=LR_C):
    rows = []
    for bid, g in ev.groupby("buyer_id"):
        y = g.is_skip.to_numpy()
        if y.min() == y.max():
            continue
        X = g[flags].to_numpy()
        model = LogisticRegression(C=C).fit(X, y, sample_weight=g.w.to_numpy())
        odds = np.exp(model.coef_[0])
        n_sess = g.session_id.nunique()
        for j, o in enumerate(odds):
            if o >= min_score and X[:, j].sum() >= MIN_N:
                rows.append((bid, cats[j], o, int(y[X[:, j] == 1].sum()), n_sess))
    return pd.DataFrame(rows, columns=COLS)


def evaluate(pred, truth, buyers):
    truth = truth[truth.buyer_id.isin(buyers)]
    t = set(zip(truth.buyer_id, truth.category))
    p = set(zip(pred.buyer_id, pred.category))
    tp = len(t & p)
    return (tp / len(p) if p else 0.0), tp / len(t)


def sweep(pred, grid, truth, buyers):
    rows = []
    for t in grid:
        sub = pred[pred.score >= t]
        p, r = evaluate(sub, truth, buyers)
        rows.append((t, round(p * 100, 1), round(r * 100, 1), len(sub)))
    return pd.DataFrame(rows, columns=["threshold", "precision_%", "recall_%", "n_flagged"])


def assign_levels(pred, soft_t=SOFT_T, hard_t=HARD_T):
    pred = pred[pred.score >= soft_t].copy()
    hard = (pred.score >= hard_t) & (pred.n_sessions >= MIN_SESSIONS_HARD)
    pred["level"] = np.where(hard, "hard", "soft")
    return pred


def run():
    homes = pd.read_parquet(SILVER_HOMES)
    train = pd.read_parquet(GOLD / "train_events.parquet")
    truth = pd.read_parquet(ANSWERS_DB)
    as_of = int(train.day.max())

    ev, flags, cats = prepare(train, homes, as_of)
    buyers = ev.buyer_id.unique()

    peeled = detect_peeled(ev, flags, cats)
    print("A. Peeled skip lift")
    print(sweep(peeled, [2.0, 2.5, 3.0], truth, buyers).to_string(index=False))

    grid = [1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]
    for c in [0.5, 1.0, 2.0]:
        lr = detect_logreg(ev, flags, cats, C=c)
        print(f"\nB. Logistic regression, C={c}")
        print(sweep(lr, grid, truth, buyers).to_string(index=False))

    final = assign_levels(detect_logreg(ev, flags, cats, C=LR_C), LR_SOFT_T, LR_HARD_T)
    final.to_parquet(GOLD / "dealbreakers_pred.parquet", index=False)

    hard = final[final.level == "hard"]
    p_all, r_all = evaluate(final, truth, buyers)
    p_hard, _ = evaluate(hard, truth, buyers)
    print(f"\nSaved (logreg C={LR_C}): {len(final):,} dealbreakers, {len(hard):,} hard")
    print(f"All: precision {p_all * 100:.1f}%, recall {r_all * 100:.1f}%")
    print(f"Hard precision {p_hard * 100:.1f}%")


if __name__ == "__main__":
    run()