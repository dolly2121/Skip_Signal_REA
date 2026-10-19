"""Ranker: base scores, remove hard dealbreakers, push soft ones down,
boost likely enquiries, keep one explore slot."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.training_data import FEATURES

GOLD = Path("data/gold")
ARTIFACTS = Path("artifacts")
K = 10
SOFT_PENALTY = 0.3
ENQ_BOOST = 0.2


def load_enquiry_probs():
    ex = pd.read_parquet(GOLD / "examples_test.parquet")
    model = joblib.load(ARTIFACTS / "next_action.joblib")
    col = list(model.classes_).index("enquire")
    ex["p_enquire"] = model.predict_proba(ex[FEATURES])[:, col]
    return ex.groupby(["buyer_id", "listing_id"]).p_enquire.mean().to_dict()


class DealbreakerRanker:
    def __init__(self, base, use_hard=True, use_soft=True,
                use_explore=True, use_boost=False, name=None,
                db_file=GOLD / "dealbreakers_pred.parquet"):
        self.base = base
        self.use_hard, self.use_soft = use_hard, use_soft
        self.use_explore, self.use_boost = use_explore, use_boost
        self.db_file = db_file
        self.name = name or f"{base.name} + dealbreakers"

    def fit(self, train, homes):
        self.base.fit(train, homes)
        flags = [c for c in homes.columns if c.startswith("db_")]
        self.flags = homes.set_index("listing_id")[flags]
        db = pd.read_parquet(self.db_file)
        by = lambda lvl: db[db.level == lvl].groupby("buyer_id").category.apply(list).to_dict()
        self.hard, self.soft = by("hard"), by("soft")
        self.p_enq = load_enquiry_probs() if self.use_boost else {}
        return self

    def _hits(self, F, cats):
        if not cats:
            return np.zeros(len(F))
        return F[[f"db_{c}" for c in cats]].to_numpy().sum(axis=1)

    def rank(self, buyer_id, cands):
        order, _ = self.base.rank(buyer_id, cands)
        ids = np.array(order)
        n = len(ids)
        score = 1 - np.arange(n) / n          # base rank turned into a score
        F = self.flags.loc[ids]

        hard = self.hard.get(buyer_id, []) if self.use_hard else []
        soft = self.soft.get(buyer_id, []) if self.use_soft else []
        hard_hit = self._hits(F, hard) > 0
        soft_hits = self._hits(F, soft)

        score = score - SOFT_PENALTY * soft_hits
        if self.use_boost:
            p = np.array([self.p_enq.get((buyer_id, x), 0.0) for x in ids])
            score = score + ENQ_BOOST * p

        hidden = set(ids[hard_hit])
        keep = ~hard_hit
        kept_ids, kept_soft = ids[keep], soft_hits[keep]
        ranked = list(kept_ids[np.argsort(-score[keep], kind="stable")])

        # Explore slot: one soft dealbreaker home in the last top 10 spot
        if self.use_explore and soft and len(ranked) > K:
            soft_set = set(kept_ids[kept_soft > 0])
            pool = [x for x in ranked[K - 1:] if x in soft_set]
            if pool:
                rest = [x for x in ranked[K - 1:] if x != pool[0]]
                ranked = ranked[:K - 1] + [pool[0]] + rest
        return ranked, hidden