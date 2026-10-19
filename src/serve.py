"""Serving logic shared by the API and the demo page.
Loads the files the nightly batch writes. Same rules as src/rank.py.

source="live":   homes in budget, top 50 by content score, then dealbreaker rules
source="replay": the homes the buyer was actually shown in the last 15 days,
                 re-ranked. Same setup as the offline evaluation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.rank import K, SOFT_PENALTY

SERVING = Path("artifacts/serving")
BUDGET_LOW, BUDGET_HIGH = 0.7, 1.15
POOL = 50


class Recommender:
    def __init__(self, folder=SERVING):
        self.homes = pd.read_parquet(folder / "listings.parquet").set_index("listing_id")
        vec = pd.read_parquet(folder / "home_vectors.parquet").set_index("listing_id")
        self.H = vec.loc[self.homes.index].to_numpy()
        self.ids = self.homes.index.to_numpy()
        self.pos = pd.Series(np.arange(len(self.ids)), index=self.ids)
        self.prices = self.homes.price.to_numpy()

        prof = pd.read_parquet(folder / "buyer_profiles.parquet").set_index("buyer_id")
        self.budget = prof.pop("budget")
        self.P = prof

        db = pd.read_parquet(folder / "dealbreakers.parquet")
        self.db = {b: g for b, g in db.groupby("buyer_id")}
        self.manifest = json.loads((folder / "manifest.json").read_text())

        self.shown = {}
        shown_path = folder / "recent_shown.parquet"
        if shown_path.exists():
            shown = pd.read_parquet(shown_path)
            self.shown = {b: dict(zip(g.listing_id, g.action))
                          for b, g in shown.groupby("buyer_id")}
        self._demo = None

    def buyers(self, with_hard=False):
        if not with_hard:
            return list(self.P.index)
        return [b for b in self.P.index
                if b in self.db and (self.db[b].level == "hard").any()]

    def demo_buyers(self, k=K, limit=20):
        """Buyers whose normal replay top 10 has the most hard dealbreaker homes."""
        if self._demo is None:
            rows = []
            for b in self.buyers(with_hard=True):
                if b not in self.shown:
                    continue
                top = self._base(b, "replay")[:k]
                hard = self.db[b][self.db[b].level == "hard"].category
                F = self.homes.loc[top, [f"db_{c}" for c in hard]].to_numpy()
                n = int((F.sum(axis=1) > 0).sum())
                if n:
                    rows.append((n, b))
            self._demo = [b for n, b in sorted(rows, reverse=True)]
        return self._demo[:limit]

    def _card(self, lid, actual):
        h = self.homes.loc[lid]
        card = {
            "listing_id": int(lid), "suburb": h.suburb, "address": h.address,
            "type": h.type, "price": int(h.price), "rooms": int(h.rooms),
            "bathroom": int(h.bathroom), "car": int(h.car),
            "distance_km": float(h.distance),
        }
        if lid in actual:
            card["actual"] = actual[lid]
        return card

    def _base(self, buyer_id, source):
        p = self.P.loc[buyer_id].to_numpy()
        if source == "replay":
            ids = np.array(list(self.shown.get(buyer_id, {})))
            if len(ids) == 0:
                return ids
            s = self.H[self.pos[ids].to_numpy()] @ p
            return ids[np.argsort(-s, kind="stable")]
        b = self.budget[buyer_id]
        idx = np.where((self.prices >= b * BUDGET_LOW) & (self.prices <= b * BUDGET_HIGH))[0]
        s = self.H[idx] @ p
        order = np.argsort(-s, kind="stable")[:POOL]
        return self.ids[idx[order]]

    def _summarise(self, out, actual):
        if actual:
            out["skipped_in_top"] = sum(1 for h in out["homes"] if h.get("actual") == "skip")
            out["removed_but_liked"] = sum(1 for h in out["removed"]
                                           if h.get("actual") in ("save", "enquire"))
        return out

    def recommend(self, buyer_id, k=K, mode="dealbreaker", source="live"):
        if buyer_id not in self.P.index:
            raise KeyError(buyer_id)
        ids = self._base(buyer_id, source)
        actual = self.shown.get(buyer_id, {}) if source == "replay" else {}
        db = self.db.get(buyer_id)
        dealbreakers = [] if db is None else [
            {"category": r.category, "label": r.label, "level": r.level,
             "score": round(float(r.score), 2), "skipped": int(r.n_skips)}
            for r in db.itertuples()]
        out = {"buyer_id": buyer_id, "mode": mode, "source": source,
               "budget": int(self.budget[buyer_id]), "dealbreakers": dealbreakers}

        if mode == "normal" or db is None or len(ids) == 0:
            out["homes"] = [self._card(x, actual) for x in ids[:k]]
            out["removed"] = []
            return self._summarise(out, actual)

        F = self.homes.loc[ids]
        hard, soft = db[db.level == "hard"], db[db.level == "soft"]

        reasons = {}
        for r in hard.itertuples():
            for lid in ids[F[f"db_{r.category}"].to_numpy() == 1]:
                reasons.setdefault(
                    lid, f"Removed: {r.label[0].lower() + r.label[1:]}. "
                         f"You skipped {r.n_skips} like this.")

        soft_hits = np.zeros(len(ids))
        for c in soft.category:
            soft_hits += F[f"db_{c}"].to_numpy()
        score = 1 - np.arange(len(ids)) / len(ids) - SOFT_PENALTY * soft_hits

        keep = np.array([x not in reasons for x in ids])
        kept, kept_soft = ids[keep], soft_hits[keep]
        ranked = list(kept[np.argsort(-score[keep], kind="stable")])
        soft_set = set(kept[kept_soft > 0])

        explore = None
        if len(soft) and len(ranked) > k:
            pool = [x for x in ranked[k - 1:] if x in soft_set]
            if pool:
                explore = pool[0]
                ranked = ranked[:k - 1] + [explore] + [x for x in ranked[k - 1:] if x != explore]

        homes = []
        for x in ranked[:k]:
            card = self._card(x, actual)
            if x == explore:
                card["note"] = "Explore slot: checking if your needs have changed"
            elif x in soft_set:
                hit = [r.label.lower() + r.label[1:] for r in soft.itertuples()
                       if self.homes.at[x, f"db_{r.category}"] == 1]
                card["note"] = "Pushed down: " + ", ".join(hit)
            homes.append(card)

        out["homes"] = homes
        out["removed"] = [{**self._card(x, actual), "reason": reasons[x]}
                          for x in ids if x in reasons][:k]
        return self._summarise(out, actual)