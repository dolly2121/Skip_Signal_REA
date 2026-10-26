"""Serving logic shared by the API and the web page.
Loads the files the nightly batch writes. Same rules as src/rank.py.

recommend(): scores a simulated buyer from the training data
learn():     live mode. A visitor rates homes on the website and the
             dealbreaker model is fitted on their clicks, right then.
"""
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from src.rank import K, SOFT_PENALTY

SERVING = Path("artifacts/serving")
CONFIG = Path("config/dealbreakers.yaml")
BUDGET_LOW, BUDGET_HIGH = 0.7, 1.15
POOL = 50

# Live mode settings. Same model and thresholds as offline, tuned for few clicks.
LR_C, LR_SOFT_T, LR_HARD_T = 2.0, 2.5, 4.0
LIVE_MIN_ACTIONS = 6     # clicks before we try to learn anything
LIVE_MIN_SEEN = 3        # homes seen in a category before it can count
LIVE_HARD_SHARE = 0.8    # share of those skipped before we hide the category
LIVE_SOFT_SHARE = 0.6
LIVE_BAND = (0.85, 1.15)

STREETVIEW_KEY = os.environ.get("STREETVIEW_KEY", "")


def lower_first(s):
    return s[0].lower() + s[1:]


def learn_dealbreakers(X, y, cats, labels, ignore=(), locked=()):
    """Fit one logistic regression on a visitor's clicks.
    X: homes x category flags, y: 1 if skipped. Returns learned dealbreakers.
    locked: categories already hidden. Once hidden, the visitor stops seeing
    them, so the odds ratio fades. They stay hidden while the visitor's own
    skip share for them stays high, or until they click Undo."""
    if len(y) < LIVE_MIN_ACTIONS or y.min() == y.max():
        return []
    odds = np.exp(LogisticRegression(C=LR_C).fit(X, y).coef_[0])
    out = []
    for j, c in enumerate(cats):
        if c in ignore:
            continue
        seen = X[:, j] == 1
        n_seen, n_skip = int(seen.sum()), int(y[seen].sum())
        if n_seen < LIVE_MIN_SEEN:
            continue
        share = n_skip / n_seen
        if share >= LIVE_HARD_SHARE and (odds[j] >= LR_HARD_T or c in locked):
            level = "hard"
        elif odds[j] >= LR_SOFT_T and share >= LIVE_SOFT_SHARE:
            level = "soft"
        else:
            continue
        out.append({"category": c, "label": labels[c], "level": level,
                    "score": round(float(odds[j]), 2),
                    "skipped": n_skip, "seen": n_seen})
    return sorted(out, key=lambda d: (d["level"] != "hard", -d["score"]))


class Recommender:
    def __init__(self, folder=SERVING):
        self.homes = pd.read_parquet(folder / "listings.parquet").set_index("listing_id")
        vec = pd.read_parquet(folder / "home_vectors.parquet").set_index("listing_id")
        self.H = vec.loc[self.homes.index].to_numpy()
        self.ids = self.homes.index.to_numpy()
        self.pos = pd.Series(np.arange(len(self.ids)), index=self.ids)
        self.prices = self.homes.price.to_numpy()

        with open(CONFIG) as f:
            cfg = yaml.safe_load(f)["categories"]
        self.cats = list(cfg)
        self.labels = {k: v["label"] for k, v in cfg.items()}
        self.flag_cols = [f"db_{c}" for c in self.cats]

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

    # ---------- shared ----------

    def _photo(self, h):
        if not STREETVIEW_KEY:
            return None
        loc = quote(f"{h.address}, {h.suburb} VIC, Australia")
        return ("https://maps.googleapis.com/maps/api/streetview?size=640x400&fov=80"
                f"&return_error_code=true&location={loc}&key={STREETVIEW_KEY}")

    def _card(self, lid, actual=None):
        h = self.homes.loc[lid]
        card = {
            "listing_id": int(lid), "suburb": h.suburb, "address": h.address,
            "type": h.type, "price": int(h.price), "rooms": int(h.rooms),
            "bathroom": int(h.bathroom), "car": int(h.car),
            "distance_km": float(h.distance),
            "flags": [c for c in self.cats if h[f"db_{c}"] == 1],
            "photo": self._photo(h),
        }
        if actual and lid in actual:
            card["actual"] = actual[lid]
        return card

    # ---------- live mode: a visitor rates homes ----------

    def budgets(self):
        return [500_000, 750_000, 1_000_000, 1_500_000, 2_000_000]

    def learn(self, budget, actions, ignore=(), locked=(), k=12):
        lo, hi = budget * LIVE_BAND[0], budget * LIVE_BAND[1]
        pool = self.ids[(self.prices >= lo) & (self.prices <= hi)]
        rng = np.random.default_rng(int(budget))
        pool = pool[rng.permutation(len(pool))]

        acted = {}
        for a in actions:
            lid = int(a["listing_id"])
            if lid in self.pos.index and a["action"] in ("skip", "save"):
                acted[lid] = a["action"]

        learned = []
        if acted:
            ids = np.array(list(acted))
            X = self.homes.loc[ids, self.flag_cols].to_numpy()
            y = np.array([1 if acted[i] == "skip" else 0 for i in ids])
            learned = learn_dealbreakers(X, y, self.cats, self.labels,
                                         set(ignore), set(locked))

        hard = [d["category"] for d in learned if d["level"] == "hard"]
        soft = [d["category"] for d in learned if d["level"] == "soft"]

        F = self.homes.loc[pool]
        hard_hit = (F[[f"db_{c}" for c in hard]].to_numpy().sum(axis=1) > 0
                    if hard else np.zeros(len(pool), bool))
        soft_hits = (F[[f"db_{c}" for c in soft]].to_numpy().sum(axis=1)
                     if soft else np.zeros(len(pool)))

        not_rated = np.array([x not in acted for x in pool])
        n = len(pool)
        score = 1 - np.arange(n) / max(n, 1) - SOFT_PENALTY * soft_hits

        # Normal recommender: next k homes, no dealbreakers
        normal_next = pool[not_rated][:k]
        normal_bad = int(hard_hit[not_rated][:k].sum())

        keep = not_rated & ~hard_hit
        order = np.argsort(-score[keep], kind="stable")
        feed_ids = pool[keep][order][:k]

        homes = []
        for x in feed_ids:
            card = self._card(x)
            hit = [self.labels[c] for c in soft if self.homes.at[x, f"db_{c}"] == 1]
            if hit:
                card["note"] = "Shown lower: " + ", ".join(lower_first(s) for s in hit)
            homes.append(card)

        hidden_examples = []
        for x in pool[not_rated & hard_hit][:3]:
            card = self._card(x)
            why = [self.labels[c] for c in hard if self.homes.at[x, f"db_{c}"] == 1]
            card["reason"] = "Hidden: " + ", ".join(lower_first(s) for s in why)
            hidden_examples.append(card)

        n_rated = len(acted)
        n_skip = sum(1 for v in acted.values() if v == "skip")
        if n_rated < LIVE_MIN_ACTIONS:
            status = f"Rate {LIVE_MIN_ACTIONS - n_rated} more homes and Skip Signal starts learning."
        elif n_skip == 0:
            status = "You saved everything. Skip a few homes you don't like."
        elif n_skip == n_rated:
            status = "You skipped everything. Save a few homes you like, so we can tell the difference."
        elif not learned:
            status = "No clear pattern yet. Keep going."
        else:
            status = "Learning from your clicks."

        return {
            "budget": int(budget),
            "rated": n_rated, "skipped": n_skip, "saved": n_rated - n_skip,
            "status": status,
            "learned": learned,
            "in_budget": int(n),
            "hidden": int(hard_hit.sum()),
            "normal_next_bad": normal_bad,
            "normal_next_count": int(len(normal_next)),
            "homes": homes,
            "normal_homes": [self._card(x) for x in normal_next],
            "hidden_examples": hidden_examples,
        }

    # ---------- simulated buyers (from the training data) ----------

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
                    lid, f"Removed: {lower_first(r.label)}. You skipped {r.n_skips} like this.")

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
                hit = [lower_first(r.label) for r in soft.itertuples()
                       if self.homes.at[x, f"db_{r.category}"] == 1]
                card["note"] = "Pushed down: " + ", ".join(hit)
            homes.append(card)

        out["homes"] = homes
        out["removed"] = [{**self._card(x, actual), "reason": reasons[x]}
                          for x in ids if x in reasons][:k]
        return self._summarise(out, actual)
