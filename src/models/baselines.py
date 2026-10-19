"""Baseline recommenders. They learn only from views, saves and enquiries.
Skips are ignored on purpose. That is the gap the dealbreaker model fills."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src.evaluate import order_by

WEIGHT = {"view": 1.0, "save": 3.0, "enquire": 5.0}
CONTENT_COLS = ["price", "rooms", "bathroom", "car", "landsize", "distance",
                "type_h", "type_u", "type_t", "lat", "lng"]


def liked_events(train):
    liked = train[train.action.isin(WEIGHT)].copy()
    liked["w"] = liked.action.map(WEIGHT)
    return liked


class RandomRanker:
    name = "Random"

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def fit(self, train, homes):
        return self

    def rank(self, buyer_id, cands):
        ids = cands.listing_id.to_numpy().copy()
        self.rng.shuffle(ids)
        return list(ids), set()


class PopularityRanker:
    name = "Popularity"

    def fit(self, train, homes):
        liked = liked_events(train)
        self.score = liked.groupby("listing_id").w.sum()
        return self

    def rank(self, buyer_id, cands):
        s = cands.listing_id.map(self.score).fillna(0).to_numpy()
        return order_by(cands, s), set()


class ContentRanker:
    name = "Content based"

    def fit(self, train, homes):
        X = homes[CONTENT_COLS].copy()
        X["price"] = np.log(X["price"])
        X["landsize"] = np.log1p(X["landsize"].fillna(X["landsize"].median()))
        X = StandardScaler().fit_transform(X)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        self.vec = pd.DataFrame(X, index=homes.listing_id)

        liked = liked_events(train)
        V = self.vec.loc[liked.listing_id].to_numpy() * liked.w.to_numpy()[:, None]
        prof = pd.DataFrame(V, index=liked.buyer_id.to_numpy()).groupby(level=0).sum()
        arr = prof.to_numpy()
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        self.profile = pd.DataFrame(arr, index=prof.index)
        return self

    def rank(self, buyer_id, cands):
        if buyer_id not in self.profile.index:
            return list(cands.listing_id), set()
        s = self.vec.loc[cands.listing_id].to_numpy() @ self.profile.loc[buyer_id].to_numpy()
        return order_by(cands, s), set()


class CFRanker:
    name = "Collaborative filtering"

    def __init__(self, k=32, seed=42):
        self.k, self.seed = k, seed

    def fit(self, train, homes):
        liked = liked_events(train)
        self.b_idx = {b: i for i, b in enumerate(liked.buyer_id.unique())}
        self.l_idx = {l: i for i, l in enumerate(homes.listing_id)}
        rows = liked.buyer_id.map(self.b_idx).to_numpy()
        cols = liked.listing_id.map(self.l_idx).to_numpy()
        M = csr_matrix((liked.w.to_numpy(), (rows, cols)),
                       shape=(len(self.b_idx), len(self.l_idx)))
        M.data = np.log1p(M.data)
        svd = TruncatedSVD(n_components=self.k, random_state=self.seed)
        self.U = svd.fit_transform(M)
        self.V = svd.components_.T
        return self

    def rank(self, buyer_id, cands):
        if buyer_id not in self.b_idx:
            return list(cands.listing_id), set()
        cols = [self.l_idx[l] for l in cands.listing_id]
        s = self.V[cols] @ self.U[self.b_idx[buyer_id]]
        return order_by(cands, s), set()