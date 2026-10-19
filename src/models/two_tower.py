"""Two tower recommender (PyTorch).

A buyer tower and a home tower each turn features into a 32 number vector.
Score = how closely the two vectors point the same way.
Trained on views, saves and enquiries with in-batch negatives.
Skips are not used, so the dealbreaker layer stays a clean add on.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.evaluate import order_by
from src.features import buyer_profile
from src.models.baselines import CONTENT_COLS, WEIGHT

DIM = 32
EMB = 16
EPOCHS = 8
BATCH = 1024
LR = 1e-3
TEMP = 0.1
SEED = 42
USER_COLS = ["log_budget", "usual_rooms", "n_sessions"]


class Tower(nn.Module):
    def __init__(self, n_ids, n_feats):
        super().__init__()
        self.emb = nn.Embedding(n_ids, EMB)
        self.mlp = nn.Sequential(
            nn.Linear(EMB + n_feats, 64), nn.ReLU(), nn.Linear(64, DIM))

    def forward(self, ids, feats):
        x = torch.cat([self.emb(ids), feats], dim=1)
        return nn.functional.normalize(self.mlp(x), dim=1)


class TwoTowerRanker:
    name = "Two tower"

    def fit(self, train, homes):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)

        # Home features
        X = homes[CONTENT_COLS].copy()
        X["price"] = np.log(X["price"])
        X["landsize"] = np.log1p(X["landsize"].fillna(X["landsize"].median()))
        self.home_x = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
        self.l_idx = {l: i for i, l in enumerate(homes.listing_id)}

        # Buyer features (no skip based features)
        prof = buyer_profile(train, homes, int(train.day.max()))
        prof["log_budget"] = np.log(prof["budget"])
        prof = prof.dropna(subset=["log_budget"]).reset_index(drop=True)
        U = prof[USER_COLS].fillna(prof[USER_COLS].median())
        self.user_x = torch.tensor(StandardScaler().fit_transform(U), dtype=torch.float32)
        self.b_idx = {b: i for i, b in enumerate(prof.buyer_id)}

        # Positive pairs
        liked = train[train.action.isin(WEIGHT) & train.buyer_id.isin(self.b_idx)]
        b = torch.tensor(liked.buyer_id.map(self.b_idx).to_numpy())
        l = torch.tensor(liked.listing_id.map(self.l_idx).to_numpy())
        w = torch.tensor(liked.action.map(WEIGHT).to_numpy(), dtype=torch.float32)

        self.user_tower = Tower(len(self.b_idx), self.user_x.shape[1])
        self.home_tower = Tower(len(self.l_idx), self.home_x.shape[1])
        params = list(self.user_tower.parameters()) + list(self.home_tower.parameters())
        opt = torch.optim.Adam(params, lr=LR)

        n = len(b)
        for epoch in range(EPOCHS):
            perm = torch.tensor(rng.permutation(n))
            total = 0.0
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                bu, li, wi = b[idx], l[idx], w[idx]
                u = self.user_tower(bu, self.user_x[bu])
                h = self.home_tower(li, self.home_x[li])
                logits = u @ h.T / TEMP
                labels = torch.arange(len(idx))
                loss = nn.functional.cross_entropy(logits, labels, reduction="none")
                loss = (loss * wi).sum() / wi.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            print(f"  two tower epoch {epoch + 1}: loss {total / n:.3f}")

        with torch.no_grad():
            self.H = self.home_tower(torch.arange(len(self.l_idx)), self.home_x).numpy()
            self.U = self.user_tower(torch.arange(len(self.b_idx)), self.user_x).numpy()
        return self

    def rank(self, buyer_id, cands):
        if buyer_id not in self.b_idx:
            return list(cands.listing_id), set()
        cols = [self.l_idx[x] for x in cands.listing_id]
        s = self.H[cols] @ self.U[self.b_idx[buyer_id]]
        return order_by(cands, s), set()