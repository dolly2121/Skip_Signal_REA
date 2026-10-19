"""Buyer simulator.

Fake buyer behaviour on real Melbourne listings.
Events go to Bronze. Hidden dealbreakers go to data/answers (never used as features).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SILVER = Path("data/silver/listings.parquet")
CONFIG = Path("config/dealbreakers.yaml")
EVENTS = Path("data/bronze/events.parquet")
ANSWERS_BUYERS = Path("data/answers/buyers.parquet")
ANSWERS_DB = Path("data/answers/dealbreakers.parquet")

N_BUYERS = 2000
N_DAYS = 60
SESSION_SIZE = 20
MEAN_SESSIONS = 6
HARD_SKIP = 0.90
SOFT_SKIP = 0.60
NOISE = 0.20
LEAVE_PROB = 0.03
START = pd.Timestamp("2026-08-01")
SEED = 42

ACTIONS = np.array(["skip", "view", "save", "enquire"])
DWELL = {"skip": (2, 5), "view": (15, 90), "save": (60, 180),
         "enquire": (90, 240), "leave": (1, 3)}

rng = np.random.default_rng(SEED)


def make_buyers(homes, names):
    anchors = homes.sample(N_BUYERS, replace=True, random_state=SEED)
    buyers, truth = [], []
    for i, a in enumerate(anchors.itertuples(index=False)):
        bid = f"b{i:04d}"
        # Only pick dealbreakers the buyer's anchor home does not break
        ok = [n for n in names if getattr(a, f"db_{n}") == 0]
        n_hard = min(len(ok), int(rng.integers(1, 3)))
        n_soft = min(len(ok) - n_hard, int(rng.integers(0, 2)))
        picks = rng.choice(ok, n_hard + n_soft, replace=False)
        for c in picks[:n_hard]:
            truth.append((bid, c, "hard"))
        for c in picks[n_hard:]:
            truth.append((bid, c, "soft"))
        buyers.append({
            "buyer_id": bid,
            "budget": a.price * rng.uniform(0.9, 1.2),
            "pref_rooms": a.rooms,
            "pref_region": a.region,
            "n_sessions": max(1, int(rng.poisson(MEAN_SESSIONS))),
        })
    truth = pd.DataFrame(truth, columns=["buyer_id", "category", "strength"])
    return pd.DataFrame(buyers), truth


def pick_homes(homes, b):
    lo, hi = b.budget * 0.7, b.budget * 1.15
    pool = homes[(homes.price >= lo) & (homes.price <= hi)]
    local = pool[pool.region == b.pref_region]
    n_local = min(len(local), int(SESSION_SIZE * 0.7))
    n_other = min(len(pool), SESSION_SIZE - n_local)
    seed = int(rng.integers(1_000_000_000))
    picked = pd.concat([local.sample(n_local, random_state=seed),
                        pool.sample(n_other, random_state=seed + 1)])
    return picked.drop_duplicates("listing_id").sample(frac=1, random_state=seed + 2)


def choose_action(h, b, hard, soft):
    if rng.random() < NOISE:
        return str(rng.choice(ACTIONS, p=[0.4, 0.4, 0.15, 0.05]))
    if any(getattr(h, f"db_{c}") for c in hard) and rng.random() < HARD_SKIP:
        return "skip"
    if any(getattr(h, f"db_{c}") for c in soft) and rng.random() < SOFT_SKIP:
        return "skip"
    room_gap = min(abs(h.rooms - b.pref_rooms) / 3, 1)
    price_gap = min(abs(h.price - b.budget) / b.budget, 1)
    fit = 1 - 0.5 * room_gap - 0.5 * price_gap
    p_skip = 0.35 - 0.20 * fit
    p_save = 0.08 + 0.17 * fit
    p_enq = 0.03 + 0.07 * fit
    p_view = 1 - p_skip - p_save - p_enq
    return str(rng.choice(ACTIONS, p=[p_skip, p_view, p_save, p_enq]))


def simulate(homes, buyers, truth):
    hard_map = truth[truth.strength == "hard"].groupby("buyer_id")["category"].apply(list).to_dict()
    soft_map = truth[truth.strength == "soft"].groupby("buyer_id")["category"].apply(list).to_dict()
    events, sid = [], 0
    for b in buyers.itertuples(index=False):
        hard = hard_map.get(b.buyer_id, [])
        soft = soft_map.get(b.buyer_id, [])
        days = np.sort(rng.integers(1, N_DAYS + 1, b.n_sessions))
        for day in days:
            sid += 1
            t = START + pd.Timedelta(days=int(day) - 1, minutes=int(rng.integers(420, 1380)))
            for pos, h in enumerate(pick_homes(homes, b).itertuples(index=False), start=1):
                action = choose_action(h, b, hard, soft)
                if action in ("skip", "view") and rng.random() < LEAVE_PROB:
                    action = "leave"
                lo, hi = DWELL[action]
                dwell = round(float(rng.uniform(lo, hi)), 1)
                events.append((f"s{sid:06d}", b.buyer_id, h.listing_id, int(day),
                               t, pos, action, dwell))
                t += pd.Timedelta(seconds=dwell)
                if action == "leave":
                    break
    cols = ["session_id", "buyer_id", "listing_id", "day",
            "event_time", "position", "action", "dwell_s"]
    return pd.DataFrame(events, columns=cols)


def check(events, homes, truth):
    """Sanity check only. Uses the answer file, so never import this into models."""
    flags = [c for c in homes.columns if c.startswith("db_")]
    ev = events[events.action != "leave"].merge(homes[["listing_id"] + flags], on="listing_id")
    ev["is_skip"] = (ev.action == "skip").astype(int)
    long = ev.melt(id_vars=["buyer_id", "is_skip"], value_vars=flags,
                   var_name="category", value_name="hit")
    long["category"] = long["category"].str.removeprefix("db_")
    long = long.merge(truth, on=["buyer_id", "category"], how="left")
    long["strength"] = long["strength"].fillna("not a dealbreaker")
    table = long[long.hit == 1].groupby("strength")["is_skip"].mean().mul(100).round(1)
    print("\nSkip rate on homes that hit a category:")
    print(table.astype(str) + "%")


def run():
    homes = pd.read_parquet(SILVER)
    with open(CONFIG) as f:
        names = list(yaml.safe_load(f)["categories"])

    buyers, truth = make_buyers(homes, names)
    events = simulate(homes, buyers, truth)

    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    ANSWERS_DB.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS, index=False)
    buyers.to_parquet(ANSWERS_BUYERS, index=False)
    truth.to_parquet(ANSWERS_DB, index=False)

    print(f"Buyers: {len(buyers):,}  Sessions: {events.session_id.nunique():,}  "
          f"Events: {len(events):,}")
    print("\nAction mix:")
    print((events.action.value_counts(normalize=True) * 100).round(1).astype(str) + "%")
    check(events, homes, truth)


if __name__ == "__main__":
    run()