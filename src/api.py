"""FastAPI service for Skip Signal."""
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.serve import Recommender

WEB = Path("web/index.html")

app = FastAPI(title="Skip Signal", version="2.0")
rec = Recommender()


class Action(BaseModel):
    listing_id: int
    action: str = Field(pattern="^(skip|save)$")


class LearnRequest(BaseModel):
    budget: int = Field(ge=100_000, le=10_000_000)
    actions: list[Action] = Field(default_factory=list, max_length=500)
    ignore: list[str] = Field(default_factory=list)
    locked: list[str] = Field(default_factory=list)
    k: int = Field(12, ge=1, le=50)


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(WEB)


@app.get("/health")
def health():
    return {"status": "ok", **rec.manifest}


@app.get("/budgets")
def budgets():
    return {"budgets": rec.budgets()}


@app.post("/learn")
def learn(req: LearnRequest):
    """Live mode. Send the homes a visitor skipped or saved. Get back what
    Skip Signal learned and their next homes, with dealbreakers removed."""
    t0 = time.perf_counter()
    out = rec.learn(req.budget, [a.model_dump() for a in req.actions],
                    ignore=req.ignore, locked=req.locked, k=req.k)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


@app.get("/buyers")
def buyers(limit: int = Query(20, ge=1, le=500), with_hard: bool = False,
           demo: bool = False):
    if demo:
        return {"buyers": rec.demo_buyers(limit=limit)}
    return {"buyers": rec.buyers(with_hard)[:limit]}


@app.get("/recommend/{buyer_id}")
def recommend(buyer_id: str,
              mode: str = Query("dealbreaker", pattern="^(normal|dealbreaker)$"),
              source: str = Query("live", pattern="^(live|replay)$"),
              k: int = Query(10, ge=1, le=50)):
    t0 = time.perf_counter()
    try:
        out = rec.recommend(buyer_id, k=k, mode=mode, source=source)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown buyer {buyer_id}")
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out
