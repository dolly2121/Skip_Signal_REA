"""FastAPI service: top 10 homes for a buyer, normal or dealbreaker aware."""
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from src.serve import Recommender

WEB = Path("web/index.html")

app = FastAPI(title="Skip Signal", version="1.0")
rec = Recommender()


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(WEB)


@app.get("/health")
def health():
    return {"status": "ok", **rec.manifest}


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