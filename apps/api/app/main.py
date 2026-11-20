"""NearZeroFP Vision API: live agent re-runs, appearance search, dataset export."""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

from .agent import FAST_MODEL, STRONG_MODEL, Evidence, Verifier  # noqa: E402
from .open_vlm import OpenVLM  # noqa: E402
from .search import Searcher  # noqa: E402

PUBLIC = Path(os.getenv("PUBLIC_DIR", ROOT / "apps" / "web" / "public"))
LIVE_DAILY_CAP = int(os.getenv("LIVE_DAILY_CAP", "60"))
LIVE_BUDGET_USD = float(os.getenv("LIVE_BUDGET_USD", "0.25"))  # hard daily $ ceiling for public calls
LIVE_ALLOW_STRONG = os.getenv("LIVE_ALLOW_STRONG", "0") == "1"  # strong model off in the public demo
PER_IP_PER_MIN = int(os.getenv("PER_IP_PER_MIN", "3"))
SEARCH_COST_ESTIMATE = 0.001

app = FastAPI(title="NearZeroFP Vision API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")],
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def load_results() -> dict[str, dict]:
    out = {}
    for p in sorted((PUBLIC / "results").glob("*.json")):
        if p.stem not in ("bench", "bench_v2", "index", "human_labels"):
            out[p.stem] = json.loads(p.read_text())
    return out


RESULTS = load_results()
_verifier: Verifier | None = None
_open_vlm = OpenVLM.from_env()
_searcher = Searcher(RESULTS)


def verifier() -> Verifier:
    global _verifier
    if _verifier is None:
        _verifier = Verifier()
    return _verifier


# --- cheap abuse protection for the public demo -------------------------------------------
_hits: dict[str, deque] = defaultdict(deque)
_day = {"date": time.strftime("%Y-%m-%d"), "count": 0, "usd": 0.0}


def guard(request: Request) -> None:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0]
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= PER_IP_PER_MIN:
        raise HTTPException(429, "Slow down: live agent calls are rate-limited for this public demo.")
    today = time.strftime("%Y-%m-%d")
    if _day["date"] != today:
        _day.update(date=today, count=0, usd=0.0)
    if _day["count"] >= LIVE_DAILY_CAP or _day["usd"] >= LIVE_BUDGET_USD:
        raise HTTPException(429, "Today's live-demo budget is used up. The precomputed decisions are still shown.")
    q.append(now)
    _day["count"] += 1


# --- routes ------------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"ok": True, "clips": list(RESULTS), "fast_model": FAST_MODEL, "strong_model": STRONG_MODEL,
            "open_vlm": _open_vlm.available, "live_calls_today": _day["count"],
            "live_usd_today": round(_day["usd"], 4), "live_budget_usd": LIVE_BUDGET_USD,
            "live_strong_model": LIVE_ALLOW_STRONG}


class TriageRequest(BaseModel):
    clip_id: str
    track_id: int
    engine: str = "claude"  # "claude" (cascade) or "open" (fine-tuned open VLM via vLLM)


@app.post("/triage")
def triage(body: TriageRequest, request: Request) -> dict:
    clip = RESULTS.get(body.clip_id)
    if not clip:
        raise HTTPException(404, "unknown clip")
    track = next((t for t in clip["tracks"] if t["id"] == body.track_id), None)
    if not track:
        raise HTTPException(404, "unknown track")
    guard(request)
    ev = Evidence.from_track(clip["clip"], track, PUBLIC)
    if body.engine == "open":
        if not _open_vlm.available:
            raise HTTPException(503, "Open VLM endpoint is not configured.")
        return _open_vlm.triage(ev)
    record = verifier().triage(ev, allow_strong=LIVE_ALLOW_STRONG, allow_audit=False)
    _day["usd"] += record.get("cost_usd", 0.0)
    record["live_mode"] = "full cascade" if LIVE_ALLOW_STRONG else "fast model only (public demo budget)"
    return record


class SearchRequest(BaseModel):
    q: str


@app.post("/search")
def search(body: SearchRequest, request: Request) -> dict:
    if not body.q.strip():
        raise HTTPException(400, "empty query")
    if len(body.q) > 200:
        raise HTTPException(400, "query too long")
    parsed = _searcher.cached(body.q)
    if parsed is None:
        guard(request)
        _day["usd"] += SEARCH_COST_ESTIMATE
    return _searcher.search(body.q)


@app.get("/dataset/export")
def export_dataset() -> StreamingResponse:
    """Corrected dataset in YOLO format: one label file per track-best-frame crop's source frame
    is out of scope for a static demo, so we export crop-level labels + VLM chat samples."""
    from .dataset import build_zip  # local import keeps cold start light

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        build_zip(zf, RESULTS, PUBLIC)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=nearzero-fp-dataset.zip"})
