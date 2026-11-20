"""Stage 2: run the verifier agent on every 'uncertain' track and write verdicts back.

Usage:  python pipeline/run_triage.py [--only street] [--limit 40] [--workers 6]
Results are cached by evidence digest, so re-runs only pay for new or changed tracks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
load_dotenv(ROOT / ".env")

from app.agent import Evidence, TriageCache, Verifier  # noqa: E402
sys.path.insert(0, str(ROOT / "pipeline"))
from write_index import write_index  # noqa: E402

PUBLIC = ROOT / "apps" / "web" / "public"


def summarise(tracks: list[dict]) -> dict:
    bands = Counter(t["band"] for t in tracks)
    tri = [t["triage"] for t in tracks if t.get("triage")]
    status = Counter(r["status"] for r in tri)
    n_unc = bands.get("uncertain", 0)
    cands = [t for t in tracks if t.get("source") == "low_conf_mining"]
    tracker = [t for t in tracks if t.get("source") != "low_conf_mining"]
    return {
        "tracker_tracks": len(tracker),
        "tracker_auto": sum(t["band"] == "auto_accept" for t in tracker),
        "tracker_uncertain": sum(t["band"] == "uncertain" for t in tracker),
        "candidates": len(cands),
        "candidates_confirmed": sum((t.get("triage") or {}).get("status") == "confirmed" for t in cands),
        "candidates_rejected": sum((t.get("triage") or {}).get("status") == "rejected" for t in cands),
        "tracks": len(tracks),
        "bands": dict(bands),
        "triaged": len(tri),
        "status": dict(status),
        "auto_resolved_pct": round(100 * (status["confirmed"] + status["rejected"]) / max(1, len(tri)), 1),
        "escalated_pct": round(100 * status["escalated"] / max(1, len(tri)), 1),
        # Operator alert load if every uncertain item alerted (i.e. threshold lowered to catch them all)
        "alerts_without_agent": bands.get("auto_accept", 0) + n_unc,
        "alerts_with_agent": bands.get("auto_accept", 0) + status["confirmed"] + status["escalated"],
        "strong_model_calls": sum(1 for r in tri if len(r["steps"]) > 1),
        "audits": sum(1 for r in tri if r.get("audit")),
        "disagreements": sum(1 for r in tri if r.get("disagreement")),
        "cost_usd": round(sum(r["cost_usd"] for r in tri), 4),
        "cost_per_event_usd": round(sum(r["cost_usd"] for r in tri) / max(1, len(tri)), 5),
        "latency_ms_p50": sorted(r["latency_ms"] for r in tri)[len(tri) // 2] if tri else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0, help="max uncertain tracks per clip (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--describe", action="store_true", help="also extract attributes for auto-accepted tracks")
    args = ap.parse_args()
    only = {c for c in args.only.split(",") if c}

    verifier = Verifier()
    cache = TriageCache(ROOT / "data" / "cache" / "triage.json")

    for path in sorted((PUBLIC / "results").glob("*.json")):
        if path.stem in ("bench", "index", "human_labels") or (only and path.stem not in only):
            continue
        result = json.loads(path.read_text())
        todo = [t for t in result["tracks"] if t["band"] == "uncertain"]
        # Longest tracks first: they are the ones an operator would actually see.
        todo.sort(key=lambda t: -t["n"])
        if args.limit:
            todo = todo[: args.limit]
        t_start = time.time()
        fresh = 0

        def work(track: dict) -> tuple[dict, dict, bool]:
            ev = Evidence.from_track(result["clip"], track, PUBLIC)
            hit = cache.get(ev)
            if hit:
                return track, hit, False
            return track, verifier.triage(ev), True

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, t) for t in todo]
            for fut in as_completed(futures):
                try:
                    track, record, was_fresh = fut.result()
                except Exception as exc:  # keep the batch going; log and move on
                    print(f"  ! {path.stem}: {type(exc).__name__}: {exc}")
                    continue
                track["triage"] = record
                if was_fresh:
                    fresh += 1
                    cache.put(Evidence.from_track(result["clip"], track, PUBLIC), record)
        if args.describe:
            confident = [t for t in result["tracks"] if t["band"] == "auto_accept" and not t.get("describe")]

            def describe(track: dict) -> tuple[dict, dict]:
                ev = Evidence.from_track(result["clip"], track, PUBLIC)
                key = "describe:" + cache.key(ev)
                hit = cache.data.get(key)
                if hit:
                    return track, hit
                rec = verifier.describe(ev)
                cache.data[key] = rec
                return track, rec

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for fut in as_completed([pool.submit(describe, t) for t in confident]):
                    try:
                        track, rec = fut.result()
                        track["describe"] = rec
                    except Exception as exc:
                        print(f"  ! describe {path.stem}: {type(exc).__name__}: {exc}")
            cache.put_raw()
        result["summary"] = summarise(result["tracks"])
        path.write_text(json.dumps(result))
        s = result["summary"]
        print(f"[{path.stem}] triaged {s['triaged']} ({fresh} fresh) status={s['status']} "
              f"cost=${s['cost_usd']} in {time.time() - t_start:.0f}s")
        write_index()


if __name__ == "__main__":
    main()
