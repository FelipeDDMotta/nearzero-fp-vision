"""Write apps/web/public/results/index.json (clip list + summaries) for the web app."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import os

RESULTS = Path(os.getenv("NZ_PUBLIC", ROOT / "apps" / "web" / "public")) / "results"
ORDER = [c["id"] for c in json.loads(Path(os.getenv("NZ_CLIPS", ROOT / "pipeline" / "clips.json")).read_text())]


def write_index() -> None:
    entries = []
    for p in RESULTS.glob("*.json"):
        if p.stem in ("index", "bench", "bench_v2", "human_labels"):
            continue
        r = json.loads(p.read_text())
        entries.append({"clip": r["clip"], "duration": r["duration"], "summary": r.get("summary")})
    entries.sort(key=lambda e: ORDER.index(e["clip"]["id"]) if e["clip"]["id"] in ORDER else 99)
    (RESULTS / "index.json").write_text(json.dumps(entries))


if __name__ == "__main__":
    write_index()
