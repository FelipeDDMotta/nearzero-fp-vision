"""Natural-language appearance search over attributes extracted by the agents.

The LLM only translates the query into a structured filter; matching itself is plain,
deterministic code. That keeps search fast, explainable ("matched: upper=red, backpack")
and cheap: one small-model call per new query, cached.
"""
from __future__ import annotations

import os
from typing import Any

import anthropic

from .agent import ACCESSORIES, COLORS, FAST_MODEL, POSES, VEHICLE_TYPES

ANY = "any"
FILTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["object_class", "upper_color", "lower_color", "accessories", "vehicle_type",
                 "vehicle_color", "pose"],
    "properties": {
        "object_class": {"type": "string", "enum": ["person", "vehicle", "bicycle", ANY]},
        "upper_color": {"type": "string", "enum": [c for c in COLORS if c not in ("unknown", "n/a")] + [ANY]},
        "lower_color": {"type": "string", "enum": [c for c in COLORS if c not in ("unknown", "n/a")] + [ANY]},
        "accessories": {"type": "array", "items": {"type": "string", "enum": [a for a in ACCESSORIES if a != "none"]}},
        "vehicle_type": {"type": "string", "enum": [v for v in VEHICLE_TYPES if v not in ("unknown", "n/a")] + [ANY]},
        "vehicle_color": {"type": "string", "enum": [c for c in COLORS if c not in ("unknown", "n/a")] + [ANY]},
        "pose": {"type": "string", "enum": [p for p in POSES if p not in ("unknown", "n/a")] + [ANY]},
    },
}
TOOL = {"name": "apply_filter", "strict": True, "input_schema": FILTER_SCHEMA,
        "description": "Structured attribute filter for the video archive. Use 'any' for "
                       "attributes the query does not mention."}
SYSTEM = ("You translate an investigator's natural-language search over security-camera footage "
          "into a structured attribute filter. Call apply_filter exactly once. Map synonyms "
          "(jacket/shirt/top -> upper_color; trousers/jeans/skirt -> lower_color; grey -> gray; "
          "bag -> handbag unless it says backpack; vest -> high_vis_vest when it implies safety gear). "
          "'car', 'truck', 'van' and similar are object_class 'vehicle' with the matching vehicle_type.")
VEHICLES = {"car", "truck", "bus", "motorcycle"}


class Searcher:
    def __init__(self, results: dict[str, dict]):
        self.results = results
        self._parsed: dict[str, dict] = {}
        self._client: anthropic.Anthropic | None = None

    def cached(self, q: str) -> dict | None:
        return self._parsed.get(q.strip().lower())

    def parse(self, q: str) -> dict:
        key = q.strip().lower()
        if key in self._parsed:
            return self._parsed[key]
        self._client = self._client or anthropic.Anthropic()
        resp = self._client.messages.create(
            model=os.getenv("SEARCH_MODEL", FAST_MODEL), max_tokens=2000, system=SYSTEM,
            tools=[TOOL], output_config={"effort": "low"},
            messages=[{"role": "user", "content": f"Query: {q}"}],
        )
        use = next((b for b in resp.content if b.type == "tool_use"), None)
        flt = dict(use.input) if use else {k: ANY for k in FILTER_SCHEMA["required"]} | {"accessories": []}
        self._parsed[key] = flt
        return flt

    @staticmethod
    def _attrs(track: dict) -> dict | None:
        tri = track.get("triage")
        if tri and tri["final"].get("attributes") and tri["status"] != "rejected":
            return tri["final"]["attributes"]
        return (track.get("describe") or {}).get("attributes")

    def search(self, q: str, limit: int = 48) -> dict:
        flt = self.parse(q)
        hits: list[dict[str, Any]] = []
        for clip_id, res in self.results.items():
            for t in res["tracks"]:
                attrs = self._attrs(t)
                if not attrs:
                    continue
                cls = (t.get("triage") or {}).get("final", {}).get("corrected_class") or t["cls"]
                score, matched, ok = self._score(flt, cls, attrs)
                if ok:
                    hits.append({"clip_id": clip_id, "track_id": t["id"], "cls": cls,
                                 "crop": t["crops"]["best"], "t0": t["t0"], "t1": t["t1"],
                                 "attributes": attrs, "matched": matched, "score": score,
                                 "status": (t.get("triage") or {}).get("status", t["band"])})
        hits.sort(key=lambda h: (-h["score"], -(h["t1"] - h["t0"])))
        return {"query": q, "filter": flt, "count": len(hits), "results": hits[:limit]}

    @staticmethod
    def _score(flt: dict, cls: str, a: dict) -> tuple[float, list[str], bool]:
        want_cls = flt.get("object_class", ANY)
        if want_cls == "person" and cls != "person":
            return 0, [], False
        if want_cls == "vehicle" and cls not in VEHICLES:
            return 0, [], False
        if want_cls == "bicycle" and cls != "bicycle":
            return 0, [], False
        score, matched = 0.0, []
        for field in ("upper_color", "lower_color", "vehicle_type", "vehicle_color", "pose"):
            want = flt.get(field, ANY)
            if want == ANY:
                continue
            have = a.get(field, "unknown")
            if have == want:
                score += 1
                matched.append(f"{field}={want}")
            elif have in ("unknown",):
                score += 0.2  # cannot rule it out
            else:
                return 0, [], False
        for acc in flt.get("accessories", []):
            if acc in (a.get("accessories") or []):
                score += 1
                matched.append(acc)
            else:
                return 0, [], False
        return score, matched, True
