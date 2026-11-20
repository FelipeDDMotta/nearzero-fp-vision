"""Verifier agent: decides whether an uncertain detection is real.

Harness design (the part worth discussing):
  * Bounded tool loop. The agent starts with only the tight crop and the detector's own
    numbers. It may *ask* for more evidence (a wider context view, other frames of the same
    track) and must finish by calling submit_verdict. MAX_STEPS caps tokens and latency.
  * Model cascade. A cheap fast model runs first; only "uncertain" or low-confidence
    verdicts are re-run on the strong model. Most of the bill is avoided this way.
  * Independent audit. A deterministic sample of cases gets a second, skeptical opinion
    with a different information path (all evidence up front, no tools). If the auditor
    disagrees, the case goes to a human instead of being auto-resolved. This is how we
    check that "done" really means done.
  * Strict schemas. Every tool uses strict JSON schemas, so verdicts always parse.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

PROMPT_VERSION = "v1"
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-5-5")
STRONG_MODEL = os.getenv("STRONG_MODEL", "claude-opus-5-5")
# USD per 1M tokens (input, output) for cost accounting.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.00, 20.00),
          "claude-sonnet-5-5": (2.00, 10.00)}
MAX_STEPS = 4
ESCALATE_BELOW = 0.70  # fast-model confidence below this -> re-run on the strong model
HUMAN_BELOW = 0.50  # final confidence below this -> a human decides, even if a verdict exists
AUDIT_RATE = 0.15

COLORS = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple",
          "pink", "brown", "beige", "multicolor", "unknown", "n/a"]
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "none"]
ACCESSORIES = ["backpack", "handbag", "hat", "helmet", "phone", "umbrella", "high_vis_vest",
               "carrying_object", "shopping_cart", "none"]
VEHICLE_TYPES = ["sedan", "suv", "pickup", "van", "truck", "bus", "motorcycle", "bicycle",
                 "unknown", "n/a"]
POSES = ["standing", "walking", "sitting", "crouching", "lying", "riding", "unknown", "n/a"]

SYSTEM = """You are the verification agent in a video-analytics pipeline for physical security.
An object detector (YOLO) running on a fixed camera produced a track it is NOT confident about.
Your job: decide whether the track is a real object of the claimed class (true_positive),
not a real object of any supported class (false_positive: shadows, reflections, posters,
mannequins, rain streaks, glare, parts of machinery, background clutter), or genuinely
impossible to decide from the evidence (uncertain).

How to work:
- You start with the tight crop only. Evidence is limited and each extra view costs time.
- Call get_context_frame when the crop alone is ambiguous (scale, scene, occlusion).
- Call get_track_frames when temporal consistency matters (flicker, motion blur, a one-frame blip).
- Finish by calling submit_verdict exactly once. Do not answer in plain text.
- If the object is real but the class is wrong (a cyclist labelled 'car'), the verdict is
  true_positive and corrected_class carries the right class. Use corrected_class 'none' only for false_positive.
- Fill attributes only from what you can actually see; use 'unknown' when not visible and
  'n/a' when the attribute does not apply (vehicle fields on a person, and vice versa).
- confidence is your probability that the verdict is correct (0 to 1). Be calibrated:
  low light, rain and tiny crops should lower it."""

AUDIT_SYSTEM = """You are an independent auditor double-checking another agent's work in a
security video-analytics pipeline. You are deliberately skeptical: detectors in bad weather
and low light produce convincing false alarms, but they also miss-label real people as
'nothing'. Look at all the evidence and give your own verdict by calling submit_verdict once.
Do not try to guess what the other agent said."""


def _enum(values: list[str]) -> dict:
    return {"type": "string", "enum": values}


VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "corrected_class", "confidence", "reason", "attributes"],
    "properties": {
        "verdict": _enum(["true_positive", "false_positive", "uncertain"]),
        "corrected_class": _enum(CLASSES),
        "confidence": {"type": "number", "description": "0..1 probability the verdict is right"},
        "reason": {"type": "string", "description": "one or two sentences citing visual evidence"},
        "attributes": {
            "type": "object",
            "additionalProperties": False,
            "required": ["upper_color", "lower_color", "accessories", "vehicle_type",
                         "vehicle_color", "pose"],
            "properties": {
                "upper_color": _enum(COLORS),
                "lower_color": _enum(COLORS),
                "accessories": {"type": "array", "items": _enum(ACCESSORIES)},
                "vehicle_type": _enum(VEHICLE_TYPES),
                "vehicle_color": _enum(COLORS),
                "pose": _enum(POSES),
            },
        },
    },
}

_EVIDENCE_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["why"],
    "properties": {"why": {"type": "string", "description": "what you expect this evidence to resolve"}},
}

TOOLS = [
    {"name": "get_context_frame", "strict": True, "input_schema": _EVIDENCE_INPUT,
     "description": "Wider view of the scene around the detection, from the track's most "
                    "confident frame. The detection box is drawn in yellow."},
    {"name": "get_track_frames", "strict": True, "input_schema": _EVIDENCE_INPUT,
     "description": "Tight crops from the first, middle and last frames of the track, with "
                    "timestamps and detector confidence, to check temporal consistency."},
    {"name": "submit_verdict", "strict": True, "input_schema": VERDICT_SCHEMA,
     "description": "Submit the final verdict for this track. Call exactly once, last."},
]
SUBMIT_ONLY = [TOOLS[2]]


@dataclass
class Evidence:
    """Everything the agent may look at for one track. Paths are relative to public_dir."""
    clip_id: str
    track_id: int
    cls: str
    score: float
    n_frames: int
    duration_s: float
    conf_min: float
    conf_median: float
    conf_max: float
    scene: str
    best: str
    context: str
    frames: list[str]
    frame_meta: list[str] = field(default_factory=list)
    public_dir: Path = Path(".")
    source: str = "tracker"

    @classmethod
    def from_track(cls, clip: dict, track: dict, public_dir: Path) -> "Evidence":
        confs = [b[5] for b in track["boxes"]]
        picks = ([track["boxes"][0], track["boxes"][len(track["boxes"]) // 2], track["boxes"][-1]]
                 if len(track["boxes"]) >= 3 else track["boxes"])
        return cls(
            clip_id=clip["id"], track_id=track["id"], cls=track["cls"], score=track["score"],
            n_frames=track["n"], duration_s=round(track["t1"] - track["t0"], 2),
            conf_min=min(confs), conf_median=track["conf_median"], conf_max=track["conf_max"],
            scene=clip["scene"], best=track["crops"]["best"], context=track["crops"]["context"],
            frames=track["crops"]["frames"],
            frame_meta=[f"t={p[0]:.1f}s conf={p[5]:.2f}" for p in picks],
            public_dir=public_dir,
            source=track.get("source", "tracker"),
        )

    def image(self, rel: str) -> dict:
        data = (self.public_dir / rel).read_bytes()
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.standard_b64encode(data).decode()}}

    def digest(self) -> str:
        h = hashlib.sha256((self.public_dir / self.best).read_bytes())
        h.update(f"{self.cls}|{self.scene}|{self.n_frames}".encode())
        return h.hexdigest()[:16]

    def brief(self) -> str:
        if self.source == "low_conf_mining":
            return (f"Camera scene: {self.scene}\n"
                    f"Detector claim: class='{self.cls}' with LOW confidence {self.score:.2f} in a single "
                    f"frame. The tracker did not confirm it. It would only raise an alert if the "
                    f"threshold were lowered, so it may well be background clutter, but it could also "
                    f"be a real object the detector nearly missed.")
        return (f"Camera scene: {self.scene}\n"
                f"Detector claim: class='{self.cls}', track score={self.score:.2f} "
                f"(per-frame confidence min {self.conf_min:.2f} / median {self.conf_median:.2f} / "
                f"max {self.conf_max:.2f}), seen in {self.n_frames} sampled frames over "
                f"{self.duration_s:.1f}s.")


def _cost(model: str, usage: Any) -> float:
    pin, pout = PRICES.get(model, (4.0, 20.0))
    inp = (usage.input_tokens or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0) \
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    return (inp * pin + (usage.output_tokens or 0) * pout) / 1e6


def _clean_verdict(v: dict) -> dict:
    v = dict(v)
    v["confidence"] = float(min(1.0, max(0.0, v.get("confidence", 0.0))))
    if v.get("verdict") == "false_positive":
        v["corrected_class"] = "none"
    return v


class Verifier:
    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()

    def _call(self, model: str, system: str, tools: list, messages: list) -> Any:
        return self.client.messages.create(
            model=model, max_tokens=4000, system=system, tools=tools, messages=messages,
            output_config={"effort": "low"},
        )

    def verify(self, ev: Evidence, model: str) -> dict:
        """One agent run on one model: the bounded evidence-gathering loop."""
        t0 = time.time()
        messages: list = [{"role": "user", "content": [
            {"type": "text", "text": ev.brief() + "\n\nTight crop of the most confident frame:"},
            ev.image(ev.best),
            {"type": "text", "text": "Verify this detection. Request evidence only if you need it."},
        ]}]
        tools_used: list[dict] = []
        tokens_in = tokens_out = 0
        cost = 0.0
        verdict = None
        for step in range(MAX_STEPS):
            last = step == MAX_STEPS - 1
            resp = self._call(model, SYSTEM, SUBMIT_ONLY if last else TOOLS, messages)
            tokens_in += resp.usage.input_tokens or 0
            tokens_out += resp.usage.output_tokens or 0
            cost += _cost(model, resp.usage)
            if resp.stop_reason == "refusal":
                break
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                messages.append({"role": "user", "content": "Call submit_verdict now."})
                continue
            results = []
            for u in uses:
                if u.name == "submit_verdict":
                    verdict = _clean_verdict(u.input)
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": "recorded"})
                elif u.name == "get_context_frame":
                    tools_used.append({"tool": u.name, "why": u.input.get("why", "")})
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": [
                        ev.image(ev.context),
                        {"type": "text", "text": "Context view; the detection is the yellow box."}]})
                elif u.name == "get_track_frames":
                    tools_used.append({"tool": u.name, "why": u.input.get("why", "")})
                    content: list = []
                    for rel, meta in zip(ev.frames, ev.frame_meta):
                        content += [{"type": "text", "text": meta}, ev.image(rel)]
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": content})
                else:
                    results.append({"type": "tool_result", "tool_use_id": u.id,
                                    "content": f"unknown tool {u.name}", "is_error": True})
            if verdict is not None:
                break
            if step == MAX_STEPS - 2:
                results.append({"type": "text", "text": "Evidence budget exhausted. Call submit_verdict now."})
            messages.append({"role": "user", "content": results})
        if verdict is None:
            verdict = {"verdict": "uncertain", "corrected_class": ev.cls, "confidence": 0.0,
                       "reason": "Agent did not return a verdict within the step budget.",
                       "attributes": {}}
        return {"model": model, **verdict, "tools_used": tools_used,
                "input_tokens": tokens_in, "output_tokens": tokens_out,
                "latency_ms": int((time.time() - t0) * 1000), "cost_usd": round(cost, 6)}

    def audit(self, ev: Evidence, model: str = FAST_MODEL) -> dict:
        """Independent single-shot second opinion with all evidence up front."""
        t0 = time.time()
        content: list = [{"type": "text", "text": ev.brief() + "\n\nTight crop:"}, ev.image(ev.best),
                         {"type": "text", "text": "Context (detection in yellow):"}, ev.image(ev.context)]
        for rel, meta in zip(ev.frames, ev.frame_meta):
            content += [{"type": "text", "text": f"Track frame {meta}:"}, ev.image(rel)]
        content.append({"type": "text", "text": "Give your verdict with submit_verdict."})
        messages = [{"role": "user", "content": content}]
        verdict = None
        tokens_in = tokens_out = 0
        cost = 0.0
        for _ in range(2):
            resp = self._call(model, AUDIT_SYSTEM, SUBMIT_ONLY, messages)
            tokens_in += resp.usage.input_tokens or 0
            tokens_out += resp.usage.output_tokens or 0
            cost += _cost(model, resp.usage)
            uses = [b for b in resp.content if b.type == "tool_use" and b.name == "submit_verdict"]
            if uses:
                verdict = _clean_verdict(uses[0].input)
                break
            messages += [{"role": "assistant", "content": resp.content},
                         {"role": "user", "content": "Call submit_verdict now."}]
        verdict = verdict or {"verdict": "uncertain", "confidence": 0.0, "reason": "no verdict"}
        return {"model": model, "verdict": verdict["verdict"], "confidence": verdict["confidence"],
                "reason": verdict.get("reason", ""), "input_tokens": tokens_in,
                "output_tokens": tokens_out, "latency_ms": int((time.time() - t0) * 1000),
                "cost_usd": round(cost, 6)}

    def describe(self, ev: Evidence, model: str = FAST_MODEL) -> dict:
        """Attributes only, for confident tracks that skip verification (feeds search)."""
        t0 = time.time()
        attr_schema = VERDICT_SCHEMA["properties"]["attributes"]
        tool = {"name": "submit_attributes", "strict": True, "input_schema": attr_schema,
                "description": "Visible appearance attributes of the detected object."}
        messages = [{"role": "user", "content": [
            {"type": "text", "text": f"Detected {ev.cls} on a security camera ({ev.scene})."},
            ev.image(ev.best),
            {"type": "text", "text": "Call submit_attributes with what is visible. Use 'unknown' when "
                                     "not visible and 'n/a' when a field does not apply."}]}]
        resp = self._call(model, "You extract appearance attributes for forensic video search.",
                          [tool], messages)
        use = next((b for b in resp.content if b.type == "tool_use"), None)
        return {"model": model, "attributes": dict(use.input) if use else {},
                "cost_usd": round(_cost(model, resp.usage), 6),
                "latency_ms": int((time.time() - t0) * 1000)}

    def triage(self, ev: Evidence, force_audit: bool = False) -> dict:
        """Cascade + audit + escalation policy. Returns the full decision record."""
        steps = [self.verify(ev, FAST_MODEL)]
        final = steps[0]
        if final["verdict"] == "uncertain" or final["confidence"] < ESCALATE_BELOW:
            steps.append(self.verify(ev, STRONG_MODEL))
            final = steps[-1]
        audit = None
        disagreement = False
        sampled = int(ev.digest(), 16) % 1000 < AUDIT_RATE * 1000
        if (sampled or force_audit) and final["verdict"] != "uncertain":
            audit = self.audit(ev)
            disagreement = audit["verdict"] != "uncertain" and audit["verdict"] != final["verdict"]
        if final["verdict"] == "uncertain" or disagreement or final["confidence"] < HUMAN_BELOW:
            status = "escalated"
        elif final["verdict"] == "true_positive":
            status = "confirmed"
        else:
            status = "rejected"
        parts = steps + ([audit] if audit else [])
        return {
            "status": status,
            "final": {k: final.get(k) for k in
                      ("verdict", "corrected_class", "confidence", "reason", "attributes", "model")},
            "steps": steps,
            "audit": audit,
            "disagreement": disagreement,
            "cost_usd": round(sum(p["cost_usd"] for p in parts), 6),
            "latency_ms": sum(p["latency_ms"] for p in parts),
            "prompt_version": PROMPT_VERSION,
        }


class TriageCache:
    """JSON-file cache keyed by evidence digest + prompt version + models."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}

    def key(self, ev: Evidence) -> str:
        return f"{PROMPT_VERSION}:{FAST_MODEL}:{STRONG_MODEL}:{ev.digest()}"

    def get(self, ev: Evidence) -> dict | None:
        return self.data.get(self.key(ev))

    def put(self, ev: Evidence, record: dict) -> None:
        self.data[self.key(ev)] = record
        self.put_raw()

    def put_raw(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))
