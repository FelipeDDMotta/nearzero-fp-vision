"""Client for the distilled open VLM (Qwen-VL + LoRA) served by vLLM on Modal.

vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint; we call it with plain
HTTP. The student was trained to emit the same verdict JSON the Claude teacher produces,
so its output plugs into the same decision record.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time

import httpx

from .agent import Evidence

STUDENT_PROMPT = (
    "You verify uncertain detections from a security camera. Scene: {scene}. "
    "Detector claim: class='{cls}', track score={score:.2f}, {n} frames over {dur:.1f}s. "
    "Image 1 is the tight crop, image 2 the context view (detection in yellow). "
    "Reply with JSON only: {{\"verdict\": \"true_positive|false_positive|uncertain\", "
    "\"corrected_class\": \"...\", \"confidence\": 0-1, \"reason\": \"...\"}}"
)


def student_messages(ev: Evidence) -> list[dict]:
    """Same prompt format used for the fine-tuning dataset (see distill/export_vlm_dataset.py)."""
    def b64(rel: str) -> str:
        return base64.standard_b64encode((ev.public_dir / rel).read_bytes()).decode()

    text = STUDENT_PROMPT.format(scene=ev.scene, cls=ev.cls, score=ev.score, n=ev.n_frames,
                                 dur=ev.duration_s)
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(ev.best)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(ev.context)}"}},
        {"type": "text", "text": text},
    ]}]


def parse_student(text: str, default_cls: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        v = {}
    verdict = v.get("verdict") if v.get("verdict") in ("true_positive", "false_positive", "uncertain") else "uncertain"
    try:
        conf = float(v.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {"verdict": verdict, "corrected_class": v.get("corrected_class", default_cls),
            "confidence": max(0.0, min(1.0, conf)), "reason": str(v.get("reason", ""))[:400],
            "valid_json": bool(v)}


class OpenVLM:
    def __init__(self, url: str | None, token: str | None, model: str):
        self.url, self.token, self.model = url, token, model

    @classmethod
    def from_env(cls) -> "OpenVLM":
        return cls(os.getenv("OPEN_VLM_URL"), os.getenv("OPEN_VLM_TOKEN"),
                   os.getenv("OPEN_VLM_MODEL", "nearzero-verifier"))

    @property
    def available(self) -> bool:
        return bool(self.url)

    def triage(self, ev: Evidence) -> dict:
        t0 = time.time()
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        r = httpx.post(f"{self.url.rstrip('/')}/v1/chat/completions", headers=headers, timeout=180,
                       json={"model": self.model, "messages": student_messages(ev),
                             "max_tokens": 200, "temperature": 0})
        r.raise_for_status()
        body = r.json()
        v = parse_student(body["choices"][0]["message"]["content"], ev.cls)
        usage = body.get("usage", {})
        status = {"true_positive": "confirmed", "false_positive": "rejected"}.get(v["verdict"], "escalated")
        if v["confidence"] < 0.5 and status != "escalated":
            status = "escalated"
        step = {"model": f"open-vlm:{self.model}", **v, "tools_used": [],
                "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                "latency_ms": int((time.time() - t0) * 1000), "cost_usd": 0.0}
        return {"status": status,
                "final": {k: step[k] for k in ("verdict", "corrected_class", "confidence", "reason", "model")}
                | {"attributes": {}},
                "steps": [step], "audit": None, "disagreement": False, "cost_usd": 0.0,
                "latency_ms": step["latency_ms"], "prompt_version": "student-v1"}
