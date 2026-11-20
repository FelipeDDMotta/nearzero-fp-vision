"""Smoke tests that run without network or an API key (the verifier is stubbed)."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

# 1x1 white JPEG
JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00fbd3ffd9"
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    public = tmp_path / "public"
    (public / "results").mkdir(parents=True)
    (public / "crops" / "demo").mkdir(parents=True)
    for name in ("1_best", "1_context", "1_f0"):
        (public / "crops" / "demo" / f"{name}.jpg").write_bytes(JPEG)
    track = {
        "id": 1, "cls": "person", "score": 0.4, "conf_max": 0.45, "conf_median": 0.38, "n": 5,
        "t0": 1.0, "t1": 2.0, "band": "uncertain",
        "boxes": [[1.0, 0.1, 0.1, 0.2, 0.3, 0.4], [2.0, 0.1, 0.1, 0.2, 0.3, 0.38]],
        "crops": {"best": "crops/demo/1_best.jpg", "context": "crops/demo/1_context.jpg",
                  "frames": ["crops/demo/1_f0.jpg"]},
        "triage": {"status": "confirmed", "final": {"verdict": "true_positive", "corrected_class": "person",
                   "confidence": 0.9, "reason": "x", "model": "m",
                   "attributes": {"upper_color": "red", "lower_color": "blue", "accessories": ["backpack"],
                                  "vehicle_type": "n/a", "vehicle_color": "n/a", "pose": "walking"}},
                   "steps": [], "audit": None, "disagreement": False, "cost_usd": 0.0, "latency_ms": 1,
                   "prompt_version": "v1"},
    }
    clip = {"clip": {"id": "demo", "title": "Demo", "scene": "test scene", "attribution": "", "video": ""},
            "fps": 10, "width": 100, "height": 100, "duration": 3, "tracks": [track]}
    (public / "results" / "demo.json").write_text(json.dumps(clip))
    monkeypatch.setenv("PUBLIC_DIR", str(public))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    import app.main as main
    main = importlib.reload(main)

    class StubVerifier:
        def triage(self, ev, force_audit=False):
            assert ev.track_id == 1 and ev.cls == "person"
            assert ev.image(ev.best)["source"]["media_type"] == "image/jpeg"
            return {"status": "rejected", "final": {"verdict": "false_positive"}, "steps": [],
                    "cost_usd": 0.0, "latency_ms": 0}

    monkeypatch.setattr(main, "_verifier", StubVerifier())
    main._searcher._parsed["red top backpack"] = {
        "object_class": "person", "upper_color": "red", "lower_color": "any", "accessories": ["backpack"],
        "vehicle_type": "any", "vehicle_color": "any", "pose": "any"}
    main._searcher._parsed["green top"] = {
        "object_class": "person", "upper_color": "green", "lower_color": "any", "accessories": [],
        "vehicle_type": "any", "vehicle_color": "any", "pose": "any"}
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["clips"] == ["demo"]


def test_triage_uses_evidence(client):
    r = client.post("/triage", json={"clip_id": "demo", "track_id": 1})
    assert r.status_code == 200 and r.json()["status"] == "rejected"


def test_triage_unknown_track(client):
    assert client.post("/triage", json={"clip_id": "demo", "track_id": 99}).status_code == 404


def test_open_engine_not_configured(client):
    assert client.post("/triage", json={"clip_id": "demo", "track_id": 1, "engine": "open"}).status_code == 503


def test_search_matches_and_excludes(client):
    hit = client.post("/search", json={"q": "red top backpack"}).json()
    assert hit["count"] == 1 and set(hit["results"][0]["matched"]) == {"upper_color=red", "backpack"}
    miss = client.post("/search", json={"q": "green top"}).json()
    assert miss["count"] == 0


def test_student_parser_is_robust():
    from app.open_vlm import parse_student

    ok = parse_student('noise {"verdict": "false_positive", "confidence": 0.8, "reason": "glare"} tail', "person")
    assert ok["verdict"] == "false_positive" and ok["valid_json"]
    bad = parse_student("I think it is a person", "person")
    assert bad["verdict"] == "uncertain" and not bad["valid_json"]
