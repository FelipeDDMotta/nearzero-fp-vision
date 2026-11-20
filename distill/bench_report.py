"""Combine vLLM benchmark predictions with the Claude teacher into bench.json for the web app.

Usage (locally, after downloading distill/out/pred_*.json from Kaggle):
  python distill/bench_report.py --holdout street-night-rain
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "apps" / "web" / "public"
OUT = ROOT / "distill" / "out"

# Rough serving cost for the student: an L4 on Modal (~$0.80/h) at measured throughput.
L4_USD_PER_HOUR = 0.80


def metrics(preds: list[dict], key: str = "pred") -> dict:
    human = [p for p in preds if p["label_source"] == "human"]
    scored = human or [p for p in preds if p["label_source"] == "teacher"]
    acc = sum(p[key] == p["target"] for p in scored) / max(1, len(scored))
    fps = [p for p in scored if p["target"] == "false_positive"]
    fp_reject = sum(p[key] == "false_positive" for p in fps) / max(1, len(fps))
    tps = [p for p in scored if p["target"] == "true_positive"]
    tp_keep = sum(p[key] != "false_positive" for p in tps) / max(1, len(tps))
    agree = sum(p[key] == p["teacher"] for p in preds) / max(1, len(preds))
    return {"n_scored": len(scored), "ground_truth": "human" if human else "teacher",
            "accuracy": round(acc, 3), "fp_rejection": round(fp_reject, 3),
            "tp_retention": round(tp_keep, 3), "teacher_agreement": round(agree, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    args = ap.parse_args()
    rows = []
    teacher_preds = None
    for f in sorted(OUT.glob("pred_*.json")):
        d = json.loads(f.read_text())
        m = metrics(d["preds"])
        teacher_preds = teacher_preds or [{**p, "pred": p["teacher"]} for p in d["preds"]]
        cost_1k = L4_USD_PER_HOUR / 3600 * d["per_sample_ms"] / 1000 * 1000
        rows.append({"name": {"base": "Qwen2.5-VL-3B (zero-shot)", "lora": "Qwen2.5-VL-3B + LoRA (distilled)"}
                     .get(d["tag"], d["tag"]), "tag": d["tag"], **m,
                     "valid_json": round(sum(p["valid_json"] for p in d["preds"]) / max(1, d["n"]), 3),
                     "latency_ms": d["per_sample_ms"], "usd_per_1k": round(cost_1k, 3),
                     "where": "vLLM, 1x T4 (batched)"})
    tri = []
    for cid in args.holdout.split(","):
        clip = json.loads((PUBLIC / "results" / f"{cid.strip()}.json").read_text())
        tri += [t["triage"] for t in clip["tracks"] if t.get("triage")]
    if teacher_preds is not None and tri:
        lat = sorted(r["latency_ms"] for r in tri)[len(tri) // 2]
        rows.insert(0, {"name": "Claude cascade (teacher)", "tag": "teacher", **metrics(teacher_preds),
                        "valid_json": 1.0, "latency_ms": lat,
                        "usd_per_1k": round(1000 * sum(r["cost_usd"] for r in tri) / len(tri), 3),
                        "where": "Anthropic API, tool loop"})
    bench = {"holdout_clip": args.holdout.replace(",", " + "), "rows": rows,
             "notes": ["Student models are scored on a camera they never saw during training.",
                       "Ground truth = human review labels where available, else teacher labels.",
                       "Student latency is batched throughput on one T4; teacher latency is a "
                       "sequential agent loop with tool calls, so they are not directly comparable."]}
    (PUBLIC / "results" / "bench.json").write_text(json.dumps(bench, indent=1))
    for r in rows:
        print(f"{r['name']:40s} acc={r['accuracy']} fp_rej={r['fp_rejection']} agree={r['teacher_agreement']} "
              f"lat={r['latency_ms']}ms $/1k={r['usd_per_1k']}")


if __name__ == "__main__":
    main()
