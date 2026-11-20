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
    """Score on decidable items only (target is true/false positive). The ground truth is human
    labels when any exist, otherwise teacher labels (and then the teacher can't be scored)."""
    decidable = [p for p in preds if p["target"] in ("true_positive", "false_positive")]
    human = [p for p in decidable if p["label_source"] == "human"]
    scored = human or decidable
    acc = sum(p[key] == p["target"] for p in scored) / max(1, len(scored))
    fps = [p for p in scored if p["target"] == "false_positive"]
    fp_reject = sum(p[key] == "false_positive" for p in fps) / max(1, len(fps))
    tps = [p for p in scored if p["target"] == "true_positive"]
    tp_keep = sum(p[key] != "false_positive" for p in tps) / max(1, len(tps))
    agree = sum(p[key] == p["teacher"] for p in preds) / max(1, len(preds))
    return {"n_scored": len(scored), "ground_truth": "human" if human else "teacher",
            "accuracy": round(acc, 3), "fp_rejection": round(fp_reject, 3),
            "tp_retention": round(tp_keep, 3), "teacher_agreement": round(agree, 3),
            "fp_counts": f"{sum(p[key] == 'false_positive' for p in fps)}/{len(fps)}",
            "tp_counts": f"{sum(p[key] != 'false_positive' for p in tps)}/{len(tps)}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    args = ap.parse_args()
    rows = []
    teacher_preds = None
    # Human labels collected after training (review page export) override the ground truth here,
    # so labelling the held-out clips never has to wait for, or touch, the training run.
    hp = PUBLIC / "results" / "human_labels.json"
    human = {k: v for k, v in (json.loads(hp.read_text()) if hp.exists() else {}).items()
             if v.get("verdict") in ("true_positive", "false_positive")}
    for f in sorted(OUT.glob("pred_*.json")):
        d = json.loads(f.read_text())
        for p in d["preds"]:
            if p["id"] in human:
                p["target"], p["label_source"] = human[p["id"]]["verdict"], "human"
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
        tm = metrics(teacher_preds)
        if tm["ground_truth"] == "teacher":  # the teacher IS the ground truth: don't report a fake 100%
            tm.update(accuracy=None, fp_rejection=None, tp_retention=None, fp_counts="-", tp_counts="-")
        rows.insert(0, {"name": "Claude cascade (teacher)", "tag": "teacher", **tm,
                        "valid_json": 1.0, "latency_ms": lat,
                        "usd_per_1k": round(1000 * sum(r["cost_usd"] for r in tri) / len(tri), 3),
                        "where": "Anthropic API, tool loop"})
    log = json.loads((OUT / "train_log.json").read_text()) if (OUT / "train_log.json").exists() else []
    losses = [e["loss"] for e in log if "loss" in e]
    final = next((e for e in log if "train_runtime" in e), {})
    training = {"base_model": "Qwen2.5-VL-3B-Instruct", "method": "QLoRA 4-bit, r=16, language layers only (Unsloth)",
                "samples": 168, "epochs": final.get("epoch"), "steps": final.get("step"),
                "runtime_s": round(final.get("train_runtime", 0)), "loss_first": round(losses[0], 2) if losses else None,
                "loss_last": round(losses[-1], 2) if losses else None, "hardware": "1x NVIDIA T4 (Kaggle, free)",
                "hf_repo": "FelipeDdM/nearzero-fp-verifier-qwen2.5-vl-3b"}
    bench = {"holdout_clip": args.holdout.replace(",", " + "), "rows": rows, "training": training,
             "notes": ["Student models are scored on a camera they never saw during training.",
                       "Ground truth = human review labels where available, else teacher labels.",
                       "Student latency is batched throughput on one T4; teacher latency is a "
                       "sequential agent loop with tool calls, so they are not directly comparable.",
                       "Items the teacher itself left 'uncertain' are excluded from accuracy (not decidable)."]}
    (PUBLIC / "results" / "bench.json").write_text(json.dumps(bench, indent=1))
    for r in rows:
        print(f"{r['name']:40s} acc={r['accuracy']} fp={r['fp_counts']} tp={r['tp_counts']} agree={r['teacher_agreement']} "
              f"lat={r['latency_ms']}ms $/1k={r['usd_per_1k']}")


if __name__ == "__main__":
    main()
