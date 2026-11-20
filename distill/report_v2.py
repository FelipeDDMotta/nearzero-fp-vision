"""Build apps/web/public/results/bench_v2.json from the v2 runs (whatever is available).

Inputs (all optional except data_v2/test.jsonl):
  distill/out/pred_v2_base.json, pred_v2_lora.json   (Kaggle, vLLM)
  distill/out/edge_cv.json                           (local, MobileNet)
  distill/out/train_log_v2.json
  apps/web/public/results/human_labels.json          (review queue export)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "distill" / "out"
PUBLIC = ROOT / "apps" / "web" / "public" / "results"
DATA = ROOT / "distill" / "data_v2"
L4_USD_PER_HOUR = 0.80


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def score(preds: list[dict]) -> dict:
    hard = [p for p in preds if p["difficulty"] == "hard"]
    easy = [p for p in preds if p["difficulty"] == "easy"]
    acc = lambda xs: round(sum(p["pred"] == p["target"] for p in xs) / max(1, len(xs)), 3)
    fps = [p for p in hard if p["target"] == "false_positive"]
    tps = [p for p in hard if p["target"] == "true_positive"]
    return {"hard_n": len(hard), "hard_acc": acc(hard), "easy_n": len(easy), "easy_acc": acc(easy),
            "hard_fp_rejected": f"{sum(p['pred'] == 'false_positive' for p in fps)}/{len(fps)}",
            "hard_real_kept": f"{sum(p['pred'] != 'false_positive' for p in tps)}/{len(tps)}"}


def main() -> None:
    test = [json.loads(l) for l in (DATA / "test.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for tag, name in (("v2_base", "Qwen2.5-VL-3B zero-shot"), ("v2_lora", "Qwen2.5-VL-3B + LoRA v2")):
        d = load(OUT / f"pred_{tag}.json")
        if d:
            rows.append({"name": name, "tag": tag, **score(d["preds"]),
                         "valid_json": round(sum(p["valid_json"] for p in d["preds"]) / max(1, d["n"]), 3),
                         "latency_ms": d["per_sample_ms"], "where": "vLLM, 1x T4 (batched)",
                         "usd_per_1k": round(L4_USD_PER_HOUR / 3600 * d["per_sample_ms"], 3), "params": "3.8B"})
    edge = load(OUT / "edge_cv.json")
    if edge:
        f = edge["fixed_split"]["all"]
        rows.append({"name": "MobileNetV3 + logistic regression (edge)", "tag": "edge", **f, "valid_json": 1.0,
                     "latency_ms": edge["cpu_ms_per_image"] * 2, "where": "CPU, laptop", "usd_per_1k": 0.0,
                     "params": "5.4M"})

    # Teacher vs humans: how often the Claude cascade agrees with the human labels.
    human = load(PUBLIC / "human_labels.json") or {}
    human = {k: v for k, v in human.items() if v.get("verdict") in ("true_positive", "false_positive")}
    teacher_vs_human = None
    if human:
        agree = total = 0
        for p in PUBLIC.glob("*.json"):
            if p.stem in ("index", "bench", "bench_v2", "human_labels"):
                continue
            r = json.loads(p.read_text())
            for t in r["tracks"]:
                key = f"{r['clip']['id']}:{t['id']}"
                if key in human and t.get("triage"):
                    total += 1
                    agree += t["triage"]["final"]["verdict"] == human[key]["verdict"]
        teacher_vs_human = {"n": total, "agreement": round(agree / max(1, total), 3)}

    log = load(OUT / "train_log_v2.json") or []
    losses = [e["loss"] for e in log if "loss" in e]
    final = next((e for e in log if "train_runtime" in e), {})
    sources = {}
    for r in test:
        sources[r["label_source"]] = sources.get(r["label_source"], 0) + 1
    bench = {
        "split": "test videos: street + walkway (all synthetic variants); train never sees them",
        "test_hard_n": sum(r["difficulty"] == "hard" for r in test), "test_easy_n": sum(r["difficulty"] == "easy" for r in test),
        "test_label_sources": sources, "rows": rows,
        "edge_cv": edge and edge["cv_leave_one_video_out"],
        "teacher_vs_human": teacher_vs_human,
        "training": {"samples_total_v2": sum(1 for _ in (DATA / "all.jsonl").read_text().splitlines()),
                     "epochs": final.get("epoch"), "steps": final.get("step"),
                     "runtime_s": round(final.get("train_runtime", 0)),
                     "loss_first": round(losses[0], 2) if losses else None,
                     "loss_last": round(losses[-1], 2) if losses else None,
                     "hf_repo": "FelipeDdM/nearzero-fp-verifier-qwen2.5-vl-3b-v2"},
    }
    (PUBLIC / "bench_v2.json").write_text(json.dumps(bench, indent=1))
    print(json.dumps({k: bench[k] for k in ("test_hard_n", "test_easy_n", "teacher_vs_human")}, indent=1))
    for r in rows:
        print(f"{r['name']:45s} hard {r['hard_acc']} (FP {r['hard_fp_rejected']}, real {r['hard_real_kept']}) easy {r['easy_acc']}")


if __name__ == "__main__":
    main()
