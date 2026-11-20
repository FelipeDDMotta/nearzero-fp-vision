"""Benchmark one model served by vLLM on the held-out test split.

Run once per model in its own process (vLLM does not reliably free GPU memory in-process):
  python distill/bench_vllm.py --model Qwen/Qwen2.5-VL-3B-Instruct --tag base
  python distill/bench_vllm.py --model ./merged --tag lora
Each run writes distill/out/pred_<tag>.json; summarise with distill/bench_report.py.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "apps" / "api"))
from app.open_vlm import parse_student  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=str(HERE / "data"))
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    data = Path(args.data)
    rows = [json.loads(l) for l in (data / "test.jsonl").read_text().splitlines() if l.strip()]
    llm = LLM(model=args.model, dtype="half", max_model_len=4096, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 2})
    params = SamplingParams(temperature=0, max_tokens=200)

    def msg(r: dict) -> list[dict]:
        imgs = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                 + base64.b64encode((data / p).read_bytes()).decode()}} for p in r["images"]]
        return [{"role": "user", "content": imgs + [{"type": "text", "text": r["prompt"]}]}]

    t0 = time.time()
    outs = llm.chat([msg(r) for r in rows], params)
    wall = time.time() - t0
    preds = []
    n_tokens = 0
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        n_tokens += len(o.outputs[0].token_ids)
        p = parse_student(text, "unknown")
        preds.append({"id": r["id"], "pred": p["verdict"], "confidence": p["confidence"],
                      "valid_json": p["valid_json"], "raw": text[:300],
                      "target": json.loads(r["target"])["verdict"], "label_source": r["label_source"],
                      "teacher": r["teacher"]})
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / f"pred_{args.tag}.json").write_text(json.dumps({
        "tag": args.tag, "model": args.model, "n": len(rows), "wall_s": round(wall, 2),
        "per_sample_ms": round(1000 * wall / max(1, len(rows)), 1),
        "output_tokens_per_s": round(n_tokens / max(wall, 1e-6), 1), "preds": preds}, indent=1))
    print(f"[{args.tag}] {len(rows)} samples in {wall:.1f}s")


if __name__ == "__main__":
    main()
