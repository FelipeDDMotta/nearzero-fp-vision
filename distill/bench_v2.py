"""v2 benchmark: one model served by vLLM on distill/data_v2/test.jsonl (leak-free, by source video).

  python distill/bench_v2.py --model unsloth/Qwen2.5-VL-3B-Instruct --tag v2_base
  python distill/bench_v2.py --model ./merged_v2 --tag v2_lora
Writes distill/out/pred_<tag>.json (run each model in its own process).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse(text: str) -> tuple[str, bool]:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        v = {}
    verdict = v.get("verdict")
    return (verdict if verdict in ("true_positive", "false_positive") else "uncertain"), bool(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    data = HERE / "data_v2"
    rows = [json.loads(l) for l in (data / "test.jsonl").read_text().splitlines() if l.strip()]
    llm = LLM(model=args.model, dtype="half", max_model_len=4096, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 2})
    params = SamplingParams(temperature=0, max_tokens=120)

    def msg(r):
        imgs = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                 + base64.b64encode((data / p).read_bytes()).decode()}} for p in r["images"]]
        return [{"role": "user", "content": imgs + [{"type": "text", "text": r["prompt"]}]}]

    t0 = time.time()
    outs = llm.chat([msg(r) for r in rows], params)
    wall = time.time() - t0
    preds = []
    for r, o in zip(rows, outs):
        verdict, valid = parse(o.outputs[0].text)
        preds.append({"id": r["id"], "group": r["group"], "difficulty": r["difficulty"],
                      "label_source": r["label_source"], "target": r["label"], "pred": verdict,
                      "valid_json": valid, "raw": o.outputs[0].text[:200]})
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / f"pred_{args.tag}.json").write_text(json.dumps({
        "tag": args.tag, "model": args.model, "n": len(rows), "wall_s": round(wall, 2),
        "per_sample_ms": round(1000 * wall / max(1, len(rows)), 1), "preds": preds}, indent=1))
    hard = [p for p in preds if p["difficulty"] == "hard"]
    print(args.tag, "hard acc", round(sum(p["pred"] == p["target"] for p in hard) / max(1, len(hard)), 3), "n", len(hard),
          "| all acc", round(sum(p["pred"] == p["target"] for p in preds) / max(1, len(preds)), 3), "n", len(preds))


if __name__ == "__main__":
    main()
