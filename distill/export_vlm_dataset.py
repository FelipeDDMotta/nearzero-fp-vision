"""Export agent decisions as a chat dataset to distill the verifier into an open VLM.

Teacher labels come from the Claude cascade (auto-resolved cases only); human review labels
override them where they exist. One whole clip is held out for evaluation, so the student
is scored on a camera it never saw (no frame-level leakage between train and test).

Output (distill/data/):
  train.jsonl, test.jsonl   {"id", "images": [best, context], "prompt", "target", "label_source"}
  images/...                copies of the referenced crops (paths in jsonl are relative)

Usage:  python distill/export_vlm_dataset.py --holdout street-night-rain
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
from app.agent import Evidence  # noqa: E402
from app.open_vlm import STUDENT_PROMPT  # noqa: E402

PUBLIC = ROOT / "apps" / "web" / "public"
OUT = ROOT / "distill" / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True, help="clip id kept out of training")
    args = ap.parse_args()

    human_path = PUBLIC / "results" / "human_labels.json"
    human = json.loads(human_path.read_text()) if human_path.exists() else {}
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    splits: dict[str, list] = {"train": [], "test": []}

    for path in sorted((PUBLIC / "results").glob("*.json")):
        if path.stem in ("bench", "human_labels"):
            continue
        res = json.loads(path.read_text())
        clip = res["clip"]
        split = "test" if clip["id"] == args.holdout else "train"
        for t in res["tracks"]:
            tri = t.get("triage")
            if not tri:
                continue
            key = f"{clip['id']}:{t['id']}"
            if key in human:
                label, source = human[key], "human"
            elif tri["status"] in ("confirmed", "rejected"):
                label, source = tri["final"], "teacher"
            elif split == "test":
                label, source = tri["final"], "teacher-escalated"  # kept for teacher-agreement only
            else:
                continue  # escalated + unlabelled: not used for training
            ev = Evidence.from_track(clip, t, PUBLIC)
            images = []
            for rel in (ev.best, ev.context):
                dst = OUT / "images" / rel.replace("crops/", "")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PUBLIC / rel, dst)
                images.append("images/" + rel.replace("crops/", ""))
            target = {"verdict": label["verdict"],
                      "corrected_class": label.get("corrected_class") or t["cls"],
                      "confidence": round(float(label.get("confidence", 1.0)), 2),
                      "reason": label.get("reason", "")}
            prompt = STUDENT_PROMPT.format(scene=ev.scene, cls=ev.cls, score=ev.score,
                                           n=ev.n_frames, dur=ev.duration_s)
            splits[split].append({
                "id": key, "images": images, "label_source": source, "prompt": prompt,
                "target": json.dumps(target),
                "teacher": tri["final"]["verdict"], "teacher_status": tri["status"],
            })
            if split == "train":
                # Augmentation: other frames of the same track are new views with the same label.
                for k, rel in enumerate(ev.frames):
                    dst = OUT / "images" / rel.replace("crops/", "")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(PUBLIC / rel, dst)
                    splits[split].append({
                        "id": f"{key}#f{k}", "images": ["images/" + rel.replace("crops/", ""), images[1]],
                        "label_source": source, "prompt": prompt, "target": json.dumps(target),
                        "teacher": tri["final"]["verdict"], "teacher_status": tri["status"],
                    })

    assert not {r["id"].split(":")[0] for r in splits["train"]} & {args.holdout}, "holdout leaked into train"
    for name, rows in splits.items():
        (OUT / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        dist = Counter(json.loads(r["target"])["verdict"] for r in rows)
        print(f"{name}: {len(rows)} samples {dict(dist)} sources={dict(Counter(r['label_source'] for r in rows))}")


if __name__ == "__main__":
    main()
