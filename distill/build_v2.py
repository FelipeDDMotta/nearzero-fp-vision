"""Dataset v2: more data at zero labelling cost, split by source video (no leakage).

Label sources, from most to least trustworthy:
  human              review-queue labels (apps/web/public/results/human_labels.json)
  no-target-scene    any detection in a scene with no people/vehicles (bolt conveyor) is a FP
  teacher            auto-resolved Claude decisions on uncertain tracks / below-threshold candidates
  detector-confident stable, high-confidence tracker tracks (all views) -> real object
  synthetic-bg       random boxes on background (no track or candidate nearby) -> FP

Difficulty: "hard" = items a real system would actually be unsure about (agent-reviewed items
and no-target-scene detections); "easy" = the free bulk. Report accuracy on the HARD set.

The student prompt deliberately carries no detector score or track length: free positives have
high scores and synthetic negatives have none, so those numbers would be a shortcut.

Usage:  python distill/build_v2.py --test-groups street,walkway
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from detect_track import _crop  # noqa: E402

SITE = ROOT / "apps" / "web" / "public"
V2 = ROOT / "data" / "v2" / "public"
WORKS = {SITE: ROOT / "data" / "work", V2: ROOT / "data" / "v2" / "work"}
OUT = ROOT / "distill" / "data_v2"
SITE_GROUPS = {"street": "street", "street-night-rain": "street", "street-dusk": "street", "store": "store",
               "store-lowlight": "store", "worker-zone": "worker", "worker-zone-lowlight": "worker",
               "parking": "parking", "parking-rain": "parking", "walkway": "walkway", "walkway-dusk": "walkway"}
CLASSES = ["person", "car", "truck", "bicycle", "motorcycle", "bus"]

PROMPT_V2 = ("You verify detections from a fixed security camera. Scene: {scene}. The detector claims a "
             "'{cls}'. Image 1 is the tight crop, image 2 the wider context (the detection is the yellow box). "
             "Is it a real {cls} (or another real person/vehicle), or a false positive? "
             "Reply with JSON only: {{\"verdict\": \"true_positive|false_positive\", \"corrected_class\": \"...\", "
             "\"confidence\": 0-1}}")


def results(root: Path):
    for p in sorted((root / "results").glob("*.json")):
        if p.stem in ("index", "bench", "bench_v2", "human_labels"):
            continue
        yield root, json.loads(p.read_text())


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def synthetic_negatives(root: Path, res: dict, n: int, rng: random.Random, img_dir: Path) -> list[dict]:
    """Random person/vehicle-shaped boxes on background, far from any track or candidate."""
    cid = res["clip"]["id"]
    cap = cv2.VideoCapture(str(WORKS[root] / f"{cid}.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps, w, h = res["fps"], res["width"], res["height"]
    out, tries = [], 0
    while len(out) < n and tries < n * 30:
        tries += 1
        idx = rng.randrange(0, max(1, total))
        t = idx / fps
        cls = rng.choice(["person", "person", "car"])
        bh = rng.uniform(0.12, 0.45)
        bw = bh * (h / w) * (rng.uniform(0.3, 0.55) if cls == "person" else rng.uniform(1.2, 2.0))
        x1, y1 = rng.uniform(0, 1 - bw), rng.uniform(0, 1 - bh)
        box = [x1, y1, x1 + bw, y1 + bh]
        near = [b[1:5] for tr in res["tracks"] for b in tr["boxes"] if abs(b[0] - t) < 1.0]
        if any(iou(box, nb) > 0.02 for nb in near):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        px = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        k = len(out)
        tight, ctx = img_dir / f"{cid}_bg{k}_t.jpg", img_dir / f"{cid}_bg{k}_c.jpg"
        cv2.imwrite(str(tight), _crop(frame, px, 0.15, 160, 320), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(ctx), _crop(frame, px, 2.0, 256, 512, draw_box=True), [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append({"tight": tight, "context": ctx, "cls": cls})
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-groups", default="street,walkway")
    ap.add_argument("--bg-per-clip", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    test_groups = set(args.test_groups.split(","))
    rng = random.Random(args.seed)
    human = {}
    hp = SITE / "results" / "human_labels.json"
    if hp.exists():
        human = {k: v for k, v in json.loads(hp.read_text()).items() if v.get("verdict") in ("true_positive", "false_positive")}
    if OUT.exists():
        shutil.rmtree(OUT)
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True)
    rows: list[dict] = []

    def add(root, clip, tight_src, ctx_src, cls, verdict, source, difficulty, uid):
        group = SITE_GROUPS.get(clip["id"]) or clip.get("group") or clip["id"]
        names = []
        for kind, src in (("t", tight_src), ("c", ctx_src)):
            src = Path(src)
            dst = img_dir / (src.name if src.parent == img_dir else f"{clip['id']}_{uid}_{kind}.jpg")
            if src != dst:
                shutil.copyfile(src, dst)
            names.append("images/" + dst.name)
        rows.append({"id": f"{clip['id']}:{uid}", "group": group, "clip": clip["id"], "images": names,
                     "claimed": cls, "label": verdict, "label_source": source, "difficulty": difficulty,
                     "prompt": PROMPT_V2.format(scene=clip["scene"], cls=cls),
                     "target": json.dumps({"verdict": verdict,
                                           "corrected_class": cls if verdict == "true_positive" else "none",
                                           "confidence": 0.9})})

    for root, res in list(results(SITE)) + list(results(V2)):
        clip = res["clip"]
        neg_scene = bool(clip.get("negative_only"))
        for t in res["tracks"]:
            key = f"{clip['id']}:{t['id']}"
            best, ctx = root / t["crops"]["best"], root / t["crops"]["context"]
            if not best.exists():
                continue
            if neg_scene:  # every detection in a scene without people/vehicles is a false alarm
                add(root, clip, best, ctx, t["cls"], "false_positive", "no-target-scene", "hard", t["id"])
            elif key in human:
                add(root, clip, best, ctx, t["cls"], human[key]["verdict"], "human", "hard", t["id"])
            elif t.get("triage") and t["triage"]["status"] in ("confirmed", "rejected"):
                v = t["triage"]["final"]["verdict"]
                cls = t["triage"]["final"].get("corrected_class") or t["cls"]
                add(root, clip, best, ctx, cls if v == "true_positive" else t["cls"], v, "teacher", "hard", t["id"])
            elif t["band"] == "auto_accept":
                for k, view in enumerate([t["crops"]["best"]] + t["crops"]["frames"]):
                    if (root / view).exists():
                        add(root, clip, root / view, ctx, t["cls"], "true_positive", "detector-confident", "easy", f"{t['id']}v{k}")
        for k, neg in enumerate(synthetic_negatives(root, res, args.bg_per_clip, rng, img_dir)):
            add(root, clip, neg["tight"], neg["context"], neg["cls"], "false_positive", "synthetic-bg", "easy", f"bg{k}")

    train = [r for r in rows if r["group"] not in test_groups]
    test = [r for r in rows if r["group"] in test_groups]
    assert not {r["group"] for r in train} & test_groups, "leak"
    (OUT / "all.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train))
    (OUT / "test.jsonl").write_text("\n".join(json.dumps(r) for r in test))
    for name, rr in (("all", rows), ("train", train), ("test", test)):
        print(name, len(rr), dict(Counter((r["difficulty"], r["label"]) for r in rr)))
    print("sources", dict(Counter(r["label_source"] for r in rows)))
    print("groups", dict(Counter(r["group"] for r in rows)))
    print("test hard", len([r for r in test if r["difficulty"] == "hard"]))


if __name__ == "__main__":
    main()
