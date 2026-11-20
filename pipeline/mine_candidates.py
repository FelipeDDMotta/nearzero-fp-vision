"""Stage 1b: mine low-confidence candidates, the alerts you'd get by lowering the threshold.

Operators face a trade-off: raise the threshold and miss real events, lower it and drown in false
alarms. This stage samples frames, runs the detector with a very low threshold, and keeps
detections that the tracker did NOT turn into tracks (no overlap with any existing track box).
They are added to the clip as single-frame "candidate" tracks in the uncertain band, so the
verifier agent decides them: most are noise (hard negatives), and some are real objects the
detector nearly missed (recall the agent wins back).

Usage:  python pipeline/mine_candidates.py [--only street] [--per-clip 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_track import COCO_KEEP, PUBLIC, ROOT, WORK, _crop  # noqa: E402
from write_index import write_index  # noqa: E402

CAND_ID_BASE = 100000
MIN_CONF, MAX_CONF = 0.04, 0.30
SAMPLE_EVERY_S = 0.5


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def track_boxes_near(tracks: list[dict], t: float, win: float = 0.3) -> list[list[float]]:
    out = []
    for tr in tracks:
        if tr["t0"] - win <= t <= tr["t1"] + win:
            out += [b[1:5] for b in tr["boxes"] if abs(b[0] - t) <= win]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--per-clip", type=int, default=30)
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()
    only = {c for c in args.only.split(",") if c}
    model = YOLO(args.model)

    for path in sorted((PUBLIC / "results").glob("*.json")):
        if path.stem in ("index", "bench", "human_labels") or (only and path.stem not in only):
            continue
        res = json.loads(path.read_text())
        res["tracks"] = [t for t in res["tracks"] if t.get("source") != "low_conf_mining"]
        video = WORK / f"{path.stem}.mp4"
        cap = cv2.VideoCapture(str(video))
        fps, w, h = res["fps"], res["width"], res["height"]
        step = max(1, round(fps * SAMPLE_EVERY_S))
        found = []
        idx = 0
        frames: dict[int, any] = {}
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = idx / fps
                r = model(frame, conf=MIN_CONF, iou=0.5, classes=list(COCO_KEEP), imgsz=640, verbose=False)[0]
                existing = track_boxes_near(res["tracks"], t)
                for c, conf, xyxy in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    if not (MIN_CONF <= conf < MAX_CONF):
                        continue
                    nb = [xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h]
                    if (nb[2] - nb[0]) * (nb[3] - nb[1]) < 0.0004:  # tiny specks: not even a candidate
                        continue
                    if any(iou(nb, e) > 0.2 for e in existing):
                        continue
                    found.append({"idx": idx, "t": t, "cls": COCO_KEEP[c], "conf": float(conf), "xyxy": xyxy, "nb": nb})
                    frames[idx] = frame
            idx += 1
        cap.release()
        # Keep the strongest candidates, at most one per class per second, spread over the clip.
        found.sort(key=lambda d: -d["conf"])
        kept, seen = [], set()
        for d in found:
            key = (d["cls"], int(d["t"]))
            if key in seen or any(k["idx"] == d["idx"] and iou(k["nb"], d["nb"]) > 0.1 for k in kept):
                continue
            # A static false-alarm source (a machine part, a pole) fires over and over; keep at most
            # two candidates per image location so the set stays diverse.
            if sum(iou(k["nb"], d["nb"]) > 0.5 for k in kept) >= 2:
                continue
            seen.add(key)
            kept.append(d)
            if len(kept) >= args.per_clip:
                break
        out_dir = PUBLIC / "crops" / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, d in enumerate(sorted(kept, key=lambda d: d["t"])):
            tid = CAND_ID_BASE + k
            frame = frames[d["idx"]]
            cv2.imwrite(str(out_dir / f"{tid}_best.jpg"), _crop(frame, d["xyxy"], 0.15, 160, 320), [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(str(out_dir / f"{tid}_context.jpg"), _crop(frame, d["xyxy"], 2.0, 256, 512, draw_box=True),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            res["tracks"].append({
                "id": tid, "cls": d["cls"], "score": round(d["conf"], 3), "conf_max": round(d["conf"], 3),
                "conf_median": round(d["conf"], 3), "n": 1, "t0": round(d["t"], 3), "t1": round(d["t"], 3),
                "band": "uncertain", "source": "low_conf_mining",
                "boxes": [[round(d["t"], 3), *[round(v, 4) for v in d["nb"]], round(d["conf"], 3)]],
                "crops": {"best": f"crops/{path.stem}/{tid}_best.jpg", "context": f"crops/{path.stem}/{tid}_context.jpg",
                          "frames": [f"crops/{path.stem}/{tid}_best.jpg"]},
            })
        path.write_text(json.dumps(res))
        print(f"[{path.stem}] {len(found)} raw low-conf detections -> {len(kept)} candidates")
    write_index()


if __name__ == "__main__":
    main()
