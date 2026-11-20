"""Stage 1: detect + track every clip, cut evidence crops, assign confidence bands.

Outputs (served statically by the web app and read by the API):
  apps/web/public/clips/<id>.mp4            H.264 clip that the console plays
  apps/web/public/crops/<id>/<tid>_*.jpg    evidence images per track
  apps/web/public/results/<id>.json         tracks, per-frame boxes, bands

Usage:  python pipeline/detect_track.py [--only street,store] [--max-seconds 60]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bands import assign_band, track_score  # noqa: E402
from degrade import DEGRADERS  # noqa: E402
from write_index import write_index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"
PUBLIC = ROOT / "apps" / "web" / "public"
COCO_KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MAX_W = 960
TRACK_FPS = 10


def transcode_h264(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-an",
           "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def render_clip(meta: dict, max_seconds: float) -> tuple[Path, float, int, int, int]:
    """Read the source, resize, optionally degrade, write an intermediate mp4."""
    cap = cv2.VideoCapture(str(RAW / meta["source"]))
    if not cap.isOpened():
        raise FileNotFoundError(RAW / meta["source"])
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / 15))  # 60 fps sources -> ~15 fps: smaller files, faster pipeline
    fps = src_fps / step
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_W / w0)
    w, h = int(round(w0 * scale / 2) * 2), int(round(h0 * scale / 2) * 2)
    degrade = DEGRADERS.get(meta.get("degrade", ""))
    WORK.mkdir(parents=True, exist_ok=True)
    out_path = WORK / f"{meta['id']}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    raw_idx = -1
    max_frames = int(max_seconds * fps)
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        raw_idx += 1
        if raw_idx % step:
            continue
        if scale < 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        if degrade:
            frame = degrade(frame, n)
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    return out_path, fps, w, h, n


def run_tracking(model: YOLO, video: Path, fps: float, tracker_cfg: Path) -> tuple[dict, int]:
    stride = max(1, round(fps / TRACK_FPS))
    cap = cv2.VideoCapture(str(video))
    tracks: dict[int, list] = defaultdict(list)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            res = model.track(frame, persist=True, tracker=str(tracker_cfg), conf=0.05, iou=0.5,
                              classes=list(COCO_KEEP), imgsz=640, verbose=False)[0]
            if res.boxes is not None and res.boxes.id is not None:
                for tid, c, conf, xyxy in zip(res.boxes.id.int().tolist(), res.boxes.cls.int().tolist(),
                                              res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
                    tracks[tid].append((idx, c, float(conf), [float(v) for v in xyxy]))
        idx += 1
    cap.release()
    return tracks, stride


def _crop(frame: np.ndarray, box: list[float], pad: float, min_side: int, max_side: int,
          draw_box: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = bw * (1 + pad) / 2, bh * (1 + pad) / 2
    X1, Y1 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    X2, Y2 = int(min(w, cx + half_w)), int(min(h, cy + half_h))
    img = frame.copy() if draw_box else frame
    if draw_box:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    crop = img[Y1:max(Y2, Y1 + 2), X1:max(X2, X1 + 2)]
    side = max(crop.shape[:2])
    target = min(max(side, min_side), max_side)
    s = target / side
    return cv2.resize(crop, (max(2, int(crop.shape[1] * s)), max(2, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)


def cut_crops(video: Path, clip_id: str, summaries: list[dict], raw_tracks: dict) -> None:
    """Second pass: save best crop, context crop and 3 track frames per track."""
    out_dir = PUBLIC / "crops" / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted: dict[int, list[tuple[int, str, list[float], str]]] = defaultdict(list)
    for s in summaries:
        dets = raw_tracks[s["id"]]
        best = max(dets, key=lambda d: d[2])
        wanted[best[0]].append((s["id"], "best", best[3], "tight"))
        wanted[best[0]].append((s["id"], "context", best[3], "context"))
        picks = [dets[0], dets[len(dets) // 2], dets[-1]] if len(dets) >= 3 else dets
        for k, d in enumerate(picks):
            wanted[d[0]].append((s["id"], f"f{k}", d[3], "tight"))
        s["crops"] = {
            "best": f"crops/{clip_id}/{s['id']}_best.jpg",
            "context": f"crops/{clip_id}/{s['id']}_context.jpg",
            "frames": [f"crops/{clip_id}/{s['id']}_f{k}.jpg" for k in range(len(picks))],
        }
    cap = cv2.VideoCapture(str(video))
    idx = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, name, box, kind in wanted.pop(idx, []):
            if kind == "context":
                img = _crop(frame, box, pad=2.0, min_side=256, max_side=512, draw_box=True)
            else:
                img = _crop(frame, box, pad=0.15, min_side=160, max_side=320)
            cv2.imwrite(str(out_dir / f"{tid}_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        idx += 1
    cap.release()


def summarise(raw_tracks: dict, fps: float, w: int, h: int) -> list[dict]:
    out = []
    for tid, dets in sorted(raw_tracks.items()):
        confs = [d[2] for d in dets]
        cls_name = COCO_KEEP[Counter(d[1] for d in dets).most_common(1)[0][0]]
        score = track_score(confs)
        out.append({
            "id": tid,
            "cls": cls_name,
            "score": round(score, 3),
            "conf_max": round(max(confs), 3),
            "conf_median": round(statistics.median(confs), 3),
            "n": len(dets),
            "t0": round(dets[0][0] / fps, 3),
            "t1": round(dets[-1][0] / fps, 3),
            "band": assign_band(score, len(dets)),
            # [t, x1, y1, x2, y2, conf] normalised to [0, 1] for the canvas overlay
            "boxes": [[round(d[0] / fps, 3), round(d[3][0] / w, 4), round(d[3][1] / h, 4),
                       round(d[3][2] / w, 4), round(d[3][3] / h, 4), round(d[2], 3)] for d in dets],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated clip ids")
    ap.add_argument("--max-seconds", type=float, default=90.0)
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()

    clips = json.loads((ROOT / "pipeline" / "clips.json").read_text())
    only = {c for c in args.only.split(",") if c}
    tracker_cfg = ROOT / "pipeline" / "bytetrack_low.yaml"
    (PUBLIC / "results").mkdir(parents=True, exist_ok=True)

    for meta in clips:
        if only and meta["id"] not in only:
            continue
        t_start = time.time()
        model = YOLO(args.model)  # fresh tracker state per clip
        video, fps, w, h, n_frames = render_clip(meta, args.max_seconds)
        raw_tracks, stride = run_tracking(model, video, fps, tracker_cfg)
        summaries = summarise(raw_tracks, fps, w, h)
        cut_crops(video, meta["id"], summaries, raw_tracks)
        transcode_h264(video, PUBLIC / "clips" / f"{meta['id']}.mp4")
        result = {
            "clip": {**meta, "video": f"clips/{meta['id']}.mp4"},
            "fps": fps, "width": w, "height": h,
            "duration": round(n_frames / fps, 3), "track_stride": stride,
            "detector": {"model": args.model, "tracker": "bytetrack (low-thresh)", "imgsz": 640},
            "tracks": summaries,
        }
        (PUBLIC / "results" / f"{meta['id']}.json").write_text(json.dumps(result))
        bands = Counter(s["band"] for s in summaries)
        print(f"[{meta['id']}] {n_frames} frames @ {fps:.1f} fps, {len(summaries)} tracks "
              f"{dict(bands)} in {time.time() - t_start:.0f}s")
        write_index()


if __name__ == "__main__":
    main()
