"""Edge baseline: MobileNetV3 features + logistic regression, leave-one-source-video-out CV.

A 3B VLM does not fit on an edge NPU; a ~5M-parameter CNN does. This trains the cheapest
plausible verifier on the same labels and measures it honestly:
  * folds = source videos (all synthetic variants of a video stay together, so nothing leaks)
  * the headline metric is accuracy on the HARD items of the held-out video
  * features: tight crop + context crop (frozen ImageNet MobileNetV3-Large) + claimed class

Usage:  python distill/edge_classifier.py [--fixed-test street,walkway]
Writes distill/out/edge_cv.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "distill" / "data_v2"
OUT = ROOT / "distill" / "out"
CLASSES = ["person", "car", "truck", "bicycle", "motorcycle", "bus"]


def features(rows: list[dict]) -> tuple[np.ndarray, float]:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights).eval()
    model.classifier = torch.nn.Identity()
    tf = weights.transforms()
    paths = sorted({p for r in rows for p in r["images"]})
    feats: dict[str, np.ndarray] = {}
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(paths), 64):
            batch = paths[i:i + 64]
            x = torch.stack([tf(Image.open(DATA / p).convert("RGB")) for p in batch])
            for p, f in zip(batch, model(x).numpy()):
                feats[p] = f
    ms_per_image = 1000 * (time.time() - t0) / max(1, len(paths))
    X = []
    for r in rows:
        onehot = np.zeros(len(CLASSES), np.float32)
        if r["claimed"] in CLASSES:
            onehot[CLASSES.index(r["claimed"])] = 1
        X.append(np.concatenate([feats[r["images"][0]], feats[r["images"][1]], onehot * 3]))
    return np.stack(X), ms_per_image


def score(rows: list[dict], pred: np.ndarray) -> dict:
    hard = [(r, p) for r, p in zip(rows, pred) if r["difficulty"] == "hard"]
    easy = [(r, p) for r, p in zip(rows, pred) if r["difficulty"] == "easy"]

    def acc(pairs):
        return round(sum((p == 1) == (r["label"] == "true_positive") for r, p in pairs) / max(1, len(pairs)), 3)

    fps = [(r, p) for r, p in hard if r["label"] == "false_positive"]
    tps = [(r, p) for r, p in hard if r["label"] == "true_positive"]
    return {"hard_n": len(hard), "hard_acc": acc(hard), "easy_n": len(easy), "easy_acc": acc(easy),
            "hard_fp_rejected": f"{sum(p == 0 for _, p in fps)}/{len(fps)}",
            "hard_real_kept": f"{sum(p == 1 for _, p in tps)}/{len(tps)}"}


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-test", default="street,walkway")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (DATA / "all.jsonl").read_text().splitlines() if l.strip()]
    X, ms = features(rows)
    y = np.array([r["label"] == "true_positive" for r in rows], int)
    groups = sorted({r["group"] for r in rows})

    # Leave-one-source-video-out CV over every group that has hard items.
    per_fold, all_pred = {}, np.full(len(rows), -1)
    for g in groups:
        te = np.array([r["group"] == g for r in rows])
        if not any(r["difficulty"] == "hard" for r, m in zip(rows, te) if m) or len(set(y[~te])) < 2:
            continue
        all_pred[te] = fit_predict(X[~te], y[~te], X[te])
        per_fold[g] = score([r for r, m in zip(rows, te) if m], all_pred[te])
    covered = all_pred >= 0
    cv_total = score([r for r, m in zip(rows, covered) if m], all_pred[covered])

    # The same fixed split the VLM uses, for a like-for-like comparison.
    fixed = set(args.fixed_test.split(","))
    te = np.array([r["group"] in fixed for r in rows])
    fixed_pred = fit_predict(X[~te], y[~te], X[te])
    fixed_score = score([r for r, m in zip(rows, te) if m], fixed_pred)
    # The original agent-labelled hard items only (comparable to the VLM benchmark)
    agent_rows = [(r, p) for r, p in zip([r for r, m in zip(rows, te) if m], fixed_pred) if r["label_source"] in ("teacher", "human")]
    fixed_agent = score([r for r, _ in agent_rows], np.array([p for _, p in agent_rows]))

    out = {"model": "MobileNetV3-Large (ImageNet, frozen) + logistic regression", "params_m": 5.4,
           "cpu_ms_per_image": round(ms, 1), "n_samples": len(rows),
           "cv_leave_one_video_out": {"total": cv_total, "per_fold": per_fold},
           "fixed_split": {"test_groups": sorted(fixed), "all": fixed_score, "agent_labelled_only": fixed_agent}}
    OUT.mkdir(exist_ok=True)
    (OUT / "edge_cv.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
