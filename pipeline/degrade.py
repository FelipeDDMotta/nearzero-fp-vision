"""Synthetic degradation: turn a clean clip into a hard one.

Off-the-shelf detectors are trained mostly on clean, well-lit images. In the field they
see night, rain, glare, dirty lenses and heavy compression, which is where false positives
come from. This script applies those conditions procedurally so we can stress the detector
and give the verifier agent realistic hard cases. It is a small stand-in for a real
synthetic-data pipeline: deterministic (seeded) and parameterised, so it can be swept.
"""
from __future__ import annotations

import numpy as np
import cv2


def _rain_layer(h: int, w: int, rng: np.random.Generator, density: float = 0.0025,
                length: int = 18, angle_deg: float = 12.0) -> np.ndarray:
    """Return a float32 HxW mask of motion-blurred rain streaks in [0, 1]."""
    layer = np.zeros((h, w), np.float32)
    n = int(h * w * density)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    layer[ys, xs] = rng.uniform(0.5, 1.0, n).astype(np.float32)
    # Directional motion blur kernel turns dots into streaks.
    k = np.zeros((length, length), np.float32)
    k[:, length // 2] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(k.sum(), 1e-6)
    layer = cv2.filter2D(layer, -1, k) * 6.0
    return np.clip(layer, 0, 1)


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def night_rain(frame: np.ndarray, idx: int, seed: int = 7) -> np.ndarray:
    # Calibrated against YOLO11n: the first version (gamma 1.9, dense rain, q=22) left the
    # detector with *zero* detections even on an obvious car, which is realistic but useless
    # for the demo. These settings push it into the uncertain band instead.
    rng = np.random.default_rng(seed + idx)
    h, w = frame.shape[:2]
    img = frame.astype(np.float32) / 255.0
    # Low light: gamma + exposure drop, and a cold LED colour cast.
    img = np.power(img, 1.7) * 0.6
    img[..., 0] *= 1.12  # B
    img[..., 2] *= 0.88  # R
    # Street-light pools: a couple of soft bright blobs.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for cx, cy in ((0.18, 0.12), (0.82, 0.2)):
        d = ((xx - cx * w) ** 2 + (yy - cy * h) ** 2) / (0.18 * w) ** 2
        img += (np.exp(-d) * 0.12)[..., None] * np.array([0.6, 0.85, 1.0], np.float32)
    # Rain streaks (fresh each frame) plus slight lens-water blur.
    rain = _rain_layer(h, w, rng, density=0.0018)
    img = img * (1 - 0.3 * rain[..., None]) + 0.35 * rain[..., None]
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    # Sensor noise grows in low light.
    img += rng.normal(0, 0.04, img.shape).astype(np.float32)
    out = np.clip(img * 255, 0, 255).astype(np.uint8)
    return _jpeg(out, 28)


def dusk_glare(frame: np.ndarray, idx: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed + idx)
    h, w = frame.shape[:2]
    img = frame.astype(np.float32) / 255.0
    # Warm, low-contrast dusk.
    img = 0.25 + (img - 0.5) * 0.55
    img[..., 2] *= 1.2
    img[..., 0] *= 0.8
    # A large lens-flare / glare veil from the top right, gently drifting.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w * (0.85 + 0.03 * np.sin(idx / 40))
    cy = h * 0.1
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / w
    veil = np.exp(-(d / 0.35) ** 2) * 0.65
    img = img * (1 - veil[..., None] * 0.5) + veil[..., None] * np.array([0.8, 0.95, 1.0], np.float32)
    # Flare ghosts along the optical axis.
    for t, r in ((0.45, 0.05), (0.62, 0.035), (0.8, 0.06)):
        gx, gy = cx + (w / 2 - cx) * 2 * t, cy + (h / 2 - cy) * 2 * t
        g = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (r * w) ** 2))
        img += g[..., None] * 0.18
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    out = np.clip(img * 255, 0, 255).astype(np.uint8)
    return _jpeg(out, 30)


DEGRADERS = {"night_rain": night_rain, "dusk_glare": dusk_glare}


def lowlight(frame: np.ndarray, idx: int, seed: int = 23) -> np.ndarray:
    """Indoor after-hours: underexposed, heavy sensor + chroma noise, motion smear, low bitrate."""
    rng = np.random.default_rng(seed + idx)
    img = frame.astype(np.float32) / 255.0
    img = np.power(img, 1.6) * 0.55
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * 0.5  # colour washes out in low light
    k = np.zeros((1, 7), np.float32)
    k[0, :] = 1 / 7
    img = cv2.filter2D(img, -1, k)  # horizontal motion smear
    img += rng.normal(0, 0.035, img.shape).astype(np.float32)
    img += rng.normal(0, 0.02, img.shape[:2] + (1,)).astype(np.float32)
    out = np.clip(img * 255, 0, 255).astype(np.uint8)
    return _jpeg(out, 26)


DEGRADERS["lowlight"] = lowlight
