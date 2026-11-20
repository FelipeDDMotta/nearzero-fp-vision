"""Confidence bands: the routing policy that decides what the agent ever sees.

A classic analytics stack has one threshold: above it you alert, below it you drop.
NearZeroFP Vision splits the space in three, so the expensive reasoning only runs where the
detector is genuinely unsure:

  auto_accept   score >= ACCEPT          -> alert without an LLM call
  uncertain     DROP <= score < ACCEPT   -> verifier agent (or flicker: very short tracks)
  drop          score < DROP             -> ignored, as a normal threshold would
"""
from __future__ import annotations

ACCEPT = 0.60
DROP = 0.15
MIN_STABLE_FRAMES = 4  # tracks shorter than this are flicker candidates even if confident


def track_score(confs: list[float]) -> float:
    """Mean of the top-3 confidences: robust to a few weak frames in a good track."""
    top = sorted(confs, reverse=True)[:3]
    return sum(top) / len(top)


def assign_band(score: float, n_frames: int) -> str:
    if score < DROP:
        return "drop"
    if score >= ACCEPT and n_frames >= MIN_STABLE_FRAMES:
        return "auto_accept"
    return "uncertain"
