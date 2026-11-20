# Build log: what broke and what I changed

A running log of real problems hit while building NearZeroFP Vision, written as they happened.
The project was built in one session with Claude Code as a pair-programming agent, so some
entries are about the product's agents and some about working with a coding agent.

---

### 1. The synthetic "night + rain" clip killed the detector completely

**What happened.** The first version of the procedural degradation (gamma 1.9, dense rain
streaks, JPEG q=22) looked realistic to a human, and the car in frame was obvious. But YOLO11n
returned **zero detections at conf ≥ 0.05** across the whole clip, so the clip produced zero tracks.

**Why it matters.** This is the real failure mode the industry talks about. Off-the-shelf
detectors trained on clean daylight data don't degrade gracefully; they fall off a cliff. For the
triage loop, though, a clip with no detections gives the agent nothing to verify.

**Fix.** I measured detector confidence on fixed frames across a sweep of degradation strengths
(clean → mild → medium → original) and picked the setting that moves detections *into the
uncertain band* (0.2–0.6) instead of erasing them. The degradation is seeded and parameterised, so
it is a small, sweepable synthetic-data generator rather than a one-off filter.

| setting | top confidences on 3 probe frames |
|---|---|
| clean | 0.90 · 0.08 · 0.70/0.64/0.55 |
| original night+rain | 0.09 · – · 0.06 |
| medium (chosen) | 0.73 · 0.06 · 0.79/0.52/0.42 |

**Takeaway.** Synthetic data needs a target metric. "Looks right" is not a spec. Calibrate the
generator against the model you are trying to stress.

---

### 2. A tracker-only pipeline gave the agent almost nothing to do

**What happened.** After detection and tracking on 10 clips (5 clean, 6 degraded), only **29 of 88
tracks** fell in the uncertain band. Clean sample clips are easy, and ByteTrack only creates tracks
above its own `new_track_thresh`. So the real false-alarm material, the low-confidence detections
a site would see if it lowered its threshold, never reached the agent. That was too little to show
the loop working, and far too little to fine-tune a student model on.

**Fix.** A mining stage (`pipeline/mine_candidates.py`) samples frames every 0.5 s, runs the detector
at conf ≥ 0.04, and keeps detections that do **not** overlap an existing track: "the alerts you'd get
by lowering the threshold". They join the uncertain band as single-frame candidates, and the agent
prompt is told what they are. A first pass took 30 near-identical crops of one machine part
(a static false-alarm source fires every frame), so candidates are now limited to two per image
location. Result: ~120 diverse candidates, a mix of clear false positives (machinery, shelving,
poles read as "person") and real objects the detector nearly missed (a car at 0.05 in the rain).

**Takeaway.** The question isn't "is the detector right?" but "what would the operator see at the
threshold they actually want to run?". Mining below threshold turns recall vs. precision into a
job for the agent.
