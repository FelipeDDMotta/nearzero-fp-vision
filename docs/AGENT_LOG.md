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

---

### 3. QLoRA on a free T4: `attn_bias is not correctly aligned (strideM)`

**What happened.** The first Kaggle run crashed in the Qwen2.5-VL fine-tune with
`RuntimeError: attn_bias is not correctly aligned (strideM). attn_bias.stride(2) = 494, and should be a multiple of 4`.
T4 GPUs (compute capability 7.5) don't support FlashAttention 2, so the stack falls back to
xformers' memory-efficient attention. xformers needs the padded sequence length to be a multiple
of 4, and a batch of two image+text samples padded to 494 tokens broke that rule.

**Fix.** Uninstall xformers so attention falls back to PyTorch SDPA, and train with
`per_device_train_batch_size=1, gradient_accumulation_steps=8` (same effective batch, no padding).

**Takeaway.** "Runs on my GPU" is hardware-specific. The cheapest accelerators often hit
kernel-level constraints, so pin the attention backend explicitly when targeting them, which
matters for edge deployments too.

---

### 4. The distilled student learned the format, not the judgment

**What happened.** QLoRA on Qwen2.5-VL-3B (168 samples, 2 epochs, 42 steps, about 6 minutes on a free
T4) cut the training loss from 5.3 to 0.64. On the two held-out clips it matched the zero-shot base
model **exactly in accuracy** (16/23 decidable items each, teacher labels as ground truth). What
changed was the **output contract**: the base wrapped its JSON in Markdown fences and reported low
confidence (mean 0.27), while the student emits bare JSON with teacher-like confidence (mean 0.63).
Three individual predictions flipped, with a net change of zero.

**Why.** 168 samples is enough to learn a format, not a visual skill. The labels come from a teacher
that is itself imperfect on degraded footage, and with n=23 one item moves accuracy by about 4
points. A loss curve going down was never evidence that the judgment improved.

**What I'd do next.** More and harder data (synthetic hard negatives from the degradation
generator, more cameras), adapt the vision layers once the data supports it, use human review
labels as ground truth, and report confidence intervals. I kept the result on the site as it is.

**Takeaway.** Separate "the model learned the interface" from "the model learned the task", and
measure the second one on data it has never seen.
