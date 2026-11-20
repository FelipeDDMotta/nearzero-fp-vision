# NearZeroFP Vision: agentic false-positive triage for video analytics

**The detector flags what it's unsure about. An agent decides. The decisions train the next model.**

NearZeroFP Vision is a working, deployed demo of a closed loop for edge video analytics:

1. A detector + tracker (YOLO11n + ByteTrack) runs on security-camera clips, including
   **synthetically degraded** night, rain, glare and low-light variants.
2. A **confidence-band router** sends only the *uncertain* tracks to an agent. Confident tracks alert
   directly, and noise below the floor is dropped.
3. A **Claude verifier agent** decides each uncertain track in a bounded tool loop. It starts from a
   tight crop and asks for a context view or other frames only when it needs them. A cheap model
   runs first and the strong model only when unsure. A skeptical **auditor** re-checks a sample, and
   any disagreement **escalates to a human**.
4. Each decision also yields searchable **appearance attributes** (clothing colour, accessories,
   vehicle type), so you can search footage in plain English.
5. Agent decisions plus human labels become **training data**. A small open VLM
   (**Qwen2.5-VL-3B + QLoRA**) is distilled from the agent, benchmarked with **vLLM** on a camera it
   never saw, and served live from a scale-to-zero GPU endpoint.

> Live demo: **https://nearzero-fp-vision.vercel.app** · API: Render · Student model: Hugging Face + Modal
> Built as an AI-engineering portfolio project in one session, with Claude Code as the coding agent.

---

## Why this exists

Video analytics rarely fails for lack of detections. It fails on **false alarms**: rain streaks,
glare, shadows, posters, machinery. Operators learn to ignore those alerts. The usual fix is a
single global threshold, which trades missed events against noise.

NearZeroFP Vision treats the threshold as a **routing decision**. Only the band where the detector is
genuinely unsure gets expensive reasoning. That reasoning is checked (cascade + audit +
escalation), and it is **recycled into training data**, so the system gets cheaper and better
the longer it runs.

## What to click

| Page | What it shows |
|---|---|
| `/console` | Operator console: video with tracked boxes coloured by decision, a timeline of events, and for each event the crops, the agent's tool calls, its verdict and reasoning, cost and latency. **Re-run agent live** calls the deployed API, with either Claude or the distilled open VLM. |
| `/review` | Human-in-the-loop queue for escalated cases (blind mode, keyboard shortcuts). Labels export as ground truth. |
| `/search` | Natural-language appearance search ("person in a dark top with a backpack"), with the parsed filter and why each result matched. |
| `/loop` | The distillation story and the teacher-vs-student benchmark table. |

## Architecture

```
pipeline/   offline, CPU (Python)
  degrade.py        seeded synthetic degradations: night_rain · dusk_glare · lowlight
  detect_track.py   YOLO11n + ByteTrack (low thresholds) -> tracks, evidence crops, H.264 clips
  bands.py          routing policy: auto_accept / uncertain / drop
  run_triage.py     runs the verifier agent on uncertain tracks (cached, parallel)
apps/api/   FastAPI on Render
  app/agent.py      verifier agent: bounded tool loop, cascade, audit, escalation
  app/search.py     LLM query -> structured filter; deterministic matching
  app/open_vlm.py   client for the distilled student served by vLLM
  app/dataset.py    decisions -> classifier crops + manifest (hard negatives included)
apps/web/   Next.js 15 + Tailwind on Vercel (static results + live API calls)
distill/    export_vlm_dataset.py · finetune_qlora.ipynb (Kaggle T4) · bench_vllm.py · bench_report.py
serving/    modal_app.py: vLLM OpenAI-compatible server, scale-to-zero, bearer auth
docs/       AGENT_LOG.md: what broke and how it was fixed
```

### The verifier agent (harness design)

- **Evidence on request.** The first message holds only the tight crop and the detector's numbers.
  Tools: `get_context_frame` (wider view, box drawn), `get_track_frames` (first/middle/last frame with
  confidences), `submit_verdict` (strict JSON schema). Each evidence request carries a `why`, which
  is logged and shown in the console.
- **Bounded.** At most 4 turns. On the last turn only `submit_verdict` is offered.
- **Cascade.** `claude-haiku-5-5` first. `claude-opus-5-5` re-runs the case only when the verdict is
  `uncertain` or confidence is below 0.70.
- **Audit.** A deterministic ~15% sample gets an independent single-shot opinion with a different
  information path (all evidence up front, skeptical prompt). Disagreement → escalate.
- **Strict schemas.** All tools use `strict: true`, so verdicts always parse. Attributes use closed
  vocabularies, which keeps search deterministic.
- **Accounting.** Tokens, latency and $ cost are recorded per step. Results are cached by evidence
  digest + prompt version + models.

### Distillation (teacher → student)

- `export_vlm_dataset.py` turns auto-resolved decisions (and human labels, which take precedence)
  into chat samples: (tight crop + context view, prompt) → verdict JSON. **One camera is held
  out**, and the export script asserts that it never appears in training.
- `finetune_qlora.ipynb`: Unsloth QLoRA on `Qwen2.5-VL-3B-Instruct` (4-bit, r=16, language layers
  only) on a free Kaggle T4. The merged fp16 weights go to the Hugging Face Hub.
- `bench_vllm.py`: vLLM serves the base and the fine-tuned model, and both are scored on the
  held-out camera. `bench_report.py` adds the Claude teacher and writes the table on `/loop`.
- `serving/modal_app.py`: the student behind an OpenAI-compatible vLLM server on Modal. The API
  proxies to it when the console's *Open VLM* toggle is on.

## Run it locally (Windows or Linux)

```bash
# Python (pipeline + API)
python -m venv .venv && .venv/Scripts/activate   # or: source .venv/bin/activate
pip install -r pipeline/requirements.txt -r apps/api/requirements.txt
cp .env.example .env                              # add ANTHROPIC_API_KEY

# sample clips (CC BY 4.0) -> data/raw/
#   https://github.com/intel-iot-devkit/sample-videos
python pipeline/detect_track.py                  # detect, track, crops, bands
python pipeline/run_triage.py --describe         # verifier agent + attributes
uvicorn app.main:app --app-dir apps/api --reload # http://localhost:8000/health

# Web
cd apps/web && npm install && npm run dev        # http://localhost:3000
```

Tests: `cd apps/api && pytest` (no network or key needed; the agent is stubbed).

## Honest notes

- The clips are short public sample videos. The night, rain, glare and low-light variants are
  **procedural degradations**, not real night footage. They are calibrated against the detector (see
  `docs/AGENT_LOG.md`).
- Metrics on the site come from the precomputed demo run. Student metrics use a small held-out set
  (one camera), so treat them as a signal, not a benchmark. Ground truth is human review labels
  where they exist, and teacher labels otherwise; the table says which.
- Results shown in the console are precomputed for speed. "Re-run agent live" exercises the real
  path, and that path is rate-limited to protect the API key.

## Licences

Code: AGPL-3.0 (compatible with Ultralytics YOLO, AGPL-3.0). Sample videos: Intel IoT DevKit
sample-videos, CC BY 4.0. Qwen2.5-VL: Qwen licence. This is an independent demo, not affiliated
with any vendor.
