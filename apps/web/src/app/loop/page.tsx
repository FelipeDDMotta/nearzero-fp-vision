import { API_URL, REPO_URL } from "@/lib/api";
import { aggregate, readIndex, readJson } from "@/lib/server-data";

interface BenchRow {
  name: string;
  tag: string;
  n_scored: number;
  ground_truth: string;
  accuracy: number | null;
  fp_rejection: number | null;
  tp_retention: number | null;
  fp_counts?: string;
  tp_counts?: string;
  teacher_agreement: number;
  valid_json: number;
  latency_ms: number;
  usd_per_1k: number;
  where: string;
}
interface Bench {
  holdout_clip: string;
  rows: BenchRow[];
  notes: string[];
  training?: {
    base_model: string;
    method: string;
    samples: number;
    epochs: number;
    steps: number;
    runtime_s: number;
    loss_first: number;
    loss_last: number;
    hardware: string;
    hf_repo: string;
  };
}

interface V2Row {
  name: string;
  tag: string;
  hard_n: number;
  hard_acc: number;
  hard_fp_rejected: string;
  hard_real_kept: string;
  easy_n: number;
  easy_acc: number;
  latency_ms: number;
  where: string;
  params: string;
}
interface BenchV2 {
  split: string;
  test_hard_n: number;
  test_easy_n: number;
  test_label_sources: Record<string, number>;
  rows: V2Row[];
  edge_cv: { total: { hard_n: number; hard_acc: number; hard_fp_rejected: string; hard_real_kept: string }; per_fold: Record<string, { hard_n: number; hard_acc: number }> } | null;
  teacher_vs_human: { n: number; agreement: number } | null;
  training: { samples_total_v2: number; epochs?: number; steps?: number; runtime_s?: number; loss_first?: number; loss_last?: number; hf_repo: string };
}

const pct = (x: number | null) => (x === null ? "is ground truth" : `${Math.round(x * 100)}%`);

export default async function LoopPage() {
  const index = await readIndex();
  const a = aggregate(index);
  const bench = await readJson<Bench>("bench.json");
  const v2 = await readJson<BenchV2>("bench_v2.json");

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-8">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">Closing the loop</p>
      <h1 className="mt-2 text-3xl font-semibold">From agent decisions to a model that runs at the edge</h1>
      <p className="mt-3 max-w-3xl text-muted">
        Calling a frontier model for every uncertain detection works, but it costs money, needs a network connection and adds latency. Each decision
        the agent makes (and each human correction) is a labelled example. Here those examples train a small open vision-language model with QLoRA. The
        model is served with vLLM and compared with its teacher on a camera it never saw.
      </p>

      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        <Box v={a.confirmed + a.rejected} l="auto-resolved agent decisions (teacher labels)" />
        <Box v={a.rejected} l="hard negatives: false positives the agent rejected" />
        <Box v={a.escalated} l="escalated cases waiting for human labels" />
      </div>

      {v2 && v2.rows.length > 0 && (
        <section className="mt-10">
          <h2 className="text-xl font-semibold">v2 benchmark: leak-free, more data</h2>
          <p className="mt-1 max-w-3xl text-sm text-muted">
            {v2.split}. Headline metric: accuracy on the <b className="text-text">hard</b> items ({v2.test_hard_n}: detections a real system would
            actually be unsure about). {v2.training.samples_total_v2} samples in total, most with free labels (detector-confident tracks, detections
            in scenes with no people, synthetic background boxes), so no extra API spend.
          </p>
          <div className="mt-4 overflow-x-auto rounded-lg border border-line">
            <table className="w-full min-w-[760px] text-sm">
              <thead className="bg-panel-2 text-left text-xs text-muted">
                <tr>
                  {["Model", "Params", "Hard acc", "FP rejected", "Real kept", "Easy acc", "Latency", "Where"].map((h) => (
                    <th key={h} className="px-3 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {v2.rows.map((r) => (
                  <tr key={r.tag} className={r.tag === "v2_lora" ? "bg-accent/5" : ""}>
                    <td className="px-3 py-2 font-medium">{r.name}</td>
                    <td className="px-3 py-2 font-mono">{r.params}</td>
                    <td className="px-3 py-2 font-mono">
                      {Math.round(r.hard_acc * 100)}% <span className="text-muted">(n={r.hard_n})</span>
                    </td>
                    <td className="px-3 py-2 font-mono">{r.hard_fp_rejected}</td>
                    <td className="px-3 py-2 font-mono">{r.hard_real_kept}</td>
                    <td className="px-3 py-2 font-mono">{Math.round(r.easy_acc * 100)}%</td>
                    <td className="px-3 py-2 font-mono">{Math.round(r.latency_ms)} ms</td>
                    <td className="px-3 py-2 text-xs text-muted">{r.where}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">
            Test labels: {Object.entries(v2.test_label_sources).map(([k, n]) => `${k} ${n}`).join(" · ")}.
            {v2.teacher_vs_human && <> The Claude teacher agrees with human review on {Math.round(v2.teacher_vs_human.agreement * 100)}% of {v2.teacher_vs_human.n} items.</>}
          </p>
          {v2.edge_cv && (
            <div className="mt-4 rounded-lg border border-line bg-panel p-4 text-sm">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted">Edge model, leave-one-source-video-out cross-validation</div>
              <p className="mt-2 text-[13px] text-muted">
                Every source video is held out once (all its synthetic variants together). Hard-item accuracy over all folds:{" "}
                <b className="text-text">{Math.round(v2.edge_cv.total.hard_acc * 100)}%</b> (n={v2.edge_cv.total.hard_n}, FP rejected{" "}
                {v2.edge_cv.total.hard_fp_rejected}, real kept {v2.edge_cv.total.hard_real_kept}). Per held-out video:{" "}
                {Object.entries(v2.edge_cv.per_fold)
                  .map(([g, f]) => `${g} ${Math.round(f.hard_acc * 100)}% (n=${f.hard_n})`)
                  .join(" · ")}
                .
              </p>
            </div>
          )}
        </section>
      )}

      <section className="mt-10">
        <h2 className="text-xl font-semibold">{v2 && v2.rows.length > 0 ? "v1 benchmark (first run, kept for the record)" : "Benchmark: teacher vs student"}</h2>
        {bench ? (
          <>
            <p className="mt-1 text-sm text-muted">
              Held-out clip: <span className="font-mono text-text">{bench.holdout_clip}</span>
            </p>
            <div className="mt-4 overflow-x-auto rounded-lg border border-line">
              <table className="w-full min-w-[760px] text-sm">
                <thead className="bg-panel-2 text-left text-xs text-muted">
                  <tr>
                    {["Model", "Accuracy", "FP rejected", "Real kept", "Agrees w/ teacher", "Valid JSON", "Latency", "$ / 1k events", "Where"].map((h) => (
                      <th key={h} className="px-3 py-2 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {bench.rows.map((r) => (
                    <tr key={r.tag} className={r.tag === "lora" ? "bg-accent/5" : ""}>
                      <td className="px-3 py-2 font-medium">{r.name}</td>
                      <td className="px-3 py-2 font-mono">{pct(r.accuracy)}</td>
                      <td className="px-3 py-2 font-mono">{r.fp_counts && r.fp_counts !== "-" ? r.fp_counts : pct(r.fp_rejection)}</td>
                      <td className="px-3 py-2 font-mono">{r.tp_counts && r.tp_counts !== "-" ? r.tp_counts : pct(r.tp_retention)}</td>
                      <td className="px-3 py-2 font-mono">{pct(r.teacher_agreement)}</td>
                      <td className="px-3 py-2 font-mono">{pct(r.valid_json)}</td>
                      <td className="px-3 py-2 font-mono">{Math.round(r.latency_ms)} ms</td>
                      <td className="px-3 py-2 font-mono">${r.usd_per_1k.toFixed(2)}</td>
                      <td className="px-3 py-2 text-xs text-muted">{r.where}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-muted">
              Scored on n={bench.rows[0]?.n_scored} samples against {bench.rows[0]?.ground_truth} labels.
            </p>
            <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-muted">
              {bench.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
            {bench.training && (
              <div className="mt-6 grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border border-line bg-panel p-4 text-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted">Training run</div>
                  <ul className="mt-2 space-y-1 text-[13px]">
                    <li>{bench.training.base_model}, {bench.training.method}</li>
                    <li>
                      {bench.training.samples} samples · {bench.training.epochs} epochs · {bench.training.steps} steps · {Math.round(bench.training.runtime_s / 60)} min on{" "}
                      {bench.training.hardware}
                    </li>
                    <li>
                      Training loss {bench.training.loss_first} → <b>{bench.training.loss_last}</b>
                    </li>
                    <li>
                      Weights:{" "}
                      <a className="text-accent hover:underline" href={`https://huggingface.co/${bench.training.hf_repo}`} target="_blank" rel="noreferrer">
                        {bench.training.hf_repo}
                      </a>
                    </li>
                  </ul>
                </div>
                <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-4 text-sm">
                  <div className="text-xs font-semibold uppercase tracking-wider text-amber-300">What the first run taught me</div>
                  <p className="mt-2 text-[13px] leading-relaxed text-muted">
                    The student learned the <b className="text-text">output contract</b> (clean JSON without code fences, and teacher-like confidence, with
                    mean confidence going from 0.27 to 0.63) but <b className="text-text">not better judgment</b>: accuracy on the held-out clips is unchanged
                    with 168 training samples, and with n=23 one item is about 4 points. Next: more and harder data (synthetic hard negatives, more
                    cameras), adapting the vision layers, and human labels as ground truth instead of the teacher.
                  </p>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="mt-4 rounded-lg border border-dashed border-line p-6 text-sm text-muted">
            The benchmark appears here after the Kaggle fine-tuning and vLLM evaluation run (<span className="font-mono">distill/</span>).
          </div>
        )}
      </section>

      <section className="mt-10 grid gap-4 md:grid-cols-3">
        <Step n="1" t="Export">
          <span className="font-mono">export_vlm_dataset.py</span> turns each decision into (crop + context, prompt) → verdict JSON. One whole camera is
          held out, so no frames leak between train and test.
        </Step>
        <Step n="2" t="Fine-tune">
          QLoRA (4-bit, r=16) on Qwen2.5-VL-3B-Instruct with Unsloth, on a free Kaggle T4. Only the language layers are adapted, and the merged weights
          are pushed to the Hugging Face Hub.
        </Step>
        <Step n="3" t="Serve">
          vLLM serves the base and fine-tuned models for the benchmark. <span className="font-mono">serving/modal_app.py</span> deploys the
          fine-tuned model as a scale-to-zero GPU endpoint for the console&apos;s &quot;Open VLM&quot; toggle, which is active only when that endpoint is configured.
        </Step>
      </section>

      <div className="mt-8 flex flex-wrap gap-3 text-sm">
        <a href={`${API_URL}/dataset/export`} className="rounded-md border border-line px-4 py-2 hover:bg-panel-2">
          Download the corrected dataset (.zip)
        </a>
        <a href={`${REPO_URL}/tree/main/distill`} target="_blank" rel="noreferrer" className="rounded-md border border-line px-4 py-2 hover:bg-panel-2">
          Training notebook &amp; scripts ↗
        </a>
      </div>
    </div>
  );
}

function Box({ v, l }: { v: number; l: string }) {
  return (
    <div className="rounded-lg border border-line bg-panel p-4">
      <div className="font-mono text-2xl">{v}</div>
      <div className="mt-1 text-xs text-muted">{l}</div>
    </div>
  );
}

function Step({ n, t, children }: { n: string; t: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-line bg-panel p-4">
      <div className="font-mono text-xs text-accent">step {n}</div>
      <div className="mt-1 font-medium">{t}</div>
      <p className="mt-2 text-sm leading-relaxed text-muted">{children}</p>
    </div>
  );
}
