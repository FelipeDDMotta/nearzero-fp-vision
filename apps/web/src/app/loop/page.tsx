import { API_URL, REPO_URL } from "@/lib/api";
import { aggregate, readIndex, readJson } from "@/lib/server-data";

interface BenchRow {
  name: string;
  tag: string;
  n_scored: number;
  ground_truth: string;
  accuracy: number;
  fp_rejection: number;
  tp_retention: number;
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
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

export default async function LoopPage() {
  const index = await readIndex();
  const a = aggregate(index);
  const bench = await readJson<Bench>("bench.json");

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

      <section className="mt-10">
        <h2 className="text-xl font-semibold">Benchmark: teacher vs student</h2>
        {bench ? (
          <>
            <p className="mt-1 text-sm text-muted">
              Held-out clip: <span className="font-mono text-text">{bench.holdout_clip}</span>
            </p>
            <div className="mt-4 overflow-x-auto rounded-lg border border-line">
              <table className="w-full min-w-[760px] text-sm">
                <thead className="bg-panel-2 text-left text-xs text-muted">
                  <tr>
                    {["Model", "Accuracy", "FP rejection", "TP retention", "Agrees w/ teacher", "Valid JSON", "Latency", "$ / 1k events", "Where"].map((h) => (
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
                      <td className="px-3 py-2 font-mono">{pct(r.fp_rejection)}</td>
                      <td className="px-3 py-2 font-mono">{pct(r.tp_retention)}</td>
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
          vLLM serves the base and fine-tuned models for the benchmark. The fine-tuned model also runs as a scale-to-zero GPU endpoint on Modal behind
          the console&apos;s &quot;Open VLM&quot; toggle.
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
