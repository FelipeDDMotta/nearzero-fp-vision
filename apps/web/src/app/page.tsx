import Link from "next/link";
import { REPO_URL } from "@/lib/api";
import { aggregate, readIndex } from "@/lib/server-data";

const LOOP = [
  { n: "01", t: "Detect + track", d: "YOLO11n + ByteTrack on every clip. Low thresholds on purpose, so weak tracks are kept, not hidden." },
  { n: "02", t: "Route by confidence", d: "Confident, stable tracks alert directly. Only the uncertain band goes to the agent. Noise below the floor is dropped." },
  { n: "03", t: "Verifier agent", d: "Claude looks at the crop and asks for more evidence only when it needs it (context view, other frames). Then it rules." },
  { n: "04", t: "Cascade + audit", d: "Cheap model first, strong model only when unsure. A skeptical auditor re-checks a sample, and disagreements go to a human." },
  { n: "05", t: "Human review", d: "Operators label only the escalated cases. Their labels are ground truth and override the agent." },
  { n: "06", t: "Distill", d: "Agent and human decisions train a small open VLM (Qwen2.5-VL + LoRA), served with vLLM, so the same check can run at the edge." },
];

export default async function Home() {
  const index = await readIndex();
  const a = aggregate(index);
  const pct = (x: number, of: number) => (of ? Math.round((100 * x) / of) : 0);

  return (
    <div className="grid-bg">
      <section className="mx-auto max-w-[1400px] px-4 pb-12 pt-16">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">Agentic false-positive triage for video analytics</p>
        <h1 className="mt-4 max-w-4xl text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">
          The detector flags what it&apos;s unsure about. <span className="text-accent">An agent decides.</span> The decisions train the next model.
        </h1>
        <p className="mt-5 max-w-2xl text-lg text-muted">
          False alarms come from rain, glare, low light and clutter, and they are why operators stop trusting analytics. NearZeroFP Vision sends only the
          uncertain detections to a vision-language agent. The agent verifies each one, extracts searchable attributes, escalates what it can&apos;t
          resolve, and turns every decision into training data.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link href="/console" className="rounded-md bg-accent px-5 py-2.5 text-sm font-semibold text-black hover:brightness-110">
            Open the operator console →
          </Link>
          <Link href="/loop" className="rounded-md border border-line px-5 py-2.5 text-sm hover:bg-panel-2">
            See the distillation results
          </Link>
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="rounded-md border border-line px-5 py-2.5 text-sm hover:bg-panel-2">
            Code &amp; write-up ↗
          </a>
        </div>
      </section>

      {a.tracks > 0 && (
        <section className="mx-auto max-w-[1400px] px-4">
          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line md:grid-cols-4">
            <Kpi v={`${pct(a.trackerAuto, a.trackerTracks)}%`} l={`of ${a.trackerTracks} tracker tracks were confident: they alerted with no LLM call`} />
            <Kpi v={a.trackerUncertain + a.candidates} l={`uncertain items sent to the agent (${a.trackerUncertain} weak tracks + ${a.candidates} below-threshold candidates)`} />
            <Kpi v={`${pct(a.confirmed + a.rejected, a.triaged)}%`} l="of agent cases auto-resolved without a human" accent />
            <Kpi v={`${pct(a.escalated, a.triaged)}%`} l="escalated to the human review queue" />
            <Kpi v={a.rejected} l="false alarms suppressed before reaching an operator" accent />
            <Kpi v={a.candConfirmed} l="real objects recovered from below the detection threshold (recall won back)" accent />
            <Kpi v={`${a.without} → ${a.with}`} l="operator alerts at a low threshold, without → with the agent" />
            <Kpi v={`$${a.triaged ? (a.cost / a.triaged).toFixed(4) : "0"}`} l={`average agent cost per uncertain item ($${a.cost.toFixed(2)} for the whole demo)`} />
          </div>
          <p className="mt-2 text-xs text-muted">
            From the precomputed run over {a.clips} clips. {a.strong} of {a.triaged} cases needed the strong model, and {a.audits} independent audits found{" "}
            {a.disagreements} disagreement(s).
          </p>
        </section>
      )}

      <section className="mx-auto max-w-[1400px] px-4 py-16">
        <h2 className="text-2xl font-semibold">The loop</h2>
        <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {LOOP.map((s) => (
            <div key={s.n} className="rounded-xl border border-line bg-panel p-5">
              <div className="font-mono text-xs text-accent">{s.n}</div>
              <div className="mt-2 font-medium">{s.t}</div>
              <p className="mt-2 text-sm leading-relaxed text-muted">{s.d}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-[1400px] px-4 pb-8">
        <h2 className="text-2xl font-semibold">Harness design choices</h2>
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <Card t="Evidence on request, bounded">
            The agent starts with one tight crop. It has to ask for the context view or for other frames, and it gets at most four turns. Easy cases stay
            cheap, and hard cases get more evidence.
          </Card>
          <Card t="Trust, but verify">
            A deterministic sample of auto-resolved cases goes to an independent auditor. The auditor sees all the evidence up front and has a skeptical
            prompt. Any disagreement goes to a human, so &quot;done&quot; is checked rather than assumed.
          </Card>
          <Card t="Every decision is data">
            Rejections become hard negatives, and human labels override the agent. The same records fine-tune a small open VLM that copies the verifier
            at a fraction of the cost.
          </Card>
        </div>
      </section>
    </div>
  );
}

function Kpi({ v, l, accent }: { v: string | number; l: string; accent?: boolean }) {
  return (
    <div className="bg-panel p-5">
      <div className={`font-mono text-2xl ${accent ? "text-accent" : ""}`}>{v}</div>
      <div className="mt-1 text-xs leading-snug text-muted">{l}</div>
    </div>
  );
}

function Card({ t, children }: { t: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-panel p-5">
      <div className="font-medium">{t}</div>
      <p className="mt-2 text-sm leading-relaxed text-muted">{children}</p>
    </div>
  );
}
