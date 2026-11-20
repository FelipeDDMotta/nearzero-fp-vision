"use client";

import { useState } from "react";
import { liveTriage } from "@/lib/api";
import { attrSummary, displayStatus, STATUS_STYLE, type Triage, type Track } from "@/lib/types";

const VERDICT_LABEL: Record<string, string> = {
  true_positive: "Real object",
  false_positive: "False positive",
  uncertain: "Uncertain",
};

function StatusPill({ status }: { status: keyof typeof STATUS_STYLE }) {
  const s = STATUS_STYLE[status];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium" style={{ borderColor: s.color, color: s.color }}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: s.color }} />
      {s.label}
    </span>
  );
}

function TriageView({ tri, title }: { tri: Triage; title: string }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted">{title}</h4>
        <span className="font-mono text-[11px] text-muted">
          ${tri.cost_usd.toFixed(4)} · {(tri.latency_ms / 1000).toFixed(1)}s
        </span>
      </div>
      <ol className="space-y-2">
        {tri.steps.map((s, i) => (
          <li key={i} className="rounded-md border border-line bg-panel-2 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="font-mono text-accent">
                {i === 0 ? "1. first pass" : `${i + 1}. escalated`} · {s.model}
              </span>
              <span className="font-mono text-muted">
                {s.input_tokens}→{s.output_tokens} tok · {(s.latency_ms / 1000).toFixed(1)}s
              </span>
            </div>
            {s.tools_used.length > 0 && (
              <div className="mt-2 space-y-1">
                {s.tools_used.map((u, k) => (
                  <div key={k} className="text-[11px] text-muted">
                    <span className="font-mono text-text">⤷ {u.tool}</span> {u.why && <>: “{u.why}”</>}
                  </div>
                ))}
              </div>
            )}
            <div className="mt-2 text-sm">
              <b>{VERDICT_LABEL[s.verdict]}</b>
              {s.corrected_class && s.corrected_class !== "none" ? ` (${s.corrected_class})` : ""} · confidence {Math.round(s.confidence * 100)}%
            </div>
            <p className="mt-1 text-[13px] leading-snug text-muted">{s.reason}</p>
          </li>
        ))}
        {tri.audit && (
          <li className={`rounded-md border p-3 ${tri.disagreement ? "border-amber-500/60 bg-amber-500/5" : "border-line bg-panel-2"}`}>
            <div className="flex justify-between text-xs">
              <span className="font-mono text-violet-300">audit · {tri.audit.model}</span>
              <span className={tri.disagreement ? "text-amber-400" : "text-emerald-400"}>{tri.disagreement ? "disagrees → escalate" : "agrees"}</span>
            </div>
            <div className="mt-1 text-sm">
              {VERDICT_LABEL[tri.audit.verdict]} · {Math.round(tri.audit.confidence * 100)}%
            </div>
            <p className="mt-1 text-[13px] text-muted">{tri.audit.reason}</p>
          </li>
        )}
      </ol>
    </div>
  );
}

export function EventDetail({ clipId, track, onSeek }: { clipId: string; track: Track; onSeek: (t: number) => void }) {
  const [live, setLive] = useState<Triage | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [engine, setEngine] = useState<"claude" | "open">("claude");
  const st = displayStatus(track);
  const attrs = track.triage?.final.attributes || track.describe?.attributes;

  const run = async () => {
    setBusy(true);
    setErr(null);
    setLive(null);
    try {
      setLive(await liveTriage(clipId, track.id, engine));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="font-mono text-xs text-muted">
            track #{track.id} · {track.t0.toFixed(1)}s–{track.t1.toFixed(1)}s · {track.n} frames
          </div>
          <div className="mt-1 text-lg font-semibold">
            Detector: {track.cls} <span className="font-mono text-sm text-muted">score {track.score.toFixed(2)}</span>
          </div>
        </div>
        <StatusPill status={st} />
      </div>

      <div className="grid grid-cols-[1fr_2fr] gap-2">
        <figure>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`/${track.crops.best}`} alt="tight crop" className="h-36 w-full rounded border border-line bg-black object-contain" />
          <figcaption className="mt-1 text-[11px] text-muted">tight crop (what the agent sees first)</figcaption>
        </figure>
        <figure>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`/${track.crops.context}`} alt="context" className="h-36 w-full rounded border border-line bg-black object-contain" />
          <figcaption className="mt-1 text-[11px] text-muted">context view (available on request)</figcaption>
        </figure>
      </div>
      <div className="flex gap-2">
        {track.crops.frames.map((f, i) => (
          // eslint-disable-next-line @next/next/no-img-element
          <img key={f} src={`/${f}`} alt={`frame ${i}`} className="h-16 flex-1 rounded border border-line bg-black object-contain" />
        ))}
      </div>
      <button onClick={() => onSeek(track.t0)} className="text-xs text-accent hover:underline">
        ⟲ jump to {track.t0.toFixed(1)}s
      </button>

      {attrs && attrSummary(attrs) && (
        <div className="rounded-md border border-line bg-panel-2 px-3 py-2 text-xs">
          <span className="text-muted">Attributes: </span>
          {attrSummary(attrs)}
        </div>
      )}

      {track.band === "auto_accept" && (
        <p className="rounded-md border border-line bg-panel-2 p-3 text-sm text-muted">
          Confident, stable track (score ≥ 0.60, ≥ 4 frames). It alerts without an LLM call. This is where most of the cost saving comes from.
        </p>
      )}
      {track.band === "drop" && (
        <p className="rounded-md border border-line bg-panel-2 p-3 text-sm text-muted">Below the drop threshold (score &lt; 0.20). It&apos;s discarded, as a normal threshold would do.</p>
      )}
      {track.triage && <TriageView tri={track.triage} title="Precomputed decision" />}

      {track.band !== "drop" && (
        <div className="space-y-3 rounded-md border border-dashed border-line p-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex overflow-hidden rounded border border-line text-xs">
              {(["claude", "open"] as const).map((e) => (
                <button key={e} onClick={() => setEngine(e)} className={`px-2.5 py-1 ${engine === e ? "bg-accent text-black" : "text-muted hover:bg-panel-2"}`}>
                  {e === "claude" ? "Claude cascade" : "Open VLM (vLLM)"}
                </button>
              ))}
            </div>
            <button onClick={run} disabled={busy} className="rounded bg-accent px-3 py-1 text-xs font-semibold text-black disabled:opacity-50">
              {busy ? (engine === "open" ? "Warming GPU / running…" : "Agent running…") : "Re-run agent live"}
            </button>
          </div>
          {err && <p className="text-xs text-red-400">{err}</p>}
          {live && <TriageView tri={live} title={`Live result → ${STATUS_STYLE[live.status].label}`} />}
          {!live && !err && <p className="text-[11px] text-muted">Calls the deployed API. The live result can differ slightly from the precomputed one.</p>}
        </div>
      )}
    </div>
  );
}
