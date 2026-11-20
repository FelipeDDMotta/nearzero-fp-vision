"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchClip, fetchIndex } from "@/lib/api";
import { STATUS_STYLE, type ClipResult, type Track } from "@/lib/types";

type Label = { verdict: "true_positive" | "false_positive" | "uncertain"; corrected_class: string; confidence: number; reason: string };
type Item = { key: string; clipId: string; clipTitle: string; track: Track };
const STORE = "nearzero-human-labels-v1";

function load(): Record<string, Label> {
  try {
    return JSON.parse(localStorage.getItem(STORE) || "{}");
  } catch {
    return {};
  }
}

export default function ReviewPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [labels, setLabels] = useState<Record<string, Label>>({});
  const [i, setI] = useState(0);
  const [blind, setBlind] = useState(true);
  const [scope, setScope] = useState<"escalated" | "all">("escalated");
  const [clipFilter, setClipFilter] = useState<string>("");
  useEffect(() => {
    const c = new URLSearchParams(window.location.search).get("clip");
    if (c) {
      setClipFilter(c);
      setScope("all");
    }
  }, []);

  useEffect(() => {
    setLabels(load());
    (async () => {
      const idx = await fetchIndex();
      const clips: ClipResult[] = await Promise.all(idx.map((e) => fetchClip(e.clip.id)));
      const out: Item[] = [];
      for (const c of clips)
        for (const t of c.tracks)
          if (t.triage) out.push({ key: `${c.clip.id}:${t.id}`, clipId: c.clip.id, clipTitle: c.clip.title, track: t });
      out.sort((a, b) => (a.track.triage!.status === "escalated" ? 0 : 1) - (b.track.triage!.status === "escalated" ? 0 : 1));
      setItems(out);
    })();
  }, []);

  const queue = useMemo(
    () =>
      items.filter((x) => (scope === "all" || x.track.triage!.status === "escalated") && (!clipFilter || x.clipId === clipFilter)),
    [items, scope, clipFilter]
  );
  const clipIds = useMemo(() => Array.from(new Set(items.map((x) => x.clipId))), [items]);
  const cur = queue[Math.min(i, Math.max(0, queue.length - 1))];

  const save = useCallback(
    (verdict: Label["verdict"]) => {
      if (!cur) return;
      const next = {
        ...labels,
        [cur.key]: { verdict, corrected_class: verdict === "false_positive" ? "none" : cur.track.triage?.final.corrected_class || cur.track.cls, confidence: 1, reason: "human review" },
      };
      setLabels(next);
      try {
        localStorage.setItem(STORE, JSON.stringify(next));
      } catch {}
      setI((x) => Math.min(x + 1, queue.length - 1));
    },
    [cur, labels, queue.length]
  );

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === "t" || e.key === "T") save("true_positive");
      else if (e.key === "f" || e.key === "F") save("false_positive");
      else if (e.key === "u" || e.key === "U") save("uncertain");
      else if (e.key === "ArrowRight") setI((x) => Math.min(x + 1, queue.length - 1));
      else if (e.key === "ArrowLeft") setI((x) => Math.max(0, x - 1));
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [save, queue.length]);

  const labelled = items.filter((x) => labels[x.key] && labels[x.key].verdict !== "uncertain");
  const agree = labelled.filter((x) => labels[x.key].verdict === x.track.triage!.final.verdict).length;

  const exportLabels = () => {
    const blob = new Blob([JSON.stringify(labels, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "human_labels.json";
    a.click();
  };

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Human review queue</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted">
            Escalated cases land here. Your labels become ground truth, override the agent in the training data, and measure how often the agent is
            right. Blind mode hides the agent&apos;s verdict so it can&apos;t bias you.
          </p>
        </div>
        <div className="text-right text-xs text-muted">
          <div>
            labelled <b className="text-text">{labelled.length}</b> / {items.length} · agent agreement{" "}
            <b className="text-text">{labelled.length ? Math.round((100 * agree) / labelled.length) : 0}%</b>
          </div>
          <div className="mt-2 flex justify-end gap-2">
            <button onClick={() => setBlind(!blind)} className="rounded border border-line px-2 py-1 hover:bg-panel-2">
              blind mode: {blind ? "on" : "off"}
            </button>
            <button onClick={() => { setScope(scope === "escalated" ? "all" : "escalated"); setI(0); }} className="rounded border border-line px-2 py-1 hover:bg-panel-2">
              queue: {scope === "escalated" ? "escalated only" : "all agent cases"}
            </button>
            <select
              value={clipFilter}
              onChange={(e) => {
                setClipFilter(e.target.value);
                setI(0);
              }}
              className="rounded border border-line bg-panel px-2 py-1"
            >
              <option value="">all clips</option>
              {clipIds.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <button onClick={exportLabels} className="rounded border border-accent px-2 py-1 text-accent hover:bg-accent/10">
              export labels
            </button>
          </div>
        </div>
      </div>

      {!cur ? (
        <div className="mt-10 rounded-lg border border-line bg-panel p-8 text-center text-muted">{items.length ? "Queue empty for this scope." : "Loading…"}</div>
      ) : (
        <div className="mt-6 grid gap-5 lg:grid-cols-[1fr_340px]">
          <div className="space-y-3 rounded-lg border border-line bg-panel p-4">
            <div className="flex items-center justify-between font-mono text-xs text-muted">
              <span>
                {i + 1} / {queue.length} · {cur.clipTitle} · track #{cur.track.id}
              </span>
              {labels[cur.key] && <span className="text-accent">labelled: {labels[cur.key].verdict}</span>}
            </div>
            <div className="grid grid-cols-[1fr_1.6fr] gap-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`/${cur.track.crops.best}`} alt="crop" className="h-64 w-full rounded border border-line bg-black object-contain" />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`/${cur.track.crops.context}`} alt="context" className="h-64 w-full rounded border border-line bg-black object-contain" />
            </div>
            <div className="flex gap-2">
              {cur.track.crops.frames.map((f) => (
                // eslint-disable-next-line @next/next/no-img-element
                <img key={f} src={`/${f}`} alt="" className="h-24 flex-1 rounded border border-line bg-black object-contain" />
              ))}
            </div>
            <div className="text-sm">
              Detector says <b>{cur.track.cls}</b> (score {cur.track.score.toFixed(2)}, {cur.track.n} frames). Is it a real {cur.track.cls} or another
              supported object?
            </div>
            <div className="flex flex-wrap gap-2">
              <button onClick={() => save("true_positive")} className="rounded bg-emerald-500/90 px-4 py-2 text-sm font-semibold text-black">
                Real object <span className="kbd ml-1 border-black/30 text-black/70">T</span>
              </button>
              <button onClick={() => save("false_positive")} className="rounded bg-red-500/90 px-4 py-2 text-sm font-semibold text-black">
                False positive <span className="kbd ml-1 border-black/30 text-black/70">F</span>
              </button>
              <button onClick={() => save("uncertain")} className="rounded border border-line px-4 py-2 text-sm">
                Can&apos;t tell <span className="kbd ml-1">U</span>
              </button>
              <span className="ml-auto self-center text-xs text-muted">
                <span className="kbd">←</span> <span className="kbd">→</span> navigate
              </span>
            </div>
          </div>
          <div className="rounded-lg border border-line bg-panel p-4 text-sm">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">Agent decision</h3>
            {blind && !labels[cur.key] ? (
              <p className="mt-3 text-muted">Hidden until you label (blind mode).</p>
            ) : (
              <div className="mt-3 space-y-2">
                <div style={{ color: STATUS_STYLE[cur.track.triage!.status].color }}>{STATUS_STYLE[cur.track.triage!.status].label}</div>
                <div>
                  {cur.track.triage!.final.verdict} · {Math.round(cur.track.triage!.final.confidence * 100)}%
                </div>
                <p className="text-muted">{cur.track.triage!.final.reason}</p>
                {cur.track.triage!.audit && (
                  <p className="text-xs text-muted">
                    Audit: {cur.track.triage!.audit.verdict}
                    {cur.track.triage!.disagreement ? " (disagreed)" : ""}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
