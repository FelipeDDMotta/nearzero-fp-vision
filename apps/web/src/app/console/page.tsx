"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { EventDetail } from "@/components/EventDetail";
import { VideoOverlay, type VideoHandle } from "@/components/VideoOverlay";
import { fetchClip, fetchIndex } from "@/lib/api";
import { attrSummary, displayStatus, STATUS_STYLE, type ClipResult, type DisplayStatus, type IndexEntry } from "@/lib/types";

const FILTERS: { key: DisplayStatus; default: boolean }[] = [
  { key: "auto", default: true },
  { key: "confirmed", default: true },
  { key: "rejected", default: true },
  { key: "escalated", default: true },
  { key: "pending", default: true },
  { key: "dropped", default: false },
];

function ConsoleInner() {
  const params = useSearchParams();
  const router = useRouter();
  const [index, setIndex] = useState<IndexEntry[]>([]);
  const [clip, setClip] = useState<ClipResult | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [visible, setVisible] = useState<Set<DisplayStatus>>(new Set(FILTERS.filter((f) => f.default).map((f) => f.key)));
  const [feedFilter, setFeedFilter] = useState<"agent" | "all">("agent");
  const video = useRef<VideoHandle>(null);
  const clipId = params.get("clip") || index[0]?.clip.id;

  useEffect(() => {
    fetchIndex().then(setIndex);
  }, []);

  useEffect(() => {
    if (!clipId) return;
    setClip(null);
    fetchClip(clipId).then((c) => {
      setClip(c);
      const t = Number(params.get("track"));
      if (t) {
        setSelected(t);
        const tr = c.tracks.find((x) => x.id === t);
        setTimeout(() => tr && video.current?.seek(tr.t0 + 0.05), 400);
      } else setSelected(null);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipId]);

  const feed = useMemo(() => {
    if (!clip) return [];
    return clip.tracks
      .filter((t) => (feedFilter === "agent" ? t.band === "uncertain" : t.band !== "drop"))
      .sort((a, b) => a.t0 - b.t0);
  }, [clip, feedFilter]);

  const sel = clip?.tracks.find((t) => t.id === selected) || null;
  const s = clip?.summary;

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-5">
      <div className="mb-4 flex flex-wrap gap-2">
        {index.map((e) => (
          <button
            key={e.clip.id}
            onClick={() => router.push(`/console?clip=${e.clip.id}`)}
            className={`rounded-md border px-3 py-1.5 text-left text-xs ${e.clip.id === clipId ? "border-accent bg-accent/10" : "border-line hover:bg-panel-2"}`}
          >
            <div className="font-medium">{e.clip.title}</div>
            <div className="font-mono text-[10px] text-muted">
              {e.summary ? `${e.summary.tracks} tracks · ${e.summary.bands.uncertain ?? 0} to agent` : `${e.duration.toFixed(0)}s`}
            </div>
          </button>
        ))}
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_420px]">
        <div className="space-y-3">
          {clip ? (
            <VideoOverlay ref={video} src={`/${clip.clip.video}`} tracks={clip.tracks} visible={visible} selectedId={selected} onSelect={setSelected} />
          ) : (
            <div className="aspect-video w-full animate-pulse rounded-lg border border-line bg-panel" />
          )}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {FILTERS.map((f) => {
              const st = STATUS_STYLE[f.key];
              const on = visible.has(f.key);
              return (
                <button
                  key={f.key}
                  onClick={() => {
                    const n = new Set(visible);
                    if (on) n.delete(f.key);
                    else n.add(f.key);
                    setVisible(n);
                  }}
                  className={`flex items-center gap-1.5 rounded border px-2 py-1 ${on ? "border-line bg-panel-2" : "border-transparent opacity-40"}`}
                >
                  <span className="h-2.5 w-2.5 rounded-sm border-2" style={{ borderColor: st.color, borderStyle: st.dash ? "dashed" : "solid" }} />
                  {st.label}
                </button>
              );
            })}
          </div>
          {clip && (
            <div className="rounded-lg border border-line bg-panel p-4">
              <div className="text-sm font-medium">{clip.clip.title}</div>
              <div className="text-xs text-muted">{clip.clip.scene}</div>
              {s && (
                <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="Alerts without agent" value={s.alerts_without_agent} />
                  <Stat label="Alerts with agent" value={s.alerts_with_agent} accent />
                  <Stat label="Agent auto-resolved" value={`${s.auto_resolved_pct}%`} />
                  <Stat label="Agent cost (clip)" value={`$${s.cost_usd.toFixed(3)}`} />
                </div>
              )}
              <div className="mt-3 text-[11px] text-muted">{clip.clip.attribution}</div>
            </div>
          )}
        </div>

        <aside className="flex max-h-[calc(100vh-110px)] flex-col overflow-hidden rounded-lg border border-line bg-panel lg:sticky lg:top-[70px]">
          {sel && clip ? (
            <div className="scroll-thin flex-1 overflow-y-auto p-4">
              <button onClick={() => setSelected(null)} className="mb-3 text-xs text-muted hover:text-text">
                ← back to events
              </button>
              <EventDetail key={`${clip.clip.id}-${sel.id}`} clipId={clip.clip.id} track={sel} onSeek={(t) => video.current?.seek(t)} />
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between border-b border-line px-4 py-3">
                <h2 className="text-sm font-semibold">Events</h2>
                <div className="flex overflow-hidden rounded border border-line text-[11px]">
                  {(["agent", "all"] as const).map((k) => (
                    <button key={k} onClick={() => setFeedFilter(k)} className={`px-2 py-0.5 ${feedFilter === k ? "bg-panel-2 text-text" : "text-muted"}`}>
                      {k === "agent" ? "Sent to agent" : "All alerts"}
                    </button>
                  ))}
                </div>
              </div>
              <ul className="scroll-thin flex-1 divide-y divide-line overflow-y-auto">
                {feed.map((t) => {
                  const st = displayStatus(t);
                  const style = STATUS_STYLE[st];
                  return (
                    <li key={t.id}>
                      <button
                        onClick={() => {
                          setSelected(t.id);
                          video.current?.seek(t.t0 + 0.05);
                        }}
                        className="flex w-full gap-3 px-4 py-2.5 text-left hover:bg-panel-2"
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={`/${t.crops.best}`} alt="" className="h-12 w-12 shrink-0 rounded border border-line bg-black object-contain" loading="lazy" />
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center justify-between gap-2">
                            <span className="truncate text-sm">
                              {t.cls} <span className="font-mono text-[11px] text-muted">#{t.id}</span>
                            </span>
                            <span className="shrink-0 text-[11px]" style={{ color: style.color }}>
                              {style.label}
                            </span>
                          </div>
                          <div className="truncate font-mono text-[11px] text-muted">
                            {t.t0.toFixed(1)}s · score {t.score.toFixed(2)}
                            {t.triage ? ` · ${Math.round(t.triage.final.confidence * 100)}% ${t.triage.steps.length > 1 ? "· escalated to strong" : ""}` : ""}
                          </div>
                          <div className="truncate text-[11px] text-muted">{t.triage?.final.reason || attrSummary(t.describe?.attributes)}</div>
                        </div>
                      </button>
                    </li>
                  );
                })}
                {clip && feed.length === 0 && <li className="p-4 text-sm text-muted">No events in this filter.</li>}
              </ul>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className="rounded-md border border-line bg-panel-2 px-3 py-2">
      <div className={`font-mono text-lg ${accent ? "text-accent" : ""}`}>{value}</div>
      <div className="text-[11px] text-muted">{label}</div>
    </div>
  );
}

export default function ConsolePage() {
  return (
    <Suspense fallback={null}>
      <ConsoleInner />
    </Suspense>
  );
}
