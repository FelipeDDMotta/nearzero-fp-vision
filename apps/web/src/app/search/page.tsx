"use client";

import Link from "next/link";
import { useState } from "react";
import { search, type SearchHit } from "@/lib/api";

const EXAMPLES = ["person in a dark top with a backpack", "white car", "someone riding a bicycle", "person wearing a hat", "red vehicle", "worker in a high-vis vest"];

export default function SearchPage() {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<{ filter: Record<string, unknown>; count: number; results: SearchHit[] } | null>(null);

  const run = async (query: string) => {
    if (!query.trim()) return;
    setQ(query);
    setBusy(true);
    setErr(null);
    try {
      setRes(await search(query));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const activeFilter = res
    ? Object.entries(res.filter).filter(([, v]) => (Array.isArray(v) ? v.length : v !== "any"))
    : [];

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-6">
      <h1 className="text-2xl font-semibold">Appearance search</h1>
      <p className="mt-1 max-w-2xl text-sm text-muted">
        Search every clip in plain English. A small model only turns the query into a structured filter. Matching is plain code over the attributes the
        agents pulled from each track, so every result shows why it matched.
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run(q);
        }}
        className="mt-5 flex gap-2"
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. person in a red jacket carrying a bag"
          className="flex-1 rounded-md border border-line bg-panel px-4 py-2.5 text-sm outline-none focus:border-accent"
        />
        <button disabled={busy} className="rounded-md bg-accent px-5 text-sm font-semibold text-black disabled:opacity-50">
          {busy ? "Searching…" : "Search"}
        </button>
      </form>
      <div className="mt-3 flex flex-wrap gap-2">
        {EXAMPLES.map((e) => (
          <button key={e} onClick={() => run(e)} className="rounded-full border border-line px-3 py-1 text-xs text-muted hover:bg-panel-2 hover:text-text">
            {e}
          </button>
        ))}
      </div>
      {err && <p className="mt-4 text-sm text-red-400">{err}</p>}
      {res && (
        <div className="mt-6">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
            <span>Parsed filter:</span>
            {activeFilter.length === 0 && <span>(no constraints)</span>}
            {activeFilter.map(([k, v]) => (
              <span key={k} className="rounded border border-accent/40 bg-accent/10 px-2 py-0.5 font-mono text-accent">
                {k}={Array.isArray(v) ? v.join("+") : String(v)}
              </span>
            ))}
            <span className="ml-auto">{res.count} match(es)</span>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
            {res.results.map((h) => (
              <Link
                key={`${h.clip_id}-${h.track_id}`}
                href={`/console?clip=${h.clip_id}&track=${h.track_id}`}
                className="group overflow-hidden rounded-lg border border-line bg-panel hover:border-accent"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`/${h.crop}`} alt="" className="h-40 w-full bg-black object-contain" loading="lazy" />
                <div className="space-y-1 p-2 text-[11px]">
                  <div className="flex justify-between">
                    <span className="font-medium text-text">{h.cls}</span>
                    <span className="font-mono text-muted">
                      {h.clip_id} · {h.t0.toFixed(1)}s
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {h.matched.map((m) => (
                      <span key={m} className="rounded bg-emerald-500/15 px-1.5 text-emerald-300">
                        {m}
                      </span>
                    ))}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
