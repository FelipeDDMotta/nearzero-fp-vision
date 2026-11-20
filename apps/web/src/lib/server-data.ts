import "server-only";
import { promises as fs } from "fs";
import path from "path";
import type { IndexEntry } from "./types";

const RESULTS = path.join(process.cwd(), "public", "results");

export async function readJson<T>(name: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(path.join(RESULTS, name), "utf-8")) as T;
  } catch {
    return null;
  }
}

export async function readIndex(): Promise<IndexEntry[]> {
  return (await readJson<IndexEntry[]>("index.json")) || [];
}

export function aggregate(index: IndexEntry[]) {
  const a = {
    clips: index.length,
    trackerTracks: 0,
    trackerAuto: 0,
    trackerUncertain: 0,
    candidates: 0,
    candConfirmed: 0,
    candRejected: 0,
    tracks: 0,
    auto: 0,
    uncertain: 0,
    dropped: 0,
    triaged: 0,
    confirmed: 0,
    rejected: 0,
    escalated: 0,
    strong: 0,
    audits: 0,
    disagreements: 0,
    cost: 0,
    without: 0,
    with: 0,
  };
  for (const e of index) {
    const s = e.summary;
    if (!s) continue;
    a.trackerTracks += s.tracker_tracks ?? 0;
    a.trackerAuto += s.tracker_auto ?? 0;
    a.trackerUncertain += s.tracker_uncertain ?? 0;
    a.candidates += s.candidates ?? 0;
    a.candConfirmed += s.candidates_confirmed ?? 0;
    a.candRejected += s.candidates_rejected ?? 0;
    a.tracks += s.tracks;
    a.auto += s.bands.auto_accept ?? 0;
    a.uncertain += s.bands.uncertain ?? 0;
    a.dropped += s.bands.drop ?? 0;
    a.triaged += s.triaged;
    a.confirmed += s.status.confirmed ?? 0;
    a.rejected += s.status.rejected ?? 0;
    a.escalated += s.status.escalated ?? 0;
    a.strong += s.strong_model_calls;
    a.audits += s.audits;
    a.disagreements += s.disagreements;
    a.cost += s.cost_usd;
    a.without += s.alerts_without_agent;
    a.with += s.alerts_with_agent;
  }
  return a;
}
