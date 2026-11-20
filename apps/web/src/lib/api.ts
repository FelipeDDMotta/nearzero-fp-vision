import type { ClipResult, IndexEntry, Triage } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
export const REPO_URL = process.env.NEXT_PUBLIC_REPO_URL || "https://github.com/FelipeDDMotta/nearzero-fp-vision";

export async function fetchIndex(): Promise<IndexEntry[]> {
  const r = await fetch("/results/index.json");
  return r.json();
}

export async function fetchClip(id: string): Promise<ClipResult> {
  const r = await fetch(`/results/${id}.json`);
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      msg = (await r.json()).detail || msg;
    } catch {}
    throw new Error(msg);
  }
  return r.json();
}

export function liveTriage(clip_id: string, track_id: number, engine: "claude" | "open"): Promise<Triage> {
  return post<Triage>("/triage", { clip_id, track_id, engine });
}

export interface SearchHit {
  clip_id: string;
  track_id: number;
  cls: string;
  crop: string;
  t0: number;
  t1: number;
  attributes: Record<string, unknown>;
  matched: string[];
  score: number;
  status: string;
}

export function search(q: string) {
  return post<{ query: string; filter: Record<string, unknown>; count: number; results: SearchHit[] }>("/search", { q });
}
