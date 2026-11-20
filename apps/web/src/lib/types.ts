export type Verdict = "true_positive" | "false_positive" | "uncertain";
export type Status = "confirmed" | "rejected" | "escalated";
export type Band = "auto_accept" | "uncertain" | "drop";

export interface Attributes {
  upper_color?: string;
  lower_color?: string;
  accessories?: string[];
  vehicle_type?: string;
  vehicle_color?: string;
  pose?: string;
}

export interface Step {
  model: string;
  verdict: Verdict;
  corrected_class?: string;
  confidence: number;
  reason: string;
  attributes?: Attributes;
  tools_used: { tool: string; why: string }[];
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  cost_usd: number;
}

export interface Audit {
  model: string;
  verdict: Verdict;
  confidence: number;
  reason: string;
  latency_ms: number;
  cost_usd: number;
}

export interface Triage {
  status: Status;
  final: {
    verdict: Verdict;
    corrected_class?: string;
    confidence: number;
    reason: string;
    attributes?: Attributes;
    model: string;
  };
  steps: Step[];
  audit: Audit | null;
  disagreement: boolean;
  cost_usd: number;
  latency_ms: number;
  prompt_version: string;
}

export interface Track {
  id: number;
  cls: string;
  score: number;
  conf_max: number;
  conf_median: number;
  n: number;
  t0: number;
  t1: number;
  band: Band;
  boxes: [number, number, number, number, number, number][];
  crops: { best: string; context: string; frames: string[] };
  triage?: Triage;
  describe?: { attributes: Attributes };
  source?: "tracker" | "low_conf_mining";
}

export interface Summary {
  tracker_tracks?: number;
  tracker_auto?: number;
  tracker_uncertain?: number;
  candidates?: number;
  candidates_confirmed?: number;
  candidates_rejected?: number;
  tracks: number;
  bands: Partial<Record<Band, number>>;
  triaged: number;
  status: Partial<Record<Status, number>>;
  auto_resolved_pct: number;
  escalated_pct: number;
  alerts_without_agent: number;
  alerts_with_agent: number;
  strong_model_calls: number;
  audits: number;
  disagreements: number;
  cost_usd: number;
  cost_per_event_usd: number;
  latency_ms_p50: number;
}

export interface ClipMeta {
  id: string;
  title: string;
  scene: string;
  attribution: string;
  video: string;
  degrade?: string;
}

export interface ClipResult {
  clip: ClipMeta;
  fps: number;
  width: number;
  height: number;
  duration: number;
  tracks: Track[];
  summary?: Summary;
}

export interface IndexEntry {
  clip: ClipMeta;
  duration: number;
  summary?: Summary;
}

export type DisplayStatus = "auto" | Status | "pending" | "dropped";

export function displayStatus(t: Track): DisplayStatus {
  if (t.band === "auto_accept") return "auto";
  if (t.band === "drop") return "dropped";
  return t.triage?.status ?? "pending";
}

export const STATUS_STYLE: Record<DisplayStatus, { label: string; color: string; dash?: number[] }> = {
  auto: { label: "Auto-accepted", color: "#38bdf8" },
  confirmed: { label: "Agent confirmed", color: "#22c55e" },
  rejected: { label: "Agent rejected (FP)", color: "#ef4444", dash: [6, 4] },
  escalated: { label: "Escalated to human", color: "#f59e0b" },
  pending: { label: "Awaiting agent", color: "#a78bfa", dash: [2, 3] },
  dropped: { label: "Below threshold", color: "#64748b", dash: [2, 4] },
};

export function attrSummary(a?: Attributes): string {
  if (!a) return "";
  const parts: string[] = [];
  const ok = (v?: string) => v && v !== "unknown" && v !== "n/a";
  if (ok(a.upper_color)) parts.push(`${a.upper_color} top`);
  if (ok(a.lower_color)) parts.push(`${a.lower_color} bottom`);
  if (ok(a.vehicle_color) || ok(a.vehicle_type))
    parts.push([ok(a.vehicle_color) ? a.vehicle_color : "", ok(a.vehicle_type) ? a.vehicle_type : ""].join(" ").trim());
  (a.accessories || []).filter((x) => x !== "none").forEach((x) => parts.push(x.replace(/_/g, " ")));
  if (ok(a.pose)) parts.push(a.pose!);
  return parts.join(" · ");
}
