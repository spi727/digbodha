// Loading and shaping DigBodha outputs (frames.csv + run meta) for the UI.

export type Risk = "safe" | "uncertain" | "caution" | "danger";
export const RISKS: Risk[] = ["safe", "uncertain", "caution", "danger"];
export const RISK_ORDER: Record<Risk, number> = { safe: 0, uncertain: 1, caution: 2, danger: 3 };

export interface Row {
  frame: number;
  time: number;
  id: number;
  label: string;
  cx: number | null;
  cy: number | null;
  box: [number, number, number, number];
  size: number | null;
  zone: string;
  rawFlow: [number, number] | null;
  ownFlow: [number, number] | null;
  motion: string;
  expansion: number | null;
  ttc: number | null;
  ttcLo: number | null;
  ttcHi: number | null;
  miss: number | null;
  onCourse: boolean;
  risk: Risk;
  reliable: boolean;
  reasons: string[];
}

export interface Track {
  id: number;
  label: string;
  first: number;
  last: number;
  seen: number;
  rows: Row[];
  worst: Risk;
  minTtc: number | null; // over reliable estimates, as analyze.py reports it
  minTtcFrame: number | null;
  everOnCourse: boolean;
  reliableFrac: number;
  mainMotion: string;
  mainZone: string;
  counts: Record<Risk, number>;
}

export interface RunMeta {
  name: string;
  title: string;
  subtitle?: string;
  prompts: string[];
  fps: number;
  frames: number;
  width: number;
  height: number;
  source: { kind: "video"; url: string } | { kind: "frames"; urlFor: (i: number) => string } | { kind: "none" };
  demo?: boolean;
  seconds?: number | null;
}

export interface Run extends RunMeta {
  rows: Row[];
  byFrame: Row[][];
  tracks: Track[];
  trackById: Map<number, Track>;
  worstByFrame: Risk[];
}

// ------------------------------------------------------------------ CSV

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.replace(/\r/g, "").split("\n").filter(Boolean);
  const head = lines[0].split(",");
  return lines.slice(1).map((l) => {
    const cells = l.split(","); // frames.csv has no quoted fields (reasons are ';'-joined)
    const o: Record<string, string> = {};
    head.forEach((h, i) => (o[h] = cells[i] ?? ""));
    return o;
  });
}

const num = (s: string | undefined): number | null => {
  if (s === undefined || s === "" || s === "None" || s === "nan") return null;
  const v = Number(s);
  return Number.isFinite(v) ? v : null;
};
const bool = (s: string | undefined) => s === "True" || s === "true" || s === "1";
const pair = (a: string, b: string): [number, number] | null => {
  const x = num(a), y = num(b);
  return x === null || y === null ? null : [x, y];
};

export function parseRows(text: string): Row[] {
  return parseCSV(text).map((r) => ({
    frame: Number(r.frame),
    time: Number(r.time_s),
    id: Number(r.id),
    label: r.label,
    cx: num(r.cx),
    cy: num(r.cy),
    box: [Number(r.bx0), Number(r.by0), Number(r.bx1), Number(r.by1)],
    size: num(r.size_px),
    zone: r.zone,
    rawFlow: pair(r.raw_dx, r.raw_dy),
    ownFlow: pair(r.own_dx, r.own_dy),
    motion: r.motion || "unknown",
    expansion: num(r.expansion_per_s),
    ttc: num(r.ttc_s),
    ttcLo: num(r.ttc_lo),
    ttcHi: num(r.ttc_hi),
    miss: num(r.miss_widths),
    onCourse: bool(r.on_course),
    risk: (RISKS.includes(r.risk as Risk) ? r.risk : "safe") as Risk,
    reliable: bool(r.reliable),
    reasons: r.reasons ? r.reasons.split(";").filter(Boolean) : [],
  }));
}

function mode(xs: string[]): string {
  const c = new Map<string, number>();
  xs.forEach((x) => c.set(x, (c.get(x) ?? 0) + 1));
  let best = "", n = -1;
  c.forEach((v, k) => v > n && ((best = k), (n = v)));
  return best;
}

export function buildRun(meta: RunMeta, rows: Row[]): Run {
  const frames = Math.max(meta.frames, 1 + rows.reduce((m, r) => Math.max(m, r.frame), -1));
  const byFrame: Row[][] = Array.from({ length: frames }, () => []);
  rows.forEach((r) => byFrame[r.frame]?.push(r));
  byFrame.forEach((rs) => rs.sort((a, b) => RISK_ORDER[b.risk] - RISK_ORDER[a.risk] || (a.ttc ?? 99) - (b.ttc ?? 99)));

  const groups = new Map<number, Row[]>();
  rows.forEach((r) => (groups.get(r.id) ?? groups.set(r.id, []).get(r.id)!).push(r));
  const tracks: Track[] = [...groups.entries()].map(([id, rs]) => {
    rs.sort((a, b) => a.frame - b.frame);
    const counts: Record<Risk, number> = { safe: 0, uncertain: 0, caution: 0, danger: 0 };
    let worst: Risk = "safe", minTtc: number | null = null, minTtcFrame: number | null = null;
    rs.forEach((r) => {
      counts[r.risk]++;
      if (RISK_ORDER[r.risk] > RISK_ORDER[worst]) worst = r.risk;
      if (r.ttc !== null && r.reliable && (minTtc === null || r.ttc < minTtc)) (minTtc = r.ttc), (minTtcFrame = r.frame);
    });
    return {
      id, label: rs[0].label, first: rs[0].frame, last: rs[rs.length - 1].frame, seen: rs.length, rows: rs,
      worst, minTtc, minTtcFrame, counts,
      everOnCourse: rs.some((r) => r.onCourse),
      reliableFrac: rs.filter((r) => r.reliable).length / rs.length,
      mainMotion: mode(rs.map((r) => r.motion)),
      mainZone: mode(rs.map((r) => r.zone)),
    };
  });
  tracks.sort((a, b) => a.first - b.first || a.id - b.id);

  const worstByFrame = byFrame.map((rs) =>
    rs.reduce<Risk>((w, r) => (RISK_ORDER[r.risk] > RISK_ORDER[w] ? r.risk : w), "safe"),
  );
  return { ...meta, frames, rows, byFrame, tracks, trackById: new Map(tracks.map((t) => [t.id, t])), worstByFrame };
}

// ------------------------------------------------------------------ sources

export const DEMO_NAME = "kitti-0007";

export const DEMO_META: RunMeta = {
  name: DEMO_NAME,
  title: "KITTI 0007",
  subtitle: "Held-out sequence · urban drive · 300 frames at 10 FPS",
  prompts: ["vehicle", "person"],
  fps: 10,
  frames: 300,
  width: 1242,
  height: 375,
  source: { kind: "video", url: "/demo/kitti_0007.mp4" },
  demo: true,
};

export interface RunSummary {
  name: string;
  title: string;
  prompts: string[];
  frames: number | null;
  demo?: boolean;
}

const splitPrompts = (p: string | undefined) => (p ?? "").split(",").map((s) => s.trim()).filter(Boolean);

// Only a positive answer is cached, so starting the server later works without a full reload.
let serverUp = false;
export async function hasServer(): Promise<boolean> {
  if (serverUp) return true;
  serverUp = (await health()) !== null;
  return serverUp;
}

export async function health(): Promise<{ sam3: boolean; cv2: boolean; python: string } | null> {
  try {
    const r = await fetch("/api/health");
    if (!r.ok || !(r.headers.get("content-type") ?? "").includes("json")) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export async function listRuns(): Promise<RunSummary[]> {
  const demo: RunSummary = { name: DEMO_NAME, title: DEMO_META.title, prompts: DEMO_META.prompts, frames: 300, demo: true };
  if (!(await hasServer())) return [demo];
  try {
    const runs: { name: string; prompts: string; frames: number | null }[] = await (await fetch("/api/runs")).json();
    return [demo, ...runs.map((r) => ({ name: r.name, title: r.name, prompts: splitPrompts(r.prompts), frames: r.frames }))];
  } catch {
    return [demo];
  }
}

const cache = new Map<string, Promise<Run>>();

export function loadRun(name: string): Promise<Run> {
  if (!cache.has(name)) {
    const p = (async () => {
      if (name === DEMO_NAME) {
        const text = await (await fetch("/demo/frames.csv")).text();
        return buildRun(DEMO_META, parseRows(text));
      }
      const base = `/api/runs/${encodeURIComponent(name)}`;
      const metaRes = await fetch(`${base}/meta`);
      if (!metaRes.ok) throw new Error((await metaRes.json().catch(() => null))?.error ?? `Run “${name}” not found`);
      const m = await metaRes.json();
      const text = await (await fetch(`${base}/frames.csv`)).text();
      const meta: RunMeta = {
        name, title: name, prompts: splitPrompts(m.prompts), fps: m.fps, frames: m.frames,
        width: m.width, height: m.height, seconds: m.seconds_total,
        subtitle: `${m.frames} frames at ${m.fps} FPS`,
        source:
          m.source === "video" ? { kind: "video", url: `${base}/video` }
          : m.source === "frames" ? { kind: "frames", urlFor: (i: number) => `${base}/frame/${i}` }
          : { kind: "none" },
      };
      return buildRun(meta, parseRows(text));
    })();
    p.catch(() => cache.delete(name));
    cache.set(name, p);
  }
  return cache.get(name)!;
}

// ------------------------------------------------------------------ evaluation

export interface Metrics {
  frames: number;
  gt_objects_within_depth: number;
  gt_matched_pct: number;
  ttc_all: TtcStats;
  ttc_reliable: TtcStats;
  ttc_unreliable: TtcStats;
  ttc_by_depth: Record<string, TtcStats>;
  approach: { tp: number; fp: number; fn: number; precision: number; recall: number };
  risk_confusion_gt_rows_pred_cols: Record<string, Record<string, number>>;
  alerts: { tp: number; fp: number; fn: number; fn_flagged_uncertain: number; precision: number; recall: number };
}
export interface TtcStats { n: number; median_abs_s: number; mean_abs_s: number; median_rel_pct: number; within_20pct: number }

export const SEQUENCES = [
  { seq: "0013", split: "tuning" },
  { seq: "0001", split: "tuning" },
  { seq: "0020", split: "tuning" },
  { seq: "0019", split: "held-out" },
  { seq: "0007", split: "held-out" },
] as const;

export async function loadMetrics(): Promise<Record<string, Metrics>> {
  const out: Record<string, Metrics> = {};
  await Promise.all(SEQUENCES.map(async ({ seq }) => {
    out[seq] = await (await fetch(`/demo/eval_${seq}.json`)).json();
  }));
  return out;
}
