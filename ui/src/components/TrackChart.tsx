import { useRef, useState } from "react";
import type { Row, Track } from "../lib/data";
import { RISK_COLOR, cap, fmtNum, fmtS } from "../lib/format";

interface Props {
  track: Track;
  fps: number;
  frame: number;
  kind: "ttc" | "miss";
  onSeek: (f: number) => void;
}

const W = 520, H = 170;
const P = { l: 34, r: 64, t: 10, b: 22 };

export function TrackChart({ track, fps, frame, kind, onSeek }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const f0 = track.first, f1 = Math.max(track.last, track.first + 1);
  const [lo, hi] = kind === "ttc" ? [0, 10] : [-4, 4];
  const X = (f: number) => P.l + ((f - f0) / (f1 - f0)) * (W - P.l - P.r);
  const Y = (v: number) => P.t + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - P.t - P.b);
  const val = (r: Row) => (kind === "ttc" ? r.ttc : r.miss);

  // split into contiguous runs where a value exists
  const runs: Row[][] = [];
  let cur: Row[] = [];
  track.rows.forEach((r, i) => {
    const prev = track.rows[i - 1];
    if (val(r) === null || (prev && r.frame !== prev.frame + 1)) {
      if (cur.length) runs.push(cur);
      cur = [];
    }
    if (val(r) !== null) cur.push(r);
  });
  if (cur.length) runs.push(cur);

  const line = (rs: Row[]) => rs.map((r, i) => `${i ? "L" : "M"}${X(r.frame).toFixed(1)},${Y(val(r)!).toFixed(1)}`).join("");
  const band = (rs: Row[]) => {
    const ok = rs.filter((r) => r.ttcLo !== null && r.ttcHi !== null);
    if (ok.length < 2) return "";
    const top = ok.map((r, i) => `${i ? "L" : "M"}${X(r.frame).toFixed(1)},${Y(r.ttcHi!).toFixed(1)}`).join("");
    const bot = [...ok].reverse().map((r) => `L${X(r.frame).toFixed(1)},${Y(r.ttcLo!).toFixed(1)}`).join("");
    return `${top}${bot}Z`;
  };

  const fromEvent = (e: React.MouseEvent) => {
    const rect = ref.current!.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    return Math.round(f0 + ((x - P.l) / (W - P.l - P.r)) * (f1 - f0));
  };

  const hf = hover !== null ? Math.max(f0, Math.min(track.last, hover)) : null;
  const hr = hf !== null ? track.rows.find((r) => r.frame === hf) : undefined;
  const inLife = frame >= track.first && frame <= track.last;
  const yTicks = kind === "ttc" ? [0, 2, 4, 6, 8, 10] : [-4, -2, 0, 2, 4];
  const tTicks = niceTicks(f0 / fps, f1 / fps);

  return (
    <div className="chart">
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        onMouseMove={(e) => setHover(fromEvent(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => onSeek(Math.max(track.first, Math.min(track.last, fromEvent(e))))}
        role="img"
        aria-label={kind === "ttc" ? `Time to contact for object ${track.id}` : `Predicted miss distance for object ${track.id}`}
      >
        {yTicks.map((v) => (
          <g key={v}>
            <line x1={P.l} x2={W - P.r} y1={Y(v)} y2={Y(v)} className={v === lo ? "baseline" : "grid"} />
            <text x={P.l - 6} y={Y(v)} className="axis-label" textAnchor="end" dominantBaseline="middle">
              {kind === "ttc" ? `${v}s` : v > 0 ? `+${v}` : v}
            </text>
          </g>
        ))}
        {tTicks.map((s) => (
          <text key={s} x={X(s * fps)} y={H - 6} className="axis-label" textAnchor="middle">{s}s</text>
        ))}

        {kind === "ttc" ? (
          <>
            <rect x={P.l} y={Y(2)} width={W - P.l - P.r} height={Y(0) - Y(2)} fill="var(--danger)" opacity="0.07" />
            <rect x={P.l} y={Y(4)} width={W - P.l - P.r} height={Y(2) - Y(4)} fill="var(--caution)" opacity="0.05" />
            <line x1={P.l} x2={W - P.r} y1={Y(2)} y2={Y(2)} stroke="var(--danger)" strokeOpacity="0.55" strokeDasharray="3 3" />
            <line x1={P.l} x2={W - P.r} y1={Y(4)} y2={Y(4)} stroke="var(--caution)" strokeOpacity="0.5" strokeDasharray="3 3" />
            <text x={W - P.r + 6} y={Y(2)} className="axis-label" dominantBaseline="middle">danger</text>
            <text x={W - P.r + 6} y={Y(4)} className="axis-label" dominantBaseline="middle">caution</text>
            {runs.map((rs, i) => <path key={`b${i}`} d={band(rs)} fill="var(--ink)" opacity="0.08" />)}
          </>
        ) : (
          <>
            <rect x={P.l} y={Y(1)} width={W - P.l - P.r} height={Y(-1) - Y(1)} fill="var(--caution)" opacity="0.08" />
            <line x1={P.l} x2={W - P.r} y1={Y(1)} y2={Y(1)} stroke="var(--caution)" strokeOpacity="0.45" strokeDasharray="3 3" />
            <line x1={P.l} x2={W - P.r} y1={Y(-1)} y2={Y(-1)} stroke="var(--caution)" strokeOpacity="0.45" strokeDasharray="3 3" />
            <text x={W - P.r + 6} y={Y(0)} className="axis-label" dominantBaseline="middle">on course</text>
          </>
        )}

        {runs.map((rs, i) => (
          <path key={i} d={line(rs)} fill="none" stroke="var(--ink-2)" strokeWidth={1.5} strokeLinejoin="round" />
        ))}
        {track.rows.filter((r) => val(r) !== null).map((r) => (
          <circle key={r.frame} cx={X(r.frame)} cy={Y(val(r)!)} r={2.6}
            fill={r.reliable ? RISK_COLOR[r.risk] : "var(--panel)"}
            stroke={r.reliable ? "none" : RISK_COLOR[r.risk]} strokeWidth={1.2} />
        ))}

        {inLife && <line x1={X(frame)} x2={X(frame)} y1={P.t} y2={H - P.b} className="cursor" />}
        {hf !== null && <line x1={X(hf)} x2={X(hf)} y1={P.t} y2={H - P.b} className="crosshair" />}
      </svg>
      {hf !== null && (
        <div className="tip tip--chart" style={{ left: `${(X(hf) / W) * 100}%` }}>
          <div className="tip__head">
            <span className="swatch" style={{ background: RISK_COLOR[hr?.risk ?? "safe"] }} />
            {(hf / fps).toFixed(1)}s · {hr ? cap(hr.risk) : "no data"}
          </div>
          {hr && (
            <div className="tip__body mono">
              {kind === "ttc" ? (
                <span>TTC {fmtS(hr.ttc)} {hr.ttcLo !== null && hr.ttcHi !== null ? `[${hr.ttcLo.toFixed(1)}–${hr.ttcHi.toFixed(1)}]` : ""}</span>
              ) : (
                <span>miss {fmtNum(hr.miss, 2)} h{hr.onCourse ? " · on course" : ""}</span>
              )}
              {!hr.reliable && <span className="muted">flagged</span>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function niceTicks(a: number, b: number): number[] {
  const span = b - a;
  const step = span > 20 ? 5 : span > 8 ? 2 : span > 3 ? 1 : 0.5;
  const out: number[] = [];
  for (let s = Math.ceil(a / step) * step; s <= b + 1e-9; s += step) out.push(+s.toFixed(1));
  return out;
}
