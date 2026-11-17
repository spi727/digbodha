import { useMemo, useRef, useState } from "react";
import type { Risk, Run, Track } from "../lib/data";
import { RISK_COLOR, cap, fmtS } from "../lib/format";

interface Props {
  run: Run;
  frame: number;
  selected: number | null;
  hovered: number | null;
  onSeek: (f: number) => void;
  onSelect: (id: number | null) => void;
  onHover: (id: number | null) => void;
}

interface Seg { x: number; w: number; risk: Risk }

function segments(t: Track): Seg[] {
  const out: Seg[] = [];
  for (const r of t.rows) {
    const last = out[out.length - 1];
    if (last && last.risk === r.risk && last.x + last.w === r.frame) last.w++;
    else out.push({ x: r.frame, w: 1, risk: r.risk });
  }
  return out;
}

const LANE = 5, GAP = 3;

export function Timeline({ run, frame, selected, hovered, onSeek, onSelect, onHover }: Props) {
  const area = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; track: Track; f: number } | null>(null);
  const lanes = useMemo(() => run.tracks.map((t) => ({ t, segs: segments(t) })), [run]);
  const alertSegs = useMemo(() => {
    const out: Seg[] = [];
    run.worstByFrame.forEach((risk, f) => {
      const last = out[out.length - 1];
      if (last && last.risk === risk) last.w++;
      else out.push({ x: f, w: 1, risk });
    });
    return out;
  }, [run]);
  const n = run.frames;
  const H = lanes.length * (LANE + GAP);
  const seconds = n / run.fps;
  const tickEvery = seconds > 120 ? 30 : seconds > 40 ? 10 : 5;
  const ticks = Array.from({ length: Math.floor(seconds / tickEvery) + 1 }, (_, i) => i * tickEvery);

  const pos = (e: React.MouseEvent) => {
    const rect = area.current!.getBoundingClientRect();
    const f = Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - rect.left) / rect.width) * n)));
    const lane = Math.floor((e.clientY - rect.top) / (LANE + GAP));
    return { f, lane, rect };
  };

  const onMove = (e: React.MouseEvent) => {
    const { f, lane, rect } = pos(e);
    const l = lanes[lane];
    if (e.buttons === 1) onSeek(f);
    if (l && f >= l.t.first - 1 && f <= l.t.last + 1) {
      onHover(l.t.id);
      setTip({ x: e.clientX - rect.left, y: lane * (LANE + GAP), track: l.t, f });
    } else {
      onHover(null);
      setTip(null);
    }
  };

  const scrub = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(Math.floor(((e.clientX - rect.left) / rect.width) * n));
  };

  const tipRow = tip ? tip.track.rows.find((r) => r.frame === tip.f) : undefined;

  return (
    <div className="timeline">
      <div className="timeline__row">
        <div className="timeline__gutter eyebrow">Alert</div>
        <div
          className="timeline__strip"
          onMouseDown={scrub}
          onMouseMove={(e) => e.buttons === 1 && scrub(e)}
          role="slider"
          aria-label="Scrub timeline"
          aria-valuemin={0}
          aria-valuemax={n - 1}
          aria-valuenow={frame}
          tabIndex={0}
        >
          <svg viewBox={`0 0 ${n} 1`} preserveAspectRatio="none">
            {alertSegs.map((s, i) => (
              <rect key={i} x={s.x} y={0} width={s.w} height={1}
                fill={s.risk === "safe" ? "rgba(255,255,255,0.06)" : RISK_COLOR[s.risk]}
                opacity={s.risk === "safe" ? 1 : 0.9} />
            ))}
          </svg>
          <div className="playhead" style={{ left: `${((frame + 0.5) / n) * 100}%` }}><span /></div>
        </div>
      </div>

      <div className="timeline__row timeline__row--lanes">
        <div className="timeline__gutter">
          {lanes.map(({ t }, i) => (
            <button
              key={t.id}
              className={`lane-label mono ${selected === t.id ? "is-selected" : ""} ${hovered === t.id ? "is-hovered" : ""}`}
              style={{ height: LANE + GAP }}
              onClick={() => onSelect(selected === t.id ? null : t.id)}
              onMouseEnter={() => onHover(t.id)}
              onMouseLeave={() => onHover(null)}
              title={`#${t.id} ${t.label}`}
            >
              <span className="lane-label__id">{selected === t.id || hovered === t.id || i % 4 === 0 ? t.id : ""}</span>
            </button>
          ))}
        </div>
        <div
          ref={area}
          className="timeline__lanes"
          style={{ height: H }}
          onMouseMove={onMove}
          onMouseDown={(e) => {
            const { f, lane } = pos(e);
            onSeek(f);
            const l = lanes[lane];
            if (l && f >= l.t.first - 1 && f <= l.t.last + 1) onSelect(l.t.id);
          }}
          onMouseLeave={() => {
            onHover(null);
            setTip(null);
          }}
        >
          <svg viewBox={`0 0 ${n} ${H}`} preserveAspectRatio="none" width="100%" height={H}>
            {lanes.map(({ t, segs }, i) => {
              const y = i * (LANE + GAP);
              const focus = selected === t.id || hovered === t.id;
              const dim = selected !== null && selected !== t.id;
              return (
                <g key={t.id} opacity={dim ? 0.35 : 1}>
                  {focus && <rect x={0} y={y - 1} width={n} height={LANE + 2} fill="rgba(255,255,255,0.06)" />}
                  {segs.map((s, j) => (
                    <rect key={j} x={s.x} y={y} width={s.w} height={LANE}
                      fill={RISK_COLOR[s.risk]} opacity={s.risk === "safe" ? 0.38 : 0.95} />
                  ))}
                </g>
              );
            })}
          </svg>
          <div className="playhead playhead--lanes" style={{ left: `${((frame + 0.5) / n) * 100}%` }} />
          {tip && (
            <div className="tip" style={{ left: tip.x, top: tip.y }}>
              <div className="tip__head">
                <span className="swatch" style={{ background: RISK_COLOR[tipRow?.risk ?? tip.track.worst] }} />
                <b>#{tip.track.id}</b> {tip.track.label}
              </div>
              <div className="tip__body mono">
                {tipRow ? (
                  <>
                    <span>{(tip.f / run.fps).toFixed(1)}s</span>
                    <span>{cap(tipRow.risk)}</span>
                    <span>TTC {fmtS(tipRow.ttc)}</span>
                  </>
                ) : (
                  <span>not in view at {(tip.f / run.fps).toFixed(1)}s</span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="timeline__row">
        <div className="timeline__gutter" />
        <div className="timeline__axis mono">
          {ticks.map((s) => (
            <span key={s} style={{ left: `${((s * run.fps) / n) * 100}%` }}>{s}s</span>
          ))}
        </div>
      </div>
    </div>
  );
}
