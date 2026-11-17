import type { Run } from "../lib/data";
import { RISK_COLOR } from "../lib/format";

// The risk rules, drawn as geometry: x = predicted lateral miss at contact (object heights),
// y = time to contact. Each object approaching the camera is a point sliding down toward you.

interface Props {
  run: Run;
  frame: number;
  selected: number | null;
  hovered: number | null;
  onSelect: (id: number | null) => void;
  onHover: (id: number | null) => void;
}

const W = 320, H = 280;
const P = { l: 30, r: 12, t: 14, b: 40 };
const MX = 4, TMAX = 6, TRAIL = 12;
const X = (m: number) => P.l + ((Math.max(-MX, Math.min(MX, m)) + MX) / (2 * MX)) * (W - P.l - P.r);
const Y = (t: number) => P.t + (1 - Math.min(t, TMAX) / TMAX) * (H - P.t - P.b);

export function ContactPlane({ run, frame, selected, hovered, onSelect, onHover }: Props) {
  const rows = run.byFrame[frame] ?? [];
  const pts = rows.filter((r) => r.ttc !== null && r.miss !== null && r.ttc <= TMAX);
  const outside = rows.length - pts.length;
  const x0 = X(-1), x1 = X(1);

  return (
    <div className="plane">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Contact plane: each approaching object by time to contact and predicted miss distance">
        <defs>
          <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--caution)" strokeOpacity="0.22" strokeWidth="1.2" />
          </pattern>
        </defs>
        {/* gridlines */}
        {[0, 1, 2, 3, 4, 5, 6].map((t) => (
          <g key={t}>
            <line x1={P.l} x2={W - P.r} y1={Y(t)} y2={Y(t)} className="grid" />
            <text x={P.l - 6} y={Y(t)} className="axis-label" textAnchor="end" dominantBaseline="middle">{t}s</text>
          </g>
        ))}
        {[-4, -2, 0, 2, 4].map((m) => (
          <text key={m} x={X(m)} y={H - P.b + 14} className="axis-label" textAnchor="middle">{m > 0 ? `+${m}` : m}</text>
        ))}
        {/* rule regions */}
        <rect x={P.l} y={Y(1.5)} width={W - P.l - P.r} height={Y(0) - Y(1.5)} fill="url(#hatch)" />
        <rect x={x0} y={Y(4)} width={x1 - x0} height={Y(0) - Y(4)} fill="var(--caution)" opacity="0.10" />
        <rect x={x0} y={Y(2)} width={x1 - x0} height={Y(0) - Y(2)} fill="var(--danger)" opacity="0.16" />
        <line x1={x0} x2={x0} y1={Y(4)} y2={Y(0)} stroke="var(--caution)" strokeOpacity="0.45" strokeDasharray="2 3" />
        <line x1={x1} x2={x1} y1={Y(4)} y2={Y(0)} stroke="var(--caution)" strokeOpacity="0.45" strokeDasharray="2 3" />
        <text x={(x0 + x1) / 2} y={Y(4) - 5} className="region-label" textAnchor="middle">on course</text>
        <text x={(x0 + x1) / 2} y={Y(1) + 3} className="region-label region-label--danger" textAnchor="middle">danger</text>
        <text x={W - P.r - 4} y={Y(1.5) + 11} className="region-label" textAnchor="end">close pass</text>
        <line x1={P.l} x2={W - P.r} y1={Y(0)} y2={Y(0)} className="baseline" />
        {/* the camera */}
        <path d={`M ${X(0)} ${Y(0) - 1} l -6 9 h 12 z`} fill="var(--ink)" />
        <text x={P.l} y={H - 6} className="axis-title">← passes left</text>
        <text x={W - P.r} y={H - 6} className="axis-title" textAnchor="end">passes right →</text>

        {/* trails */}
        {pts.map((r) => {
          const t = run.trackById.get(r.id)!;
          const hist = t.rows.filter((q) => q.frame <= frame && q.frame > frame - TRAIL && q.ttc !== null && q.miss !== null && q.ttc <= TMAX);
          if (hist.length < 2) return null;
          const d = hist.map((q, i) => `${i ? "L" : "M"}${X(q.miss!).toFixed(1)},${Y(q.ttc!).toFixed(1)}`).join(" ");
          const dim = selected !== null && selected !== r.id;
          return <path key={`t${r.id}`} d={d} fill="none" stroke={RISK_COLOR[r.risk]} strokeWidth={1.5} strokeOpacity={dim ? 0.15 : 0.45} strokeLinecap="round" />;
        })}
        {/* points */}
        {pts.map((r) => {
          const focus = selected === r.id || hovered === r.id;
          const dim = selected !== null && selected !== r.id;
          const cx = X(r.miss!), cy = Y(r.ttc!);
          return (
            <g key={r.id} opacity={dim ? 0.35 : 1} style={{ cursor: "pointer" }}
              onMouseEnter={() => onHover(r.id)} onMouseLeave={() => onHover(null)}
              onClick={() => onSelect(selected === r.id ? null : r.id)}>
              <circle cx={cx} cy={cy} r={12} fill="transparent" />
              <circle cx={cx} cy={cy} r={focus ? 6 : 4.5} fill={r.reliable ? RISK_COLOR[r.risk] : "var(--bg)"}
                stroke={r.reliable ? "var(--bg)" : RISK_COLOR[r.risk]} strokeWidth={r.reliable ? 2 : 1.6} />
              {(focus || r.risk === "danger" || r.risk === "caution") && (
                <text x={cx + 9} y={cy} className="point-label" dominantBaseline="middle">#{r.id}</text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="plane__foot">
        <span><i className="dot dot--solid" /> reliable</span>
        <span><i className="dot dot--hollow" /> flagged</span>
        <span className="muted">{outside > 0 ? `${outside} not approaching or > ${TMAX}s` : " "}</span>
      </div>
    </div>
  );
}
