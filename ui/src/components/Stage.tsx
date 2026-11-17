import { useEffect, useRef, useState } from "react";
import type { Row, Run } from "../lib/data";
import type { Player } from "../lib/player";
import { RISK_HEX, cap, fmtTime } from "../lib/format";

interface Props {
  run: Run;
  player: Player;
  selected?: number | null;
  hovered?: number | null;
  onSelect?: (id: number | null) => void;
  onHover?: (id: number | null) => void;
  zones?: boolean;
  compact?: boolean;
  loop?: boolean;
  hud?: boolean;
}

const MONO = '500 11px "Geist Mono", ui-monospace, monospace';

export function Stage({ run, player, selected = null, hovered = null, onSelect, onHover, zones = true, compact = false, loop, hud = true }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const rows = run.byFrame[player.frame] ?? [];
  const worst = run.worstByFrame[player.frame] ?? "safe";

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(size.w * dpr);
    c.height = Math.round(size.h * dpr);
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size.w, size.h);
    const sx = size.w / run.width, sy = size.h / run.height;

    if (zones && !compact) drawZones(g, size.w, size.h);

    const focus = selected ?? null;
    const ordered = [...rows].sort((a, b) => (a.id === focus ? 1 : 0) - (b.id === focus ? 1 : 0));
    for (const r of ordered) {
      const dim = focus !== null && r.id !== focus;
      drawObject(g, r, sx, sy, { dim, selected: r.id === focus, hovered: r.id === hovered });
    }
    drawLabels(g, rows, sx, sy, { focus, hovered, compact, w: size.w, h: size.h });
  }, [rows, size, run.width, run.height, selected, hovered, zones, compact]);

  // frame-folder runs: warm the cache for the next frames so playback doesn't flicker
  useEffect(() => {
    if (run.source.kind !== "frames") return;
    for (let k = 1; k <= 8 && player.frame + k < run.frames; k++) new Image().src = run.source.urlFor(player.frame + k);
  }, [run, player.frame]);

  const hit = (e: React.MouseEvent): number | null => {
    const rect = wrap.current!.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * run.width;
    const y = ((e.clientY - rect.top) / rect.height) * run.height;
    let best: Row | null = null, area = Infinity;
    for (const r of rows) {
      const [x0, y0, x1, y1] = r.box;
      const pad = 6;
      if (x >= x0 - pad && x <= x1 + pad && y >= y0 - pad && y <= y1 + pad) {
        const a = (x1 - x0) * (y1 - y0);
        if (a < area) (best = r), (area = a);
      }
    }
    return best?.id ?? null;
  };

  const src = run.source;
  const t = player.frame / run.fps;

  return (
    <div
      ref={wrap}
      className={`stage ${worst === "danger" ? "stage--danger" : ""} ${compact ? "stage--compact" : ""}`}
      style={{ aspectRatio: `${run.width} / ${run.height}` }}
      onMouseMove={onHover ? (e) => onHover(hit(e)) : undefined}
      onMouseLeave={onHover ? () => onHover(null) : undefined}
      onClick={onSelect ? (e) => onSelect(hit(e)) : undefined}
      data-hover={hovered !== null ? "" : undefined}
    >
      {src.kind === "video" && (
        <video ref={player.videoRef} src={src.url} muted playsInline preload="auto" loop={loop} className="stage__media" />
      )}
      {src.kind === "frames" && <img src={src.urlFor(player.frame)} alt="" className="stage__media" draggable={false} />}
      {src.kind === "none" && (
        <div className="stage__empty">
          <span>No source pixels for this run. Boxes are drawn on a blank canvas.</span>
        </div>
      )}
      <canvas ref={canvas} className="stage__overlay" />
      {!player.ready && <div className="stage__loading"><span className="spinner" /> Loading video…</div>}
      {hud && (
        <>
          <div className="hud hud--tl mono">
            <span>T+{fmtTime(t)}</span>
            <span className="hud__sep" />
            <span>F{String(player.frame).padStart(4, "0")}</span>
            {!compact && (
              <>
                <span className="hud__sep" />
                <span>{rows.length} tracked</span>
              </>
            )}
          </div>
          <div className={`hud hud--tr alert alert--${worst}`}>
            <span className="alert__dot" />
            <span>{worst === "safe" ? "All clear" : cap(worst)}</span>
          </div>
        </>
      )}
    </div>
  );
}

function drawZones(g: CanvasRenderingContext2D, w: number, h: number) {
  g.save();
  g.strokeStyle = "rgba(255,255,255,0.14)";
  g.setLineDash([2, 6]);
  g.lineWidth = 1;
  for (const x of [w / 3, (2 * w) / 3]) {
    g.beginPath();
    g.moveTo(Math.round(x) + 0.5, 0);
    g.lineTo(Math.round(x) + 0.5, h);
    g.stroke();
  }
  g.setLineDash([]);
  // heading marker: the image centre is the assumed direction of travel
  g.strokeStyle = "rgba(255,255,255,0.32)";
  g.beginPath();
  g.moveTo(w / 2, h - 12);
  g.lineTo(w / 2, h - 4);
  g.moveTo(w / 2 - 5, h - 8);
  g.lineTo(w / 2 + 5, h - 8);
  g.stroke();
  g.font = '500 9.5px "Geist Mono", ui-monospace, monospace';
  g.fillStyle = "rgba(255,255,255,0.38)";
  g.textBaseline = "bottom";
  g.textAlign = "left";
  g.fillText("LEFT", 8, h - 6);
  g.textAlign = "center";
  g.fillText("CENTRE", w / 2, h - 16);
  g.textAlign = "right";
  g.fillText("RIGHT", w - 8, h - 6);
  g.restore();
}

function drawObject(
  g: CanvasRenderingContext2D, r: Row, sx: number, sy: number,
  o: { dim: boolean; selected: boolean; hovered: boolean },
) {
  const [bx0, by0, bx1, by1] = r.box;
  const x0 = bx0 * sx, y0 = by0 * sy, x1 = bx1 * sx, y1 = by1 * sy;
  const w = x1 - x0, h = y1 - y0;
  const col = RISK_HEX[r.risk];
  const alert = r.risk === "danger" || r.risk === "caution";
  g.save();
  g.globalAlpha = o.dim ? 0.28 : 1;

  // tint
  g.fillStyle = col;
  g.globalAlpha *= o.selected ? 0.16 : alert ? 0.12 : 0.05;
  g.fillRect(x0, y0, w, h);
  g.globalAlpha = o.dim ? 0.28 : 1;

  // frame: corner brackets (full outline when focused or uncertain)
  g.strokeStyle = col;
  g.lineWidth = o.selected ? 2 : alert ? 1.6 : 1.2;
  if (r.risk === "safe" && !o.selected && !o.hovered) g.globalAlpha *= 0.75;
  if (r.risk === "uncertain") {
    g.setLineDash([4, 3]);
    g.strokeRect(x0 + 0.5, y0 + 0.5, w, h);
    g.setLineDash([]);
  } else if (o.selected || o.hovered) {
    g.strokeRect(x0 + 0.5, y0 + 0.5, w, h);
  } else {
    const k = Math.max(4, Math.min(14, 0.28 * Math.min(w, h)));
    g.beginPath();
    g.moveTo(x0, y0 + k); g.lineTo(x0, y0); g.lineTo(x0 + k, y0);
    g.moveTo(x1 - k, y0); g.lineTo(x1, y0); g.lineTo(x1, y0 + k);
    g.moveTo(x1, y1 - k); g.lineTo(x1, y1); g.lineTo(x1 - k, y1);
    g.moveTo(x0 + k, y1); g.lineTo(x0, y1); g.lineTo(x0, y1 - k);
    g.stroke();
  }
  if (o.hovered && !o.selected) {
    g.strokeStyle = "rgba(255,255,255,0.9)";
    g.lineWidth = 1;
    g.strokeRect(x0 - 2.5, y0 - 2.5, w + 5, h + 5);
  }
  g.restore();
}

type Rect = { x: number; y: number; w: number; h: number };
const overlaps = (a: Rect, b: Rect) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
const PRIORITY: Record<Row["risk"], number> = { danger: 3, caution: 2, uncertain: 1, safe: 0 };

// Labels are placed greedily, most important first, and skipped rather than stacked when they
// would collide: in a crowded street the alerts stay readable and quiet objects keep just a box.
function drawLabels(
  g: CanvasRenderingContext2D, rows: Row[], sx: number, sy: number,
  o: { focus: number | null; hovered: number | null; compact: boolean; w: number; h: number },
) {
  const rank = (r: Row) =>
    (r.id === o.focus ? 100 : 0) + (r.id === o.hovered ? 50 : 0) + PRIORITY[r.risk] * 10 +
    Math.min(9, ((r.box[3] - r.box[1]) * sy) / 20);
  const order = [...rows].sort((a, b) => rank(b) - rank(a));
  const placed: Rect[] = [];
  g.save();
  g.font = MONO;
  g.textBaseline = "middle";
  for (const r of order) {
    const alert = r.risk === "danger" || r.risk === "caution";
    const focus = r.id === o.focus || r.id === o.hovered;
    if (o.compact && !alert && !focus) continue;
    const x0 = r.box[0] * sx, y0 = r.box[1] * sy, y1 = r.box[3] * sy;
    const ttc = r.ttc !== null && r.ttc < 10 ? ` ${r.ttc.toFixed(1)}s` : "";
    const full = o.compact ? `${r.label}${ttc}` : `#${r.id} ${r.label}${ttc}${r.reliable ? "" : " ?"}`;
    const short = `#${r.id}${ttc}`;
    const ph = 17;
    let spot: Rect | null = null, text = full;
    for (const t of [full, short]) {
      const pw = g.measureText(t).width + 14;
      const px = Math.min(Math.max(0, x0), o.w - pw);
      const cands = [y0 - ph - 3, y1 + 3, y0 + 3].filter((y) => y >= 2 && y + ph <= o.h - 2);
      for (const py of cands) {
        const rect = { x: px, y: py, w: pw, h: ph };
        if (!placed.some((p) => overlaps(p, rect))) {
          spot = rect;
          text = t;
          break;
        }
      }
      if (spot) break;
    }
    if (!spot) {
      if (!focus) continue;
      const pw = g.measureText(full).width + 14;
      spot = { x: Math.min(Math.max(0, x0), o.w - pw), y: Math.max(2, y0 - ph - 3), w: pw, h: ph };
      text = full;
    }
    placed.push({ x: spot.x - 2, y: spot.y - 1, w: spot.w + 4, h: spot.h + 2 });
    g.globalAlpha = o.focus !== null && r.id !== o.focus ? 0.4 : 1;
    g.fillStyle = "rgba(7,9,12,0.84)";
    roundRect(g, spot.x, spot.y, spot.w, spot.h, 4);
    g.fill();
    g.fillStyle = RISK_HEX[r.risk];
    g.fillRect(spot.x, spot.y + 3, 2, ph - 6);
    g.fillStyle = "rgba(236,239,243,0.95)";
    g.fillText(text, spot.x + 8, spot.y + ph / 2 + 0.5);
  }
  g.restore();
}

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}
