import { useEffect, useRef, useState } from "react";

// Interactive version of the notebook's synthetic check: a vehicle 30 m away closing at
// 10 m/s. Its image height s = fH/Z, so 1/s falls on a straight line that hits zero at contact.
const F = 700, HGT = 1.5, Z0 = 30, FPS = 10, T_END = 2.6, WINDOW = 0.8;

function fit(ts: number[], ys: number[]) {
  const n = ts.length, tm = ts.reduce((a, b) => a + b) / n, ym = ys.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0;
  ts.forEach((t, i) => ((sxy += (t - tm) * (ys[i] - ym)), (sxx += (t - tm) ** 2)));
  const b = sxy / sxx;
  return { a: ym - b * tm, b };
}

export function Looming() {
  const [speed, setSpeed] = useState(10);
  const [t, setT] = useState(1.2);
  const [playing, setPlaying] = useState(true);
  const last = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = (now: number) => {
      if (last.current !== null) {
        setT((v) => {
          const nv = v + (now - last.current!) / 1000 / 1.6;
          return nv > tEnd(speed) ? 0.5 : nv;
        });
      }
      last.current = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      last.current = null;
    };
  }, [playing, speed]);

  const Z = (time: number) => Z0 - speed * time;
  const s = (time: number) => (F * HGT) / Z(time);
  const tc = Math.min(t, tEnd(speed));
  const samples: number[] = [];
  for (let k = 0; k / FPS <= tc + 1e-9; k++) samples.push(k / FPS);
  const win = samples.filter((x) => x > tc - WINDOW - 1e-9);
  const { a, b } = win.length >= 5 ? fit(win, win.map((x) => 1 / s(x))) : { a: 0, b: 0 };
  const uNow = a + b * tc;
  const est = win.length >= 5 && b < 0 ? uNow / -b : null;
  const truth = Z(tc) / speed;

  // plot geometry
  const W = 420, H = 230, P = { l: 40, r: 16, t: 16, b: 28 };
  const tMax = Z0 / speed;
  const uMax = 1 / s(0);
  const X = (x: number) => P.l + (x / tMax) * (W - P.l - P.r);
  const Y = (u: number) => P.t + (1 - u / uMax) * (H - P.t - P.b);
  const imgH = Math.min(150, s(tc) * 0.55);

  return (
    <div className="looming">
      <div className="looming__scene" aria-hidden="true">
        <div className="looming__road" />
        <div className="looming__car" style={{ height: imgH, width: imgH * 1.45 }}>
          <span className="looming__dim mono">s = {s(tc).toFixed(0)} px</span>
        </div>
        <div className="looming__readout mono">
          <span>Z = {Z(tc).toFixed(1)} m</span>
          <span className="muted">unknown to the estimator</span>
        </div>
      </div>

      <div className="looming__plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Inverse image size over time, with the fitted line reaching zero at contact">
          {[0, 0.5, 1].map((k) => (
            <line key={k} x1={P.l} x2={W - P.r} y1={Y(k * uMax)} y2={Y(k * uMax)} className={k === 0 ? "baseline" : "grid"} />
          ))}
          {Array.from({ length: Math.floor(tMax) + 1 }, (_, i) => (
            <text key={i} x={X(i)} y={H - 8} className="axis-label" textAnchor="middle">{i}s</text>
          ))}
          <text x={P.l - 8} y={Y(uMax)} className="axis-label" textAnchor="end" dominantBaseline="middle">1/s</text>
          <text x={P.l - 8} y={Y(0)} className="axis-label" textAnchor="end" dominantBaseline="middle">0</text>
          {/* fitting window */}
          {win.length > 0 && (
            <rect x={X(win[0])} y={P.t} width={Math.max(2, X(tc) - X(win[0]))} height={H - P.t - P.b} fill="var(--ink)" opacity="0.05" />
          )}
          {/* extrapolated fit */}
          {est !== null && (
            <line x1={X(win[0])} y1={Y(a + b * win[0])} x2={X(tc + est)} y2={Y(0)} stroke="var(--caution)" strokeWidth={1.5} strokeDasharray="4 4" />
          )}
          {samples.map((x) => (
            <circle key={x} cx={X(x)} cy={Y(1 / s(x))} r={win.includes(x) ? 3.5 : 2.5}
              fill={win.includes(x) ? "var(--ink)" : "var(--ink-3)"} />
          ))}
          {est !== null && (
            <g>
              <circle cx={X(tc + est)} cy={Y(0)} r={5} fill="var(--danger)" stroke="var(--bg)" strokeWidth={2} />
              <text x={X(tc + est)} y={Y(0) - 12} className="point-label" textAnchor="middle">contact</text>
            </g>
          )}
          <line x1={X(tc)} x2={X(tc)} y1={P.t} y2={H - P.b} className="cursor" />
        </svg>
        <div className="looming__stats">
          <div><span className="muted small">Estimated TTC</span><b className="mono">{est !== null ? `${est.toFixed(2)} s` : "warming up"}</b></div>
          <div><span className="muted small">True TTC</span><b className="mono">{truth.toFixed(2)} s</b></div>
          <div className="looming__controls">
            <button className="icon-btn" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause animation" : "Play animation"}>
              {playing ? <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7.5 5h3v14h-3zM13.5 5h3v14h-3z" /></svg>
                : <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7 5.5v13a.6.6 0 0 0 .9.5l10.4-6.5a.6.6 0 0 0 0-1L7.9 5a.6.6 0 0 0-.9.5Z" /></svg>}
            </button>
            <label className="slider">
              <span className="muted small">Closing speed <b className="mono">{speed} m/s</b></span>
              <input type="range" min={4} max={20} step={1} value={speed}
                onChange={(e) => { setSpeed(+e.target.value); setT(0.5); }} />
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}

const tEnd = (speed: number) => Math.min(T_END * (10 / speed) * 1.2, Z0 / speed - 0.4);
