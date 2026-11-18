import { useEffect, useState } from "react";
import { SEQUENCES, loadMetrics, type Metrics } from "../lib/data";
import { cap } from "../lib/format";

const PRED = ["safe", "uncertain", "caution", "danger"];
const GT = ["safe", "caution", "danger"];

export function Evaluation() {
  const [m, setM] = useState<Record<string, Metrics> | null>(null);
  const [seq, setSeq] = useState("0007");
  useEffect(() => void loadMetrics().then(setM), []);
  if (!m) return <div className="page page--center"><div className="loading"><span className="spinner" /> Loading metrics…</div></div>;
  const cur = m[seq];

  return (
    <div className="page eval">
      <header className="page__head">
        <div className="eyebrow">Evaluation · KITTI Object Tracking</div>
        <h1 className="h1">How well does it hold up?</h1>
        <p className="muted lede-sm">
          Reference TTC is derived from KITTI’s 3D labels as <span className="mono">Z / −dZ/dt</span> for approaching objects within 40 m.
          Predictions match labels by class and 2D IoU ≥ 0.5. Rules and reliability checks were tuned on three sequences, then
          frozen before the two held-out ones were scored.
        </p>
      </header>

      <section className="card">
        <div className="card__head"><h3>Per sequence</h3><span className="muted small">Final configuration: height-based TTC with width fallback, no debounce</span></div>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th />
                {SEQUENCES.map((s) => (
                  <th key={s.seq} className={s.split === "held-out" ? "is-held" : ""}>
                    <span className="mono">{s.seq}</span>
                    <span className="table__split">{s.split}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <Row label="Median relative TTC error, reliable" hint="lower is better" vals={SEQUENCES.map((s) => `${m[s.seq].ttc_reliable.median_rel_pct.toFixed(1)}%`)} />
              <Row label="Reliable TTC within 20% of truth" vals={SEQUENCES.map((s) => `${m[s.seq].ttc_reliable.within_20pct.toFixed(0)}%`)} />
              <Row label="Median relative TTC error, flagged" hint="the flag catches bad estimates" vals={SEQUENCES.map((s) => `${m[s.seq].ttc_unreliable.median_rel_pct.toFixed(1)}%`)} />
              <Row label="Approach precision / recall" vals={SEQUENCES.map((s) => `${m[s.seq].approach.precision.toFixed(2)} / ${m[s.seq].approach.recall.toFixed(2)}`)} />
              <Row label="Alert precision / recall" hint="per matched object-frame" vals={SEQUENCES.map((s) => `${m[s.seq].alerts.precision.toFixed(2)} / ${m[s.seq].alerts.recall.toFixed(2)}`)} />
              <Row label="Labelled objects matched" vals={SEQUENCES.map((s) => `${m[s.seq].gt_matched_pct.toFixed(0)}%`)} />
              <Row label="Frames" vals={SEQUENCES.map((s) => String(m[s.seq].frames))} />
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <div className="card__head">
          <h3>Does the reliability flag mean anything?</h3>
          <div className="legend">
            <span><i className="swatch swatch--solid" /> Marked reliable</span>
            <span><i className="swatch swatch--hatch" /> Flagged unreliable</span>
          </div>
        </div>
        <p className="muted small pad-x">Median relative TTC error. Flagged estimates are several times worse, so the flag is doing its job.</p>
        <PairedBars m={m} />
      </section>

      <div className="seqpick">
        <span className="muted small">Sequence</span>
        <div className="seg">
          {SEQUENCES.map((s) => (
            <button key={s.seq} className={seq === s.seq ? "is-on" : ""} onClick={() => setSeq(s.seq)}>
              <span className="mono">{s.seq}</span>{s.split === "held-out" && <span className="dotmark" title="held out" />}
            </button>
          ))}
        </div>
        <span className="muted small">{SEQUENCES.find((s) => s.seq === seq)!.split}</span>
      </div>

      <div className="eval__grid">
        <section className="card">
          <div className="card__head"><h3>Risk: truth vs prediction</h3><span className="muted small">object-frames · row %</span></div>
          <Confusion c={cur.risk_confusion_gt_rows_pred_cols} />
          <p className="muted small pad">
            {cur.alerts.fn_flagged_uncertain} of {cur.alerts.fn} missed alerts were labelled <b>uncertain</b>, so the system knew it couldn’t tell.
          </p>
        </section>
        <section className="card">
          <div className="card__head"><h3>TTC error by distance</h3><span className="muted small">median relative error, reliable estimates</span></div>
          <DepthBars d={cur.ttc_by_depth} />
        </section>
      </div>
    </div>
  );
}

function Row({ label, hint, vals }: { label: string; hint?: string; vals: string[] }) {
  return (
    <tr>
      <th scope="row">{label}{hint && <span className="table__hint">{hint}</span>}</th>
      {vals.map((v, i) => <td key={i} className={`mono ${SEQUENCES[i].split === "held-out" ? "is-held" : ""}`}>{v}</td>)}
    </tr>
  );
}

function PairedBars({ m }: { m: Record<string, Metrics> }) {
  const max = Math.max(...SEQUENCES.map((s) => m[s.seq].ttc_unreliable.median_rel_pct)) * 1.1;
  return (
    <div className="pbars">
      {SEQUENCES.map((s) => {
        const r = m[s.seq].ttc_reliable, u = m[s.seq].ttc_unreliable;
        return (
          <div key={s.seq} className="pbars__row">
            <div className="pbars__label"><span className="mono">{s.seq}</span><span className="muted small">{s.split}</span></div>
            <div className="pbars__bars">
              <div className="pbar" title={`Reliable: ${r.median_rel_pct}% (n=${r.n})`}>
                <span className="pbar__fill pbar__fill--solid" style={{ width: `${(r.median_rel_pct / max) * 100}%` }} />
                <span className="pbar__val mono">{r.median_rel_pct.toFixed(1)}%</span>
              </div>
              <div className="pbar" title={`Flagged: ${u.median_rel_pct}% (n=${u.n})`}>
                <span className="pbar__fill pbar__fill--hatch" style={{ width: `${(u.median_rel_pct / max) * 100}%` }} />
                <span className="pbar__val mono">{u.median_rel_pct.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Confusion({ c }: { c: Record<string, Record<string, number>> }) {
  return (
    <div className="cm">
      <div className="cm__corner small muted">truth ↓ · predicted →</div>
      {PRED.map((p) => <div key={p} className="cm__col small">{cap(p)}</div>)}
      {GT.map((g) => {
        const row = c[g] ?? {};
        const total = PRED.reduce((s, p) => s + (row[p] ?? 0), 0) || 1;
        return (
          <div key={g} className="cm__row">
            <div className="cm__rowlabel small">{cap(g)}</div>
            {PRED.map((p) => {
              const v = row[p] ?? 0, share = v / total;
              const hit = g === p;
              return (
                <div key={p} className={`cm__cell ${hit ? "is-diag" : ""}`}
                  style={{ ["--a" as string]: (0.06 + share * 0.85).toFixed(3) }}
                  title={`truth ${g}, predicted ${p}: ${v} (${Math.round(share * 100)}% of row)`}>
                  <b className="mono">{Math.round(share * 100)}%</b>
                  <span className="mono small">{v}</span>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

function DepthBars({ d }: { d: Metrics["ttc_by_depth"] }) {
  const entries = Object.entries(d);
  const max = Math.max(30, ...entries.map(([, v]) => v.median_rel_pct)) * 1.1;
  return (
    <div className="dbars">
      {entries.map(([k, v]) => (
        <div key={k} className="dbars__col" title={`${k}: ${v.median_rel_pct}% (n=${v.n})`}>
          <div className="dbars__track">
            <span style={{ height: `${(v.median_rel_pct / max) * 100}%` }}>
              <b className="dbars__val mono">{v.median_rel_pct.toFixed(1)}%</b>
            </span>
          </div>
          <span className="dbars__label">{k}</span>
          <span className="muted small mono">n={v.n}</span>
        </div>
      ))}
    </div>
  );
}
