import { useEffect, useMemo, useState } from "react";
import { listRuns, loadRun, type Run, type RunSummary, type Track } from "../lib/data";
import { usePlayer, type Player } from "../lib/player";
import { MOTION_COPY, REASON_COPY, RISK_COLOR, cap, fmtS, fmtTime } from "../lib/format";
import { Stage } from "../components/Stage";
import { Timeline } from "../components/Timeline";
import { ContactPlane } from "../components/ContactPlane";
import { TrackChart } from "../components/TrackChart";
import { Icon } from "../components/Icon";
import { RiskBadge, RiskLegend } from "../components/Risk";

export function Viewer({ name }: { name: string }) {
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    setRun(null);
    setError(null);
    loadRun(name).then(setRun, (e) => setError(String(e.message ?? e)));
  }, [name]);
  useEffect(() => void listRuns().then(setRuns), []);

  if (error) {
    return (
      <div className="page page--center">
        <div className="empty">
          <Icon name="alert" size={22} />
          <h2>Couldn’t open this run</h2>
          <p className="muted">{error}</p>
          <a className="btn" href="#/run/kitti-0007">Open the KITTI 0007 demo</a>
        </div>
      </div>
    );
  }
  if (!run) {
    return (
      <div className="page page--center">
        <div className="loading"><span className="spinner" /> Loading run…</div>
      </div>
    );
  }
  return <ViewerLoaded run={run} runs={runs} />;
}

function ViewerLoaded({ run, runs }: { run: Run; runs: RunSummary[] }) {
  const player = usePlayer(run);
  const [selected, setSelected] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [zones, setZones] = useState(true);
  const track = selected !== null ? run.trackById.get(selected) ?? null : null;

  const alertStarts = useMemo(() => {
    const out: number[] = [];
    run.worstByFrame.forEach((r, f) => {
      const alert = r === "caution" || r === "danger";
      const prev = run.worstByFrame[f - 1];
      if (alert && !(prev === "caution" || prev === "danger")) out.push(f);
    });
    return out;
  }, [run]);
  const nextAlert = () => {
    const f = alertStarts.find((s) => s > player.frame) ?? alertStarts[0];
    if (f !== undefined) {
      player.pause();
      player.seek(f);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest("input, textarea, select")) return;
      if (e.key === " " || e.key === "k") (e.preventDefault(), player.toggle());
      else if (e.key === "ArrowRight") (e.preventDefault(), player.step(e.shiftKey ? 10 : 1));
      else if (e.key === "ArrowLeft") (e.preventDefault(), player.step(e.shiftKey ? -10 : -1));
      else if (e.key === "Escape") setSelected(null);
      else if (e.key === "a") nextAlert();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const stats = useMemo(() => {
    const danger = run.tracks.filter((t) => t.worst === "danger").length;
    const caution = run.tracks.filter((t) => t.worst === "caution").length;
    const alertFrames = run.worstByFrame.filter((r) => r === "caution" || r === "danger").length;
    return { danger, caution, alertFrames };
  }, [run]);

  return (
    <div className="page viewer">
      <header className="viewer__head">
        <div className="viewer__title">
          <div className="eyebrow">{run.demo ? "Demo run" : "Run"} · {run.subtitle}</div>
          <div className="viewer__h">
            <h1>{run.title}</h1>
            {runs.length > 1 && (
              <select className="select" value={run.name} onChange={(e) => (location.hash = `#/run/${e.target.value}`)} aria-label="Switch run">
                {runs.map((r) => <option key={r.name} value={r.name}>{r.title}</option>)}
              </select>
            )}
          </div>
          <div className="chips">
            <span className="muted small">Prompts</span>
            {run.prompts.map((p) => <span key={p} className="chip mono">{p}</span>)}
          </div>
        </div>
        <dl className="kpis">
          <div><dt>Objects</dt><dd>{run.tracks.length}</dd></div>
          <div><dt>Reached danger</dt><dd>{stats.danger}</dd></div>
          <div><dt>Reached caution</dt><dd>{stats.caution}</dd></div>
          <div><dt>Time on alert</dt><dd>{fmtS(stats.alertFrames / run.fps)}</dd></div>
        </dl>
      </header>

      <div className="viewer__grid">
        <section className="viewer__main">
          <div className="card card--flush">
            <Stage run={run} player={player} selected={selected} hovered={hovered} onSelect={setSelected} onHover={setHovered} zones={zones} />
            <Transport player={player} run={run} zones={zones} setZones={setZones} onNextAlert={alertStarts.length ? nextAlert : undefined} />
          </div>
          <div className="card">
            <div className="card__head">
              <h3>Risk timeline</h3>
              <RiskLegend />
            </div>
            <Timeline run={run} frame={player.frame} selected={selected} hovered={hovered}
              onSeek={(f) => (player.pause(), player.seek(f))} onSelect={setSelected} onHover={setHovered} />
          </div>
        </section>

        <aside className="viewer__aside">
          <div className="card">
            <div className="card__head">
              <h3>Contact plane</h3>
              <span className="muted small mono">{fmtTime(player.frame / run.fps)}</span>
            </div>
            <ContactPlane run={run} frame={player.frame} selected={selected} hovered={hovered} onSelect={setSelected} onHover={setHovered} />
          </div>
          <InView run={run} frame={player.frame} selected={selected} hovered={hovered} onSelect={setSelected} onHover={setHovered} />
        </aside>
      </div>

      {track ? (
        <TrackDetail key={track.id} track={track} run={run} player={player} onClose={() => setSelected(null)} />
      ) : (
        <div className="hint">
          <Icon name="layers" />
          Click any object on the video, timeline or contact plane to see how its time to contact evolved.
          <span className="kbd-row"><kbd>Space</kbd> play <kbd>←</kbd><kbd>→</kbd> step <kbd>A</kbd> next alert</span>
        </div>
      )}
    </div>
  );
}

function Transport({ player, run, zones, setZones, onNextAlert }: {
  player: Player; run: Run; zones: boolean; setZones: (v: boolean) => void; onNextAlert?: () => void;
}) {
  return (
    <div className="transport">
      <div className="transport__group">
        <button className="icon-btn" onClick={() => player.step(-1)} aria-label="Previous frame" title="Previous frame (←)"><Icon name="prev" /></button>
        <button className="icon-btn icon-btn--primary" onClick={player.toggle} aria-label={player.playing ? "Pause" : "Play"} title="Play / pause (Space)">
          <Icon name={player.playing ? "pause" : "play"} size={18} />
        </button>
        <button className="icon-btn" onClick={() => player.step(1)} aria-label="Next frame" title="Next frame (→)"><Icon name="next" /></button>
        <span className="transport__time mono">
          {fmtTime(player.frame / run.fps)} <span className="muted">/ {fmtTime((run.frames - 1) / run.fps)}</span>
        </span>
      </div>
      <div className="transport__group">
        {onNextAlert && (
          <button className="btn btn--ghost btn--sm" onClick={onNextAlert} title="Jump to the next caution or danger (A)">
            <Icon name="skip" size={12} /> Next alert
          </button>
        )}
        <button className={`btn btn--ghost btn--sm ${zones ? "is-on" : ""}`} onClick={() => setZones(!zones)} aria-pressed={zones}>Zones</button>
        <div className="seg" role="group" aria-label="Playback speed">
          {[0.25, 0.5, 1].map((r) => (
            <button key={r} className={player.rate === r ? "is-on" : ""} onClick={() => player.setRate(r)}>{r}×</button>
          ))}
        </div>
      </div>
    </div>
  );
}

function InView({ run, frame, selected, hovered, onSelect, onHover }: {
  run: Run; frame: number; selected: number | null; hovered: number | null;
  onSelect: (id: number | null) => void; onHover: (id: number | null) => void;
}) {
  const rows = run.byFrame[frame] ?? [];
  return (
    <div className="card inview">
      <div className="card__head">
        <h3>In view</h3>
        <span className="muted small">{rows.length} object{rows.length === 1 ? "" : "s"}</span>
      </div>
      {rows.length === 0 && <p className="muted small pad">Nothing tracked in this frame.</p>}
      <ul className="inview__list">
        {rows.map((r) => {
          const ttcPct = r.ttc !== null ? Math.max(0, Math.min(1, 1 - r.ttc / 10)) : 0;
          return (
            <li key={r.id}>
              <button
                className={`obj ${selected === r.id ? "is-selected" : ""} ${hovered === r.id ? "is-hovered" : ""}`}
                onClick={() => onSelect(selected === r.id ? null : r.id)}
                onMouseEnter={() => onHover(r.id)}
                onMouseLeave={() => onHover(null)}
                style={{ ["--risk" as string]: RISK_COLOR[r.risk] }}
              >
                <span className="obj__bar" />
                <span className="obj__main">
                  <span className="obj__name"><b className="mono">#{r.id}</b> {r.label}</span>
                  <span className="obj__meta">{cap(r.zone)} · {MOTION_COPY[r.motion] ?? r.motion}{r.onCourse ? " · on course" : ""}</span>
                  {r.reasons.length > 0 && (
                    <span className="obj__flags">{r.reasons.map((x) => <span key={x} className="flag" title={REASON_COPY[x] ?? x}>{x.replace(/_/g, " ")}</span>)}</span>
                  )}
                </span>
                <span className="obj__ttc">
                  <span className="mono obj__ttc-v">{r.ttc !== null && r.ttc < 10 ? fmtS(r.ttc) : "—"}</span>
                  <span className="meter"><span style={{ width: `${ttcPct * 100}%` }} /></span>
                  <RiskBadge risk={r.risk} small />
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function TrackDetail({ track, run, player, onClose }: { track: Track; run: Run; player: Player; onClose: () => void }) {
  const reasons = useMemo(() => {
    const c = new Map<string, number>();
    track.rows.forEach((r) => r.reasons.forEach((x) => c.set(x, (c.get(x) ?? 0) + 1)));
    return [...c.entries()].sort((a, b) => b[1] - a[1]);
  }, [track]);
  const seek = (f: number) => (player.pause(), player.seek(f));
  const riskShare = (["danger", "caution", "uncertain", "safe"] as const).filter((r) => track.counts[r] > 0);

  return (
    <section className="detail card">
      <div className="detail__head">
        <div>
          <div className="eyebrow">Object</div>
          <h2><span className="mono">#{track.id}</span> {track.label} <RiskBadge risk={track.worst} /></h2>
          <p className="muted small">
            In view {fmtS(track.first / run.fps)} – {fmtS(track.last / run.fps)} · {track.seen} frames ·
            mostly {(MOTION_COPY[track.mainMotion] ?? track.mainMotion).toLowerCase()} in the {track.mainZone} zone
          </p>
        </div>
        <div className="detail__actions">
          <button className="btn btn--ghost btn--sm" onClick={() => seek(track.first)}>Go to first frame</button>
          {track.minTtcFrame !== null && (
            <button className="btn btn--ghost btn--sm" onClick={() => seek(track.minTtcFrame!)}>Go to closest approach</button>
          )}
          <button className="icon-btn" onClick={onClose} aria-label="Close object detail"><Icon name="close" /></button>
        </div>
      </div>

      <dl className="tiles">
        <div className="tile"><dt>Closest reliable TTC</dt><dd className="mono">{fmtS(track.minTtc)}</dd></div>
        <div className="tile"><dt>Worst risk</dt><dd>{cap(track.worst)}</dd></div>
        <div className="tile"><dt>Ever on course</dt><dd>{track.everOnCourse ? "Yes" : "No"}</dd></div>
        <div className="tile"><dt>Reliable estimates</dt><dd className="mono">{Math.round(track.reliableFrac * 100)}%</dd></div>
        <div className="tile tile--wide">
          <dt>Frames by risk</dt>
          <dd>
            <div className="share">
              {riskShare.map((r) => (
                <span key={r} style={{ flex: track.counts[r], background: RISK_COLOR[r] }} title={`${cap(r)}: ${track.counts[r]} frames`} />
              ))}
            </div>
            <div className="share__legend small">
              {riskShare.map((r) => (
                <span key={r}><i className="swatch" style={{ background: RISK_COLOR[r] }} />{cap(r)} <span className="mono muted">{track.counts[r]}</span></span>
              ))}
            </div>
          </dd>
        </div>
      </dl>

      <div className="detail__charts">
        <div>
          <div className="chart__title">
            <h4>Time to contact</h4>
            <span className="muted small">Line fit to 1/size over 0.8 s · shaded band = low/high estimate · hollow = flagged unreliable</span>
          </div>
          <TrackChart track={track} fps={run.fps} frame={player.frame} kind="ttc" onSeek={seek} />
        </div>
        <div>
          <div className="chart__title">
            <h4>Predicted miss at contact</h4>
            <span className="muted small">Lateral offset in object heights · |m| &lt; 1 counts as on course</span>
          </div>
          <TrackChart track={track} fps={run.fps} frame={player.frame} kind="miss" onSeek={seek} />
        </div>
      </div>

      {reasons.length > 0 && (
        <div className="reasons">
          <h4>Why estimates were flagged</h4>
          <ul>
            {reasons.map(([k, v]) => (
              <li key={k}>
                <span className="flag">{k.replace(/_/g, " ")}</span>
                <span className="muted">{REASON_COPY[k] ?? k}</span>
                <span className="mono">{v} frame{v === 1 ? "" : "s"}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

