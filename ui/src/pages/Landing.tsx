import { useEffect, useMemo, useState } from "react";
import { DEMO_NAME, loadRun, type Risk, type Run } from "../lib/data";
import { usePlayer } from "../lib/player";
import { RISK_COLOR, RISK_COPY, fmtS } from "../lib/format";
import { Stage } from "../components/Stage";
import { Looming } from "../components/Looming";
import { Icon } from "../components/Icon";
import { RiskBadge } from "../components/Risk";

export function Landing() {
  const [run, setRun] = useState<Run | null>(null);
  useEffect(() => void loadRun(DEMO_NAME).then(setRun, () => {}), []);

  return (
    <div className="landing">
      <section className="hero">
        <div className="hero__glow" aria-hidden="true" />
        <div className="hero__copy">
          <div className="eyebrow eyebrow--pill">
            <span lang="sa" className="deva">दिग्बोध</span>
            <span className="eyebrow__sep" />
            DigBodha 1<span className="hide-sm">&nbsp;· single-camera collision-risk estimation</span>
          </div>
          <h1 className="display">
            See what’s coming, <em>and&nbsp;when.</em>
          </h1>
          <p className="lede">
            Give it one ordinary camera and a few words like{" "}
            <span className="nowrap"><span className="chip mono">vehicle</span> <span className="chip mono">person</span>.</span> DigBodha tracks every match, removes the camera’s own motion,
            and estimates when each object will reach you, with no depth sensor and no calibration.
          </p>
          <div className="cta">
            <a className="btn btn--primary" href={`#/run/${DEMO_NAME}`}>
              Explore the KITTI run <Icon name="arrow" />
            </a>
            <a className="btn" href="#/analyze">
              <Icon name="upload" /> Analyze a video
            </a>
          </div>
        </div>

        <div className="hero__stage">{run ? <HeroPlayer run={run} /> : <div className="stage stage--placeholder" />}</div>
      </section>

      <section className="band">
        <div className="band__intro">
          <div className="eyebrow">Held-out KITTI sequences</div>
          <p className="muted small">Pooled over sequences 0019 and 0007, which were never used for tuning.</p>
        </div>
        <dl className="band__stats">
          <div><dt>Median TTC error</dt><dd>11.5<small>%</small></dd><p>on estimates the system marks reliable</p></div>
          <div><dt>Approach detection</dt><dd>0.93<small> / 0.90</small></dd><p>precision / recall</p></div>
          <div><dt>Hazards warned</dt><dd>40<small> / 57</small></dd><p>hazard events, 70%</p></div>
          <div><dt>Median warning lead</dt><dd>1.2<small> s</small></dd><p>before the hazard</p></div>
        </dl>
        <a className="band__link" href="#/evaluation">Full evaluation <Icon name="arrow" size={14} /></a>
      </section>

      <section className="section">
        <header className="section__head">
          <div className="eyebrow">How it works</div>
          <h2 className="h2">Three stages, from pixels to a risk call you can inspect.</h2>
        </header>
        <ol className="stages">
          <li className="stagecard">
            <span className="stagecard__n mono">01</span>
            <StageArt kind="segment" />
            <h3>Find and follow</h3>
            <p>SAM 3.1 segments every object matching your words and keeps its identity, running in 60-frame chunks stitched together by mask IoU and Hungarian matching.</p>
            <span className="stagecard__tag mono">src/segment.py</span>
          </li>
          <li className="stagecard">
            <span className="stagecard__n mono">02</span>
            <StageArt kind="ego" />
            <h3>Cancel the camera</h3>
            <p>Shi–Tomasi corners on the background are tracked with Lucas–Kanade flow. A RANSAC homography predicts how a static scene should move, and what remains is each object’s own motion.</p>
            <span className="stagecard__tag mono">src/egomotion.py</span>
          </li>
          <li className="stagecard">
            <span className="stagecard__n mono">03</span>
            <StageArt kind="ttc" />
            <h3>Read the looming</h3>
            <p>As something approaches it grows in the image. A line fit to 1/size reaches zero at the moment of contact, so focal length, real size and distance all cancel out.</p>
            <span className="stagecard__tag mono">src/ttc.py</span>
          </li>
        </ol>
      </section>

      <section className="section section--split">
        <div className="section__copy">
          <div className="eyebrow">The principle</div>
          <h2 className="h2">Time to contact, <em>without knowing the distance.</em></h2>
          <p className="muted">
            An object of height <i>H</i> at depth <i>Z</i> appears <i>s = fH/Z</i> pixels tall. If it closes at a steady
            speed, <i>1/s</i> falls on a straight line in time. Fit that line over the last 0.8 s and its zero crossing is
            the time to contact.
          </p>
          <div className="formula mono">
            TTC = u<sub>now</sub> / −b,&nbsp;&nbsp;where&nbsp; 1/s ≈ a + b·t
          </div>
          <p className="muted small">After Lee (1976). DigBodha fits 1/s rather than log s, which lagged under constant-speed approach in the team’s tests.</p>
        </div>
        <Looming />
      </section>

      <section className="section">
        <header className="section__head">
          <div className="eyebrow">The risk call</div>
          <h2 className="h2">Four states, each with its reasons, and no black box.</h2>
          <p className="muted">
            Whether an object is on course comes from its predicted sideways miss at contact, measured in its own heights. When a
            reliability check fails, DigBodha says <b>uncertain</b> instead of guessing.
          </p>
        </header>
        <div className="rules">
          {(["danger", "caution", "uncertain", "safe"] as Risk[]).map((r) => (
            <div key={r} className="rule" style={{ ["--risk" as string]: RISK_COLOR[r] }}>
              <RiskBadge risk={r} />
              <p>{RISK_COPY[r].rule}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="notice">
          <Icon name="alert" size={18} />
          <div>
            <b>Research prototype, not a safety system.</b>
            <p className="muted small">
              It runs offline at about 1–2 FPS on an 8 GB laptop GPU. It assumes straight-line motion, treats the image centre as the
              direction of travel, and has only been evaluated on KITTI driving scenes. Don’t rely on it for driving or navigation.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

function HeroPlayer({ run }: { run: Run }) {
  const player = usePlayer(run, { loop: true, autoplay: true });
  const rows = run.byFrame[player.frame] ?? [];
  const nearest = rows.filter((r) => r.ttc !== null && r.reliable).sort((a, b) => a.ttc! - b.ttc!)[0];
  const alertSegs = useMemo(() => {
    const out: { x: number; w: number; risk: Risk }[] = [];
    run.worstByFrame.forEach((risk, f) => {
      const last = out[out.length - 1];
      if (last && last.risk === risk) last.w++;
      else out.push({ x: f, w: 1, risk });
    });
    return out;
  }, [run]);

  return (
    <div className="heroplayer">
      <Stage run={run} player={player} compact loop zones={false} />
      <div className="heroplayer__bar">
        <button className="icon-btn" onClick={player.toggle} aria-label={player.playing ? "Pause" : "Play"}>
          <Icon name={player.playing ? "pause" : "play"} size={14} />
        </button>
        <div className="heroplayer__track" onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          player.seek(((e.clientX - r.left) / r.width) * run.frames);
        }}>
          <svg viewBox={`0 0 ${run.frames} 1`} preserveAspectRatio="none">
            {alertSegs.map((s, i) => (
              <rect key={i} x={s.x} width={s.w} y={0} height={1}
                fill={s.risk === "safe" ? "rgba(255,255,255,0.10)" : RISK_COLOR[s.risk]} />
            ))}
          </svg>
          <div className="playhead" style={{ left: `${((player.frame + 0.5) / run.frames) * 100}%` }}><span /></div>
        </div>
        <div className="heroplayer__readout">
          <span><span className="muted">Tracked</span> <b className="mono">{rows.length}</b></span>
          <span><span className="muted">Nearest TTC</span> <b className="mono">{nearest ? fmtS(nearest.ttc) : "—"}</b></span>
        </div>
      </div>
      <p className="caption muted small">
        KITTI tracking sequence 0007, held out from tuning. Live overlay drawn from the pipeline’s <span className="mono">frames.csv</span>.
      </p>
    </div>
  );
}

function StageArt({ kind }: { kind: "segment" | "ego" | "ttc" }) {
  if (kind === "segment") {
    return (
      <svg viewBox="0 0 200 90" className="art" aria-hidden="true">
        <rect x="1" y="1" width="198" height="88" rx="8" className="art__frame" />
        <path d="M30 70 Q 40 40 70 42 L 92 44 Q 104 46 106 70 Z" className="art__mask art__mask--a" />
        <path d="M128 72 L 132 34 Q 136 26 142 34 L 146 72 Z" className="art__mask art__mask--b" />
        <rect x="26" y="38" width="84" height="36" className="art__box" />
        <rect x="124" y="24" width="26" height="50" className="art__box" />
        <text x="26" y="33" className="art__label">#1 vehicle</text>
        <text x="124" y="19" className="art__label">#2 person</text>
      </svg>
    );
  }
  if (kind === "ego") {
    const arrows = [[30, 24], [62, 18], [150, 20], [176, 34], [22, 62], [178, 68], [44, 80], [160, 80]];
    return (
      <svg viewBox="0 0 200 90" className="art" aria-hidden="true">
        <rect x="1" y="1" width="198" height="88" rx="8" className="art__frame" />
        {arrows.map(([x, y], i) => {
          const dx = (x - 100) * 0.12, dy = (y - 50) * 0.12;
          return (
            <g key={i}>
              <circle cx={x} cy={y} r="1.8" className="art__pt" />
              <line x1={x} y1={y} x2={x + dx} y2={y + dy} className="art__flow" />
            </g>
          );
        })}
        <rect x="82" y="36" width="36" height="28" className="art__box art__box--hot" />
        <line x1="100" y1="50" x2="88" y2="50" className="art__own" />
        <text x="82" y="31" className="art__label">own motion</text>
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 200 90" className="art" aria-hidden="true">
      <rect x="1" y="1" width="198" height="88" rx="8" className="art__frame" />
      <line x1="16" y1="76" x2="186" y2="76" className="art__axis" />
      {[0, 1, 2, 3, 4, 5, 6].map((i) => (
        <circle key={i} cx={22 + i * 18} cy={18 + i * 7.56} r="2.6" className="art__pt art__pt--hi" />
      ))}
      <line x1="22" y1="18" x2="160" y2="76" className="art__fit" />
      <circle cx="160" cy="76" r="4" className="art__hit" />
      <text x="160" y="66" className="art__label" textAnchor="middle">TTC</text>
      <text x="22" y="12" className="art__label">1/s</text>
    </svg>
  );
}
