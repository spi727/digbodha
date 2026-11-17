import { useEffect, useRef, useState } from "react";
import { health, listRuns, type RunSummary } from "../lib/data";
import { Icon } from "../components/Icon";

type Health = Awaited<ReturnType<typeof health>>;
interface Job {
  id: string; name: string; status: "running" | "done" | "failed"; stage: string; progress: number;
  lines: string[]; started: number; ended: number | null;
}

const SUGGEST = ["vehicle", "person", "bicycle", "dog", "bus", "truck", "motorcycle", "auto rickshaw"];
const STEPS = [
  { key: "upload", label: "Upload" },
  { key: "frames extracted", label: "Extract frames" },
  { key: "model loaded", label: "Load SAM 3.1" },
  { key: "segmenting", label: "Segment & track" },
  { key: "writing outputs", label: "Score risk" },
];

export function Analyze() {
  const [h, setH] = useState<Health | undefined>(undefined);
  const [file, setFile] = useState<File | null>(null);
  const [prompts, setPrompts] = useState<string[]>(["vehicle", "person"]);
  const [draft, setDraft] = useState("");
  const [adv, setAdv] = useState({ chunk: 60, overlap: 2, max_objects: 16 });
  const [showAdv, setShowAdv] = useState(false);
  const [drag, setDrag] = useState(false);
  const [upload, setUpload] = useState<number | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const input = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    health().then(setH);
    listRuns().then(setRuns);
  }, []);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const id = window.setInterval(async () => {
      try {
        const j = await (await fetch(`/api/jobs/${job.id}`)).json();
        setJob(j);
        if (j.status !== "running") listRuns().then(setRuns);
      } catch { /* keep polling */ }
    }, 1000);
    return () => window.clearInterval(id);
  }, [job?.id, job?.status]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [job?.lines.length]);

  const addPrompt = (p: string) => {
    const v = p.trim().toLowerCase();
    if (v && !prompts.includes(v)) setPrompts([...prompts, v]);
    setDraft("");
  };

  const submit = () => {
    if (!file || !prompts.length) return;
    setErr(null);
    const q = new URLSearchParams({
      filename: file.name, prompts: prompts.join(","), chunk: String(adv.chunk),
      overlap: String(adv.overlap), max_objects: String(adv.max_objects),
    });
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/analyze?${q}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && setUpload(e.loaded / e.total);
    xhr.onload = () => {
      setUpload(null);
      try {
        const res = JSON.parse(xhr.responseText);
        if (xhr.status >= 300) throw new Error(res.error ?? xhr.statusText);
        setJob({ id: res.job, name: res.name, status: "running", stage: "starting", progress: 0, lines: [], started: Date.now() / 1000, ended: null });
      } catch (e) {
        setErr(String((e as Error).message));
      }
    };
    xhr.onerror = () => {
      setUpload(null);
      setErr("Upload failed. Is the server still running?");
    };
    setUpload(0);
    xhr.send(file);
  };

  const offline = h === null;
  const busy = upload !== null || job?.status === "running";

  return (
    <div className="page analyze">
      <header className="page__head">
        <div className="eyebrow">New analysis</div>
        <h1 className="h1">Analyze a video</h1>
        <p className="muted lede-sm">Drop in a clip from a forward-facing camera and name what to watch for. Results open in the same viewer as the demo.</p>
      </header>

      <ServerStatus h={h} />

      <div className="analyze__grid">
        <section className={`card form ${offline ? "is-disabled" : ""}`}>
          <label
            className={`drop ${drag ? "is-drag" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(e) => (e.preventDefault(), setDrag(true))}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              const f = e.dataTransfer.files[0];
              if (f) setFile(f);
            }}
          >
            <input ref={input} type="file" accept="video/*" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} disabled={offline || busy} />
            <span className="drop__icon"><Icon name="upload" size={20} /></span>
            {file ? (
              <>
                <b>{file.name}</b>
                <span className="muted small mono">{(file.size / 1e6).toFixed(1)} MB · click to replace</span>
              </>
            ) : (
              <>
                <b>Drop a video here, or click to browse</b>
                <span className="muted small">MP4, MOV, AVI, MKV or WebM. Frames are extracted at the video’s own rate.</span>
              </>
            )}
          </label>

          <div className="field">
            <div className="field__label">
              <span>What to track</span>
              <span className="muted small">open-vocabulary, so plain words work</span>
            </div>
            <div className="tokens" onClick={(e) => (e.currentTarget.querySelector("input") as HTMLInputElement)?.focus()}>
              {prompts.map((p) => (
                <span key={p} className="token mono">
                  {p}
                  <button onClick={() => setPrompts(prompts.filter((x) => x !== p))} aria-label={`Remove ${p}`}><Icon name="close" size={11} /></button>
                </span>
              ))}
              <input
                value={draft}
                placeholder={prompts.length ? "add another…" : "e.g. vehicle"}
                onChange={(e) => (e.target.value.endsWith(",") ? addPrompt(e.target.value.slice(0, -1)) : setDraft(e.target.value))}
                onKeyDown={(e) => {
                  if (e.key === "Enter") (e.preventDefault(), addPrompt(draft));
                  if (e.key === "Backspace" && !draft && prompts.length) setPrompts(prompts.slice(0, -1));
                }}
                aria-label="Add a prompt"
              />
            </div>
            <div className="suggest">
              {SUGGEST.filter((s) => !prompts.includes(s)).map((s) => (
                <button key={s} className="chip chip--btn" onClick={() => addPrompt(s)}>+ {s}</button>
              ))}
            </div>
            <p className="muted small">Each extra prompt costs speed: about 2.1 FPS with one prompt and 0.6–1 FPS with two on an 8 GB GPU.</p>
          </div>

          <button className="disclosure" onClick={() => setShowAdv(!showAdv)} aria-expanded={showAdv}>
            <span className={`chev ${showAdv ? "is-open" : ""}`}>›</span> Advanced
          </button>
          {showAdv && (
            <div className="adv">
              {([
                ["chunk", "Chunk length", "frames per SAM pass; halves on out-of-memory"],
                ["overlap", "Chunk overlap", "frames shared to link IDs across chunks"],
                ["max_objects", "Max objects", "upper bound on tracked objects"],
              ] as const).map(([k, label, hint]) => (
                <label key={k} className="num">
                  <span>{label}</span>
                  <input type="number" min={1} value={adv[k]} onChange={(e) => setAdv({ ...adv, [k]: Math.max(1, +e.target.value || 1) })} />
                  <span className="muted small">{hint}</span>
                </label>
              ))}
            </div>
          )}

          {err && <div className="error small"><Icon name="alert" size={14} /> {err}</div>}

          <button className="btn btn--primary btn--block" disabled={!file || !prompts.length || offline || busy} onClick={submit}>
            {upload !== null ? `Uploading ${(upload * 100).toFixed(0)}%` : job?.status === "running" ? "Analyzing…" : "Run DigBodha"}
            {!busy && <Icon name="arrow" />}
          </button>
        </section>

        <section className="card job">
          {job ? <JobView job={job} logRef={logRef} /> : <RecentRuns runs={runs} />}
        </section>
      </div>
    </div>
  );
}

function ServerStatus({ h }: { h: Health | undefined }) {
  if (h === undefined) return <div className="status"><span className="spinner" /> Checking for the local server…</div>;
  if (h === null) {
    return (
      <div className="status status--off">
        <span className="status__dot" />
        <div>
          <b>The local DigBodha server isn’t running.</b>
          <p className="muted small">Live analysis needs SAM 3.1 on a CUDA GPU. Start the server from the repo root in your <span className="mono">cvproj</span> environment, then reload:</p>
          <pre className="code mono">python ui/server.py</pre>
          <p className="muted small">You can still explore the <a href="#/run/kitti-0007">KITTI 0007 demo run</a> without it.</p>
        </div>
      </div>
    );
  }
  return (
    <div className={`status ${h.sam3 ? "status--on" : "status--warn"}`}>
      <span className="status__dot" />
      <div>
        <b>{h.sam3 ? "Server connected, and SAM 3 is installed." : "Server connected, but SAM 3 isn’t importable."}</b>
        <p className="muted small">
          Python {h.python} · OpenCV {h.cv2 ? "available" : "missing"}
          {!h.sam3 && " · install requirements.txt and log in to Hugging Face to run new analyses"}
        </p>
      </div>
    </div>
  );
}

function JobView({ job, logRef }: { job: Job; logRef: React.RefObject<HTMLPreElement> }) {
  const idx = Math.max(1, STEPS.findIndex((s) => s.key === job.stage));
  const doneAll = job.status === "done";
  const elapsed = ((job.ended ?? Date.now() / 1000) - job.started);
  return (
    <>
      <div className="card__head">
        <h3>{doneAll ? "Analysis complete" : job.status === "failed" ? "Analysis failed" : "Analyzing"} <span className="muted mono small">{job.name}</span></h3>
        <span className="muted small mono">{Math.floor(elapsed / 60)}m {Math.floor(elapsed % 60)}s</span>
      </div>
      <ol className="steps">
        {STEPS.map((s, i) => {
          const state = doneAll || i < idx ? "done" : i === idx && job.status === "running" ? "active" : job.status === "failed" && i === idx ? "failed" : "todo";
          return (
            <li key={s.key} className={`step step--${state}`}>
              <span className="step__mark">{state === "done" ? <Icon name="check" size={12} /> : i + 1}</span>
              <span>{s.label}</span>
              {s.key === "segmenting" && state === "active" && <span className="mono small muted">{Math.round(job.progress * 100)}%</span>}
            </li>
          );
        })}
      </ol>
      <div className="progress"><span style={{ width: `${(doneAll ? 1 : job.progress) * 100}%` }} /></div>
      <pre ref={logRef} className="log mono">{job.lines.join("\n") || "Starting analyze.py…"}</pre>
      {doneAll && <a className="btn btn--primary btn--block" href={`#/run/${job.name}`}>Open results <Icon name="arrow" /></a>}
    </>
  );
}

function RecentRuns({ runs }: { runs: RunSummary[] }) {
  return (
    <>
      <div className="card__head"><h3>Runs</h3><span className="muted small">from outputs/</span></div>
      <ul className="runs">
        {runs.map((r) => (
          <li key={r.name}>
            <a href={`#/run/${r.name}`} className="runrow">
              <span className="runrow__name">{r.title}{r.demo && <span className="chip chip--sm">demo</span>}</span>
              <span className="muted small mono">{r.prompts.join(", ")}{r.frames ? ` · ${r.frames} frames` : ""}</span>
              <Icon name="arrow" size={14} />
            </a>
          </li>
        ))}
      </ul>
      {runs.length <= 1 && <p className="muted small pad">Runs you analyze, or ones already in <span className="mono">outputs/</span>, show up here.</p>}
    </>
  );
}
