"""
Digbodha 1 - local API for the web UI. Standard library only, so it runs in the same
environment as analyze.py with nothing extra to install.

    python ui/server.py                 # http://127.0.0.1:8765
    python ui/server.py --port 9000

What it does
  GET  /api/health                 is the pipeline importable here (sam3, cv2)?
  GET  /api/runs                   every outputs/<name>/ that has a frames.csv
  GET  /api/runs/<name>/meta       fps, frame count, image size, where the pixels come from
  GET  /api/runs/<name>/frames.csv the pipeline's per-object-per-frame output
  GET  /api/runs/<name>/video      the source video (HTTP Range supported, so seeking works)
  GET  /api/runs/<name>/frame/<i>  frame i as JPG, for runs made from a folder of frames
  POST /api/analyze?name=&prompts=&chunk=&overlap=&max_objects=   body = the raw video file
  GET  /api/jobs/<id>              status, progress and the log tail of an analyze.py run

If ui/dist exists (npm run build), the built UI is served at / as well, so one command
is enough for a demo. During development run `npm run dev` in ui/ instead; Vite proxies /api here.
"""
import argparse
import csv
import glob
import importlib.util
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

UI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(UI_DIR)
OUTPUTS = os.path.join(REPO, "outputs")
UPLOADS = os.path.join(OUTPUTS, "_uploads")
DIST = os.path.join(UI_DIR, "dist")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

# Windows reads MIME types from the registry, which can map .js to text/plain and break the UI.
for _ext, _type in {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".csv": "text/csv",
                    ".json": "application/json", ".mp4": "video/mp4", ".webm": "video/webm"}.items():
    mimetypes.add_type(_type, _ext)

JOBS = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------- runs on disk

def run_dir(name):
    if not NAME_RE.match(name) or name.startswith("."):
        return None
    d = os.path.join(OUTPUTS, name)
    return d if os.path.isfile(os.path.join(d, "frames.csv")) else None


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def source_of(d):
    """Where this run's pixels are: ('video', path) | ('frames', [jpg paths]) | (None, None)."""
    run = read_json(os.path.join(d, "run.json")) or {}
    src = run.get("video")
    candidates = []
    if src:
        candidates += [src, os.path.join(REPO, src)]
    for c in candidates:
        if os.path.isfile(c) and c.lower().endswith(VIDEO_EXT):
            return "video", c
        if os.path.isdir(c):
            jpgs = sorted(glob.glob(os.path.join(c, "*.jpg")))
            if jpgs:
                return "frames", jpgs
    jpgs = sorted(glob.glob(os.path.join(d, "frames", "*.jpg")))  # extracted by analyze.py
    if jpgs:
        return "frames", jpgs
    return None, None


def image_size(kind, src, d):
    """(width, height) of the frames analyze.py measured boxes in."""
    try:
        import cv2
        if kind == "frames":
            img = cv2.imread(src[0])
            if img is not None:
                return img.shape[1], img.shape[0]
        if kind == "video":
            cap = cv2.VideoCapture(src)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w and h:
                return w, h
    except ImportError:
        pass
    w = h = 0  # fall back to the largest box seen
    with open(os.path.join(d, "frames.csv")) as f:
        for r in csv.DictReader(f):
            w, h = max(w, float(r["bx1"] or 0)), max(h, float(r["by1"] or 0))
    return int(w) or 1242, int(h) or 375


def run_meta(name):
    d = run_dir(name)
    if d is None:
        return None
    run = read_json(os.path.join(d, "run.json")) or {}
    kind, src = source_of(d)
    w, h = image_size(kind, src, d)
    n = run.get("frames")
    if n is None:
        with open(os.path.join(d, "frames.csv")) as f:
            n = 1 + max((int(r["frame"]) for r in csv.DictReader(f)), default=-1)
    return {
        "name": name,
        "prompts": run.get("prompts", ""),
        "fps": run.get("fps") or 10,
        "frames": n,
        "width": w,
        "height": h,
        "seconds_total": run.get("seconds_total"),
        "source": {"video": "video", "frames": "frames"}.get(kind, "none"),
        "modified": os.path.getmtime(os.path.join(d, "frames.csv")),
    }


def list_runs():
    if not os.path.isdir(OUTPUTS):
        return []
    out = []
    for name in sorted(os.listdir(OUTPUTS)):
        if run_dir(name):
            run = read_json(os.path.join(OUTPUTS, name, "run.json")) or {}
            out.append({"name": name, "prompts": run.get("prompts", ""), "frames": run.get("frames"),
                        "fps": run.get("fps"),
                        "modified": os.path.getmtime(os.path.join(OUTPUTS, name, "frames.csv"))})
    out.sort(key=lambda r: -r["modified"])
    return out


# ---------------------------------------------------------------- analyze.py jobs

PROGRESS_TOTAL = re.compile(r"\[segment\] (\d+) frames")
PROGRESS_CHUNK = re.compile(r"\[segment\] chunk \d+: frames (\d+)-(\d+)")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def start_job(name, video_path, prompts, chunk, overlap, max_objects):
    job_id = uuid.uuid4().hex[:10]
    cmd = [sys.executable, "-u", os.path.join(REPO, "analyze.py"), "--video", video_path,
           "--prompts", prompts, "--out", os.path.join(OUTPUTS, name), "--no_video",
           "--chunk", str(chunk), "--overlap", str(overlap), "--max_objects", str(max_objects)]
    job = {"id": job_id, "name": name, "status": "running", "stage": "starting", "progress": 0.0,
           "lines": [], "started": time.time(), "ended": None, "returncode": None}
    with JOBS_LOCK:
        JOBS[job_id] = job

    def work():
        total = None
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "PYTHON_COLORS": "0"}
            p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, env=env, errors="replace")
        except OSError as e:
            job.update(status="failed", lines=[str(e)], ended=time.time())
            return
        for line in p.stdout:
            line = ANSI.sub("", line.rstrip())
            if "\r" in line:  # tqdm-style progress bars redraw with carriage returns
                line = line.split("\r")[-1]
            with JOBS_LOCK:
                job["lines"] = (job["lines"] + [line])[-400:]
                if "[analyze] extracted" in line:
                    job["stage"] = "frames extracted"
                elif "SAM 3.1 ready" in line:
                    job["stage"] = "model loaded"
                m = PROGRESS_TOTAL.search(line)
                if m:
                    total = int(m.group(1))
                    job["stage"] = "segmenting"
                m = PROGRESS_CHUNK.search(line)
                if m and total:
                    job["progress"] = min(0.99, (int(m.group(2)) + 1) / total)
                if line.startswith("=== Digbodha"):
                    job["stage"] = "writing outputs"
        p.wait()
        with JOBS_LOCK:
            job.update(returncode=p.returncode, ended=time.time(),
                       status="done" if p.returncode == 0 else "failed",
                       progress=1.0 if p.returncode == 0 else job["progress"])

    threading.Thread(target=work, daemon=True).start()
    return job


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "Digbodha/1"

    def log_message(self, fmt, *args):
        if "/api/jobs/" not in self.path:  # the UI polls jobs; keep the console readable
            sys.stderr.write("[ui] " + fmt % args + "\n")

    # helpers
    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def fail(self, status, msg):
        self.send_json({"error": msg}, status)

    def send_file(self, path, ctype=None, cache="no-cache"):
        try:
            size = os.path.getsize(path)
        except OSError:
            return self.fail(404, "not found")
        ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):  # suffix range: last N bytes
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                buf = f.read(min(1 << 16, left))
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):
                    return  # the browser cancelled a range request while seeking
                left -= len(buf)

    # routes
    def do_GET(self):
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]

        if parts[:1] != ["api"]:
            return self.serve_ui(url.path)
        if parts == ["api", "health"]:
            return self.send_json({
                "ok": True,
                "python": sys.version.split()[0],
                "sam3": importlib.util.find_spec("sam3") is not None,
                "cv2": importlib.util.find_spec("cv2") is not None,
                "outputs": OUTPUTS,
            })
        if parts == ["api", "runs"]:
            return self.send_json(list_runs())
        if len(parts) >= 4 and parts[1] == "runs":
            name, what = parts[2], parts[3]
            d = run_dir(name)
            if d is None:
                return self.fail(404, f"no run named {name!r} in outputs/")
            if what == "meta":
                return self.send_json(run_meta(name))
            if what == "frames.csv":
                return self.send_file(os.path.join(d, "frames.csv"), "text/csv")
            kind, src = source_of(d)
            if what == "video" and kind == "video":
                return self.send_file(src)
            if what == "frame" and kind == "frames" and len(parts) == 5 and parts[4].isdigit():
                i = int(parts[4])
                if i < len(src):
                    return self.send_file(src[i], "image/jpeg", cache="max-age=3600")
            return self.fail(404, "not available for this run")
        if len(parts) == 3 and parts[1] == "jobs":
            with JOBS_LOCK:
                job = JOBS.get(parts[2])
                if job is None:
                    return self.fail(404, "no such job")
                return self.send_json({**job, "lines": job["lines"][-60:]})
        return self.fail(404, "unknown endpoint")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/analyze":
            return self.fail(404, "unknown endpoint")
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        filename = os.path.basename(q.get("filename", "upload.mp4"))
        ext = os.path.splitext(filename)[1].lower()
        if ext not in VIDEO_EXT:
            return self.fail(400, f"expected a video file ({', '.join(VIDEO_EXT)})")
        base = re.sub(r"[^A-Za-z0-9_-]+", "-", q.get("name") or os.path.splitext(filename)[0]).strip("-")
        name = (base or "run")[:60]
        if os.path.exists(os.path.join(OUTPUTS, name)):
            name = f"{name}-{time.strftime('%H%M%S')}"
        prompts = ",".join(p.strip() for p in q.get("prompts", "vehicle,person").split(",") if p.strip())
        try:
            chunk = int(q.get("chunk", 60))
            overlap = int(q.get("overlap", 2))
            max_objects = int(q.get("max_objects", 16))
        except ValueError:
            return self.fail(400, "chunk, overlap and max_objects must be integers")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self.fail(400, "empty upload")

        os.makedirs(UPLOADS, exist_ok=True)
        path = os.path.join(UPLOADS, name + ext)
        with open(path, "wb") as f:
            left = length
            while left > 0:
                buf = self.rfile.read(min(1 << 20, left))
                if not buf:
                    break
                f.write(buf)
                left -= len(buf)
        job = start_job(name, path, prompts or "vehicle,person", chunk, overlap, max_objects)
        self.send_json({"job": job["id"], "name": name}, 202)

    def serve_ui(self, path):
        if not os.path.isdir(DIST):
            return self.fail(404, "UI not built: run `npm run build` in ui/, or use `npm run dev`")
        rel = os.path.normpath(unquote(path).lstrip("/")) if path not in ("", "/") else "index.html"
        full = os.path.join(DIST, rel)
        if rel.startswith("..") or not os.path.isfile(full):
            full = os.path.join(DIST, "index.html")  # single-page app
        return self.send_file(full)


def main():
    ap = argparse.ArgumentParser(description="Digbodha 1 web UI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    os.makedirs(OUTPUTS, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[ui] Digbodha API on http://{args.host}:{args.port}  (runs from {OUTPUTS})")
    if os.path.isdir(DIST):
        print(f"[ui] serving the built UI at http://{args.host}:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
