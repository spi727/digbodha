# DigBodha web UI

A web front end for the pipeline: a landing page that explains the method, a frame-accurate run viewer, the KITTI evaluation, and a page to start new analyses.

It ships with the held-out **KITTI 0007** run (video and `frames.csv` from the results pack), so it works with no GPU and no backend.

## Run it

**Just the UI and the demo run** (Node 18+):

```bash
cd ui
npm install
npm run dev          # http://localhost:5173
```

**With your own runs and live analysis.** Start the API from the repo root in the `cvproj` environment. It's standard library only and needs nothing beyond what `analyze.py` already uses.

```bash
python ui/server.py  # http://127.0.0.1:8765
```

With both running, Vite proxies `/api` to the server. For a demo you can skip Node: run `npm run build` once, and `python ui/server.py` then serves the whole app at http://127.0.0.1:8765.

## What's in it

| Page | What it shows |
|---|---|
| **Landing** (`#/`) | Live hero player with overlays drawn from `frames.csv`, held-out numbers, the three stages, an interactive looming → TTC explainer, and the four risk states. |
| **Viewer** (`#/run/<name>`) | Video with risk-coloured boxes (click to select), transport with 0.25×/0.5×/1× speed, and a risk timeline with one lane per tracked object. Also the *contact plane* (predicted miss against TTC, so the risk rules become regions), the objects in view with their reliability flags, and a per-object panel with TTC and miss-distance history. |
| **Evaluation** (`#/evaluation`) | Per-sequence metrics for tuning vs held-out, reliable vs flagged TTC error, the risk confusion matrix, and TTC error by distance. |
| **Analyze** (`#/analyze`) | Upload a video, choose prompts and chunking, and watch `analyze.py` progress with its log. When it finishes, the results open in the viewer. |

Keyboard shortcuts in the viewer: <kbd>Space</kbd> play/pause, <kbd>←</kbd>/<kbd>→</kbd> step one frame (add <kbd>Shift</kbd> for 10), <kbd>A</kbd> next alert, <kbd>Esc</kbd> deselect.

## How runs are found

The server lists every `outputs/<name>/` that contains a `frames.csv`. It finds the pixels from `run.json`'s `video` field (an MP4, or a folder of JPGs), or from `outputs/<name>/frames/`, which `analyze.py` writes for MP4 inputs. Box coordinates are drawn in the source image's pixel space, so overlays line up at any display size.

## Layout

```text
ui/
├── server.py            # stdlib HTTP API + static server for the built UI
├── public/demo/         # KITTI 0007: video, frames.csv, eval metrics (CC BY-NC-SA 3.0)
└── src/
    ├── lib/             # data loading (CSV → runs/tracks), playback, formatting
    ├── components/      # Stage (video + canvas overlay), Timeline, ContactPlane, TrackChart, Looming
    └── pages/           # Landing, Viewer, Evaluation, Analyze
```

React 18 + TypeScript + Vite, with no UI or chart libraries. Charts are hand-built SVG and the overlay is a canvas.
