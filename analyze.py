"""
Digbodha 1 - text-prompted motion and collision-risk estimation from a single camera.

    video + object names  ->  per-object tracks, motion, time to contact, collision risk

Pipeline (all 2D, one camera):
  1. SAM 3.1 segments and tracks every object matching the prompts      (src/segment.py)
     run in 60-frame chunks with object IDs linked across chunks
  2. camera motion from the background, each object's own motion       (src/egomotion.py)
  3. time to contact from image growth, collision course, risk label   (src/ttc.py)

Outputs (in --out):
  frames.csv     one row per object per frame: position, motion, TTC, miss distance, risk
  tracks.json    one summary per object: label, lifetime, lowest TTC, worst risk, ...
  annotated.mp4  the video with masks coloured by risk and labels (unless --no_video)
  run.json       settings and timing of this run

Usage (repo root, cvproj env):
  python analyze.py --video data/kitti_0000_100 --fps 10
  python analyze.py --video clip.mp4 --prompts "vehicle,person,dog" --out outputs/clip
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.egomotion import EgoMotion  # noqa: E402
from src.segment import build_predictor, segment_video  # noqa: E402
from src.ttc import RISK_ORDER, TTCEstimator  # noqa: E402

RISK_BGR = {"safe": (80, 180, 60), "uncertain": (200, 120, 200), "caution": (0, 170, 255),
            "danger": (40, 40, 230)}
CSV_COLS = ["frame", "time_s", "id", "label", "cx", "cy", "bx0", "by0", "bx1", "by1", "size_px", "zone",
            "raw_dx", "raw_dy", "own_dx", "own_dy", "motion", "expansion_per_s",
            "ttc_s", "ttc_lo", "ttc_hi", "miss_widths", "on_course", "risk", "reliable", "reasons"]


def video_to_frames(path, out_dir):
    """Extract an .mp4 to numbered JPGs (SAM 3.1's folder loader reads .jpg). Returns fps."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    cap.release()
    print(f"[analyze] extracted {i} frames at {fps:.1f} fps to {out_dir}")
    return fps


def annotate(img, objects, est, ego_res, frame_idx, fps):
    overlay = img.copy()
    for o in objects:
        e = est.get(o["id"])
        if e is None:
            continue
        overlay[o["mask"]] = RISK_BGR[e["risk"]]
    img = cv2.addWeighted(overlay, 0.40, img, 0.60, 0)
    for o in objects:
        e = est.get(o["id"])
        if e is None:
            continue
        ys, xs = np.nonzero(o["mask"])
        if not len(xs):
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        ttc = f"{e['ttc_s']:.1f}s" if e["ttc_s"] is not None else "-"
        course = " on course" if e["on_course"] else ""
        tag = "" if e["reliable"] else " ?"
        text = f"#{o['id']} {o['label']} TTC {ttc}{course}{tag}"
        color = RISK_BGR[e["risk"]]
        cv2.putText(img, text, (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, text, (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    worst = max((est[o["id"]]["risk"] for o in objects if o["id"] in est),
                key=lambda r: RISK_ORDER[r], default="safe")
    cam = ego_res["camera"]
    header = (f"t={frame_idx / fps:5.1f}s  frame {frame_idx}  objects {len(objects)}  "
              f"camera {cam['shift_center']} px  ALERT: {worst.upper()}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RISK_BGR[worst], 1)
    return img


def main():
    ap = argparse.ArgumentParser(description="Digbodha 1: motion and collision risk from video")
    ap.add_argument("--video", required=True, help="folder of .jpg frames, or an .mp4 file")
    ap.add_argument("--prompts", default="vehicle,person", help="comma-separated object names")
    ap.add_argument("--fps", type=float, default=None, help="frame rate (needed for a folder; KITTI = 10)")
    ap.add_argument("--out", default=None, help="output folder (default outputs/<video name>)")
    ap.add_argument("--max_objects", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--no_video", action="store_true", help="skip writing annotated.mp4")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(os.path.normpath(args.video)))[0]
    out = args.out or os.path.join("outputs", name)
    os.makedirs(out, exist_ok=True)

    frames_dir, fps = args.video, args.fps
    if args.video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        frames_dir = os.path.join(out, "frames")
        vid_fps = video_to_frames(args.video, frames_dir)
        fps = fps or vid_fps
    if fps is None:
        sys.exit("--fps is required when --video is a folder of frames (KITTI: --fps 10)")

    t_start = time.perf_counter()
    predictor = build_predictor(max_objects=args.max_objects, low_mem=True)
    ego = EgoMotion()
    ttc = TTCEstimator(fps=fps)
    chunk_stats, tracks, writer = [], {}, None

    with open(os.path.join(out, "frames.csv"), "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=CSV_COLS)
        w.writeheader()
        for fr in segment_video(predictor, frames_dir, args.prompts, chunk=args.chunk,
                                overlap=args.overlap, stats=chunk_stats):
            img = cv2.imread(fr["path"])
            ego_res = ego.step(img, fr["objects"])
            est = ttc.update(fr["frame"], fr["objects"], ego_res)

            for o in fr["objects"]:
                e, m = est.get(o["id"]), ego_res["objects"].get(o["id"], {})
                if e is None:
                    continue
                c = m.get("centroid") or (None, None)
                raw, own = m.get("raw_flow") or (None, None), m.get("own_flow") or (None, None)
                w.writerow({"frame": fr["frame"], "time_s": round(fr["frame"] / fps, 3),
                            "id": o["id"], "label": o["label"], "cx": c[0], "cy": c[1],
                            "bx0": e["box"][0], "by0": e["box"][1], "bx1": e["box"][2], "by1": e["box"][3],
                            "size_px": e["size_px"], "zone": e["zone"],
                            "raw_dx": raw[0], "raw_dy": raw[1], "own_dx": own[0], "own_dy": own[1],
                            "motion": e["motion"], "expansion_per_s": e["expansion_per_s"],
                            "ttc_s": e["ttc_s"], "ttc_lo": e["ttc_lo"], "ttc_hi": e["ttc_hi"],
                            "miss_widths": e["miss_widths"], "on_course": e["on_course"],
                            "risk": e["risk"], "reliable": e["reliable"],
                            "reasons": ";".join(e["reasons"])})

                tr = tracks.setdefault(o["id"], {
                    "id": o["id"], "label": o["label"], "first_frame": fr["frame"],
                    "frames_seen": 0, "min_ttc_s": None, "min_ttc_frame": None,
                    "worst_risk": "safe", "frames_uncertain": 0, "frames_caution": 0, "frames_danger": 0,
                    "ever_on_course": False, "_motion": Counter(), "_zone": Counter(),
                    "_reliable": 0})
                tr["last_frame"] = fr["frame"]
                tr["frames_seen"] += 1
                tr["_motion"][e["motion"]] += 1
                tr["_zone"][e["zone"]] += 1
                tr["_reliable"] += int(e["reliable"])
                if e["risk"] == "uncertain":
                    tr["frames_uncertain"] += 1
                if e["risk"] == "caution":
                    tr["frames_caution"] += 1
                if e["risk"] == "danger":
                    tr["frames_danger"] += 1
                if RISK_ORDER[e["risk"]] > RISK_ORDER[tr["worst_risk"]]:
                    tr["worst_risk"] = e["risk"]
                if e["on_course"]:
                    tr["ever_on_course"] = True
                if e["ttc_s"] is not None and e["reliable"] and \
                        (tr["min_ttc_s"] is None or e["ttc_s"] < tr["min_ttc_s"]):
                    tr["min_ttc_s"], tr["min_ttc_frame"] = e["ttc_s"], fr["frame"]

            if not args.no_video:
                vis = annotate(img, fr["objects"], est, ego_res, fr["frame"], fps)
                if writer is None:
                    h, wid = vis.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(out, "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (wid, h))
                writer.write(vis)

    if writer is not None:
        writer.release()

    summary = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        tr["main_motion"] = tr.pop("_motion").most_common(1)[0][0]
        tr["main_zone"] = tr.pop("_zone").most_common(1)[0][0]
        tr["reliable_fraction"] = round(tr.pop("_reliable") / tr["frames_seen"], 2)
        summary.append(tr)
    with open(os.path.join(out, "tracks.json"), "w") as f:
        json.dump(summary, f, indent=2)

    total = time.perf_counter() - t_start
    n_frames = sum(s["end"] - s["start"] for s in chunk_stats) - args.overlap * max(len(chunk_stats) - 1, 0)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump({"video": args.video, "prompts": args.prompts, "fps": fps,
                   "max_objects": args.max_objects, "chunk": args.chunk, "overlap": args.overlap,
                   "frames": n_frames, "seconds_total": round(total, 1),
                   "chunks": chunk_stats}, f, indent=2)

    print(f"\n=== Digbodha 1: {n_frames} frames in {total:.0f} s ({n_frames / total:.2f} FPS incl. model load) ===")
    print(f"{'id':>4} {'label':<9} {'frames':>11} {'motion':<13} {'zone':<7} {'min TTC':>8} "
          f"{'worst':<8} {'danger':>6} {'on course':>9} {'reliable':>8}")
    for tr in summary:
        mt = f"{tr['min_ttc_s']:.1f}s" if tr["min_ttc_s"] is not None else "-"
        print(f"{tr['id']:>4} {tr['label']:<9} {tr['first_frame']:>5}-{tr['last_frame']:<5} "
              f"{tr['main_motion']:<13} {tr['main_zone']:<7} {mt:>8} {tr['worst_risk']:<8} "
              f"{tr['frames_danger']:>6} {str(tr['ever_on_course']):>9} {tr['reliable_fraction']:>8}")
    print(f"\noutputs in {out}/  (frames.csv, tracks.json, run.json"
          f"{'' if args.no_video else ', annotated.mp4'})")


if __name__ == "__main__":
    main()