"""
Digbodha 1 - test the chunked SAM 3.1 runner (src/segment.py) on a long clip.

Checks three things:
  1. No crash and steady speed on a clip longer than one session can hold (270 frames).
  2. Object IDs stay the same across chunk seams (tracks.csv + overlays at the seams).
  3. Whether out_boxes_xywh is normalised (0-1) or in pixels.

Usage (repo root, cvproj env):
  python scripts/test_chunked.py
  python scripts/test_chunked.py --video data/kitti_0000 --prompts "car,person"
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # repo root
import sam3  # noqa: E402
from src.segment import build_predictor, chunk_ranges, list_frames, segment_video  # noqa: E402

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128)]


def draw(frame_rec, out_path):
    img = cv2.imread(frame_rec["path"])
    overlay = img.copy()
    for o in frame_rec["objects"]:
        color = PALETTE[o["id"] % len(PALETTE)]
        overlay[o["mask"]] = color
        ys, xs = np.nonzero(o["mask"])
        if len(xs):
            cv2.putText(img, f"#{o['id']} {o['label']}", (int(xs.min()), max(int(ys.min()) - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    cv2.putText(img, f"frame {frame_rec['frame']}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    cv2.imwrite(out_path, img)


def main():
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(sam3_root, "assets", "videos", "0001"))
    ap.add_argument("--prompts", default="person", help="comma-separated, e.g. 'car,person'")
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--out", default="outputs/chunked_test")
    ap.add_argument("--save_every", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    predictor = build_predictor(max_objects=args.max_objects, low_mem=True)
    print(f"model built in {time.perf_counter() - t0:.1f} s")

    stats, tracks = [], {}  # tracks: id -> {label, first, last, n}
    box_max, n_frames = 0.0, 0
    # frames on both sides of every seam: last frame from the old chunk, first from the new one
    ranges = chunk_ranges(len(list_frames(args.video)), args.chunk, args.overlap)
    seam_frames = set()
    for start, _ in ranges[1:]:
        seam_frames.update({start + args.overlap - 1, start + args.overlap})
    t0 = time.perf_counter()

    for fr in segment_video(predictor, args.video, args.prompts,
                            chunk=args.chunk, overlap=args.overlap, stats=stats):
        n_frames += 1
        for o in fr["objects"]:
            t = tracks.setdefault(o["id"], {"label": o["label"], "first": fr["frame"],
                                            "last": fr["frame"], "n": 0})
            t["last"], t["n"] = fr["frame"], t["n"] + 1
            box_max = max(box_max, max(o["box_xywh"]))
        if fr["frame"] % args.save_every == 0 or fr["frame"] in seam_frames:
            draw(fr, os.path.join(args.out, f"frame_{fr['frame']:05d}.jpg"))

    total = time.perf_counter() - t0
    print(f"\n=== Chunked run ===\n  frames: {n_frames}\n  total time (incl. all setup): {total:.1f} s"
          f"\n  overall FPS: {n_frames / total:.2f}")

    with open(os.path.join(args.out, "chunk_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    with open(os.path.join(args.out, "tracks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "first_frame", "last_frame", "frames_seen"])
        for gid in sorted(tracks):
            t = tracks[gid]
            w.writerow([gid, t["label"], t["first"], t["last"], t["n"]])

    print("\n=== Tracks (an ID spanning several chunks = linking worked) ===")
    for gid in sorted(tracks):
        t = tracks[gid]
        print(f"  #{gid:<3} {t['label']:<10} frames {t['first']:>4}-{t['last']:<4} seen {t['n']}")
    unit = "normalised (0-1)" if box_max <= 1.0 else "pixels"
    print(f"\n  out_boxes_xywh largest value: {box_max:.3f} -> {unit}")
    print(f"  outputs in {args.out}/ (chunk_stats.csv, tracks.csv, overlays incl. seam frames)")


if __name__ == "__main__":
    main()