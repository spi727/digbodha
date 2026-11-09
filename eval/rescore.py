"""
Digbodha 1 - re-run ONLY the time-to-contact / risk stage on saved results (no SAM, no GPU).

frames.csv from analyze.py already holds, per object per frame, the box of its cleaned mask and its
raw and ego-compensated motion: everything src/ttc.py needs. So a change to the TTC/risk logic can
be evaluated in seconds instead of re-running segmentation.

Usage (repo root):
  python eval/rescore.py --pred outputs/kitti_0013/frames.csv --frames data/kitti_0013_340 \
      --fps 10 --out outputs/kitti_0013_rescored/frames.csv
  # options passed to the estimator, e.g.:  --debounce 2  (turn debouncing on)
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.ttc import TTCEstimator  # noqa: E402

COLS = ["frame", "time_s", "id", "label", "cx", "cy", "bx0", "by0", "bx1", "by1", "size_px", "zone",
        "raw_dx", "raw_dy", "own_dx", "own_dy", "motion", "expansion_per_s",
        "ttc_s", "ttc_lo", "ttc_hi", "miss_widths", "on_course", "risk", "reliable", "reasons"]


def num(v):
    return float(v) if v not in ("", "None", None) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="frames.csv from analyze.py")
    ap.add_argument("--frames", required=True, help="the JPG frame folder (only to read the image size)")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", required=True, help="output frames.csv")
    ap.add_argument("--debounce", type=int, default=1)
    ap.add_argument("--turn_px", type=float, default=5.0)
    args = ap.parse_args()

    first = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))[0]
    img_h, img_w = cv2.imread(first).shape[:2]

    by_frame = defaultdict(list)
    with open(args.pred) as f:
        for r in csv.DictReader(f):
            by_frame[int(r["frame"])].append(r)

    est = TTCEstimator(fps=args.fps, debounce=args.debounce, turn_px=args.turn_px)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for fr in sorted(by_frame):
            rows = by_frame[fr]
            items = []
            for r in rows:
                raw = (num(r["raw_dx"]), num(r["raw_dy"])) if num(r["raw_dx"]) is not None else None
                own = (num(r["own_dx"]), num(r["own_dy"])) if num(r["own_dx"]) is not None else None
                items.append({"id": int(r["id"]), "label": r["label"],
                              "box": (int(r["bx0"]), int(r["by0"]), int(r["bx1"]), int(r["by1"])),
                              "raw_flow": raw, "own_flow": own})
            res = est.update_boxes(fr, items, img_w, img_h)
            for r in rows:
                e = res[int(r["id"])]
                r.update({"size_px": e["size_px"], "zone": e["zone"], "motion": e["motion"],
                          "expansion_per_s": e["expansion_per_s"], "ttc_s": e["ttc_s"],
                          "ttc_lo": e["ttc_lo"], "ttc_hi": e["ttc_hi"], "miss_widths": e["miss_widths"],
                          "on_course": e["on_course"], "risk": e["risk"], "reliable": e["reliable"],
                          "reasons": ";".join(e["reasons"])})
                w.writerow({k: r.get(k) for k in COLS})
                n += 1
    print(f"re-scored {n} object-frames ({len(by_frame)} frames, image {img_w}x{img_h}) -> {args.out}")


if __name__ == "__main__":
    main()