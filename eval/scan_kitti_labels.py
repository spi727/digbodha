"""
Digbodha 1 - rank KITTI tracking sequences by how many genuinely risky moments they contain.

Uses only the label files (no images), with the same ground-truth TTC / miss / risk rules as
eval/evaluate_kitti.py, so the evaluation can run on sequences that actually test collision
warnings instead of quiet ones.

Usage (repo root):
  python eval/scan_kitti_labels.py --labels_dir ~/datasets/kitti_tracking/training/label_02
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_kitti import gt_per_frame, load_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max_depth", type=float, default=40.0)
    ap.add_argument("--first_n", type=int, default=0, help="only count the first N frames (0 = all)")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(args.labels_dir), "*.txt"))):
        seq = os.path.splitext(os.path.basename(path))[0]
        gt = gt_per_frame(load_labels(path), args.fps)
        frames = sorted(gt)
        if args.first_n:
            frames = [f for f in frames if f < args.first_n]
        objs = [g for f in frames for g in gt[f] if 0 < g["z"] <= args.max_depth]
        appr = [g for g in objs if g["approaching"]]
        near = [g for g in appr if g["z"] <= 20]
        caution = sum(g["risk"] == "caution" for g in objs)
        danger = sum(g["risk"] == "danger" for g in objs)
        course = sum(bool(g["on_course"]) for g in objs)
        types = {}
        for g in appr:
            types[g["type"]] = types.get(g["type"], 0) + 1
        n_frames = (max(frames) + 1) if frames else 0
        rows.append((caution + 3 * danger, seq, n_frames, len(objs), len(appr), len(near),
                     course, caution, danger, types))

    rows.sort(reverse=True)
    print(f"{'seq':<6}{'frames':>7}{'objs':>7}{'approach':>9}{'near<20m':>9}{'on course':>10}"
          f"{'caution':>8}{'danger':>7}  approaching types")
    for score, seq, n, o, a, nr, c, ca, d, types in rows:
        t = ", ".join(f"{k}:{v}" for k, v in sorted(types.items(), key=lambda kv: -kv[1]))
        print(f"{seq:<6}{n:>7}{o:>7}{a:>9}{nr:>9}{c:>10}{ca:>8}{d:>7}  {t}")
    print("\n(counts are object-frames within", args.max_depth, "m; risky = caution + danger by the GT rules)")


if __name__ == "__main__":
    main()