"""
Digbodha 1 - test ego-motion compensation (src/egomotion.py) on real video.

Runs the chunked SAM 3.1 segmentation, then for every frame estimates the camera's motion
from the background and each object's own motion with the camera removed.

Outputs (in --out):
  camera.csv   per frame: camera shift at the image centre, RANSAC inliers, reliable flag
  motion.csv   per frame per object: centroid, raw flow, own (compensated) flow, method
  frame_XXXXX.jpg overlays: grey arrow = raw motion, coloured arrow = own motion (x5)

Usage (repo root, cvproj env):
  python scripts/test_egomotion.py                                   # Meta sample, still camera
  python scripts/test_egomotion.py --video data/kitti_0000_100 --prompts "car,person" \
      --out outputs/ego_kitti                                         # moving camera
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
from src.egomotion import EgoMotion  # noqa: E402
from src.segment import build_predictor, segment_video  # noqa: E402

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128)]


def draw(img, fr, res, arrow_scale, out_path):
    overlay = img.copy()
    for o in fr["objects"]:
        overlay[o["mask"]] = PALETTE[o["id"] % len(PALETTE)]
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for o in fr["objects"]:
        m = res["objects"].get(o["id"])
        if not m or m["centroid"] is None:
            continue
        color = PALETTE[o["id"] % len(PALETTE)]
        cx, cy = m["centroid"]
        p0 = (int(cx), int(cy))
        if m["raw_flow"] is not None:
            rx, ry = m["raw_flow"]
            cv2.arrowedLine(img, p0, (int(cx + rx * arrow_scale), int(cy + ry * arrow_scale)),
                            (160, 160, 160), 2, tipLength=0.25)
            ox, oy = m["own_flow"]
            cv2.arrowedLine(img, p0, (int(cx + ox * arrow_scale), int(cy + oy * arrow_scale)),
                            color, 3, tipLength=0.3)
        tag = "" if m["reliable"] else " ?"
        cv2.putText(img, f"#{o['id']} {o['label']}{tag}", (p0[0] + 6, p0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cam = res["camera"]
    cv2.putText(img, f"frame {fr['frame']}  camera shift {cam['shift_center']} px  "
                     f"inliers {cam['inliers']}{'' if cam['reliable'] else ' (unreliable)'}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(out_path, img)


def main():
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(sam3_root, "assets", "videos", "0001"))
    ap.add_argument("--prompts", default="person")
    ap.add_argument("--out", default="outputs/ego_test")
    ap.add_argument("--save_every", type=int, default=15)
    ap.add_argument("--arrow_scale", type=float, default=5.0, help="arrow length = flow x this")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    predictor = build_predictor()
    ego = EgoMotion()
    cam_rows, obj_rows = [], []
    t0 = time.perf_counter()

    for fr in segment_video(predictor, args.video, args.prompts):
        img = cv2.imread(fr["path"])
        t1 = time.perf_counter()
        res = ego.step(img, fr["objects"])
        ego_ms = (time.perf_counter() - t1) * 1000

        cam = res["camera"]
        cam_rows.append([fr["frame"], *cam["shift_center"], cam["inliers"], cam["inlier_ratio"],
                         int(cam["reliable"]), round(ego_ms, 1)])
        labels = {o["id"]: o["label"] for o in fr["objects"]}
        for oid, m in res["objects"].items():
            if m["own_flow"] is None:
                continue
            obj_rows.append([fr["frame"], oid, labels[oid], *m["centroid"], *m["raw_flow"],
                             *m["own_flow"], m["method"], m["n_points"], int(m["reliable"])])
        if fr["frame"] % args.save_every == 0:
            draw(img, fr, res, args.arrow_scale, os.path.join(args.out, f"frame_{fr['frame']:05d}.jpg"))

    total = time.perf_counter() - t0
    with open(os.path.join(args.out, "camera.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "shift_x", "shift_y", "inliers", "inlier_ratio", "reliable", "ego_ms"])
        w.writerows(cam_rows)
    with open(os.path.join(args.out, "motion.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "id", "label", "cx", "cy", "raw_dx", "raw_dy", "own_dx", "own_dy",
                    "method", "n_points", "reliable"])
        w.writerows(obj_rows)

    shifts = np.array([[r[1], r[2]] for r in cam_rows[1:]]) if len(cam_rows) > 1 else np.zeros((1, 2))
    rel = np.mean([r[5] for r in cam_rows[1:]]) if len(cam_rows) > 1 else 0
    ms = np.mean([r[6] for r in cam_rows])
    print(f"\n=== Ego-motion summary ({len(cam_rows)} frames, {total:.0f} s total) ===")
    print(f"  camera shift per frame: mean |{np.linalg.norm(shifts, axis=1).mean():.2f}| px, "
          f"max |{np.linalg.norm(shifts, axis=1).max():.2f}| px")
    print(f"  camera estimate reliable in {rel * 100:.0f}% of frames")
    print(f"  ego-motion cost: {ms:.1f} ms per frame (SAM 3.1 is ~480 ms)")
    if obj_rows:
        ids = sorted({r[1] for r in obj_rows})
        print("  per object: mean |raw| vs mean |own| flow (px/frame), share from points method")
        for oid in ids:
            rows = [r for r in obj_rows if r[1] == oid]
            raw = np.mean([np.hypot(r[5], r[6]) for r in rows])
            own = np.mean([np.hypot(r[7], r[8]) for r in rows])
            pts = np.mean([r[9] == "points" for r in rows])
            print(f"    #{oid:<3} {rows[0][2]:<10} raw {raw:6.2f}  own {own:6.2f}  points {pts * 100:3.0f}%")
    print(f"  outputs in {args.out}/ (camera.csv, motion.csv, overlays)")


if __name__ == "__main__":
    main()