"""
Digbodha 1 - evaluate against KITTI Tracking ground truth.

KITTI labels give every object's 3D position in camera coordinates (x right, y down, z forward,
in metres) in every frame. From those we compute the TRUE quantities Digbodha estimates from 2D:

  true TTC      = Z / (-dZ/dt)          (depth over closing speed; inf if not closing)
  true miss     = X at contact / H, with X extrapolated linearly to the contact time
                  (same units and rule as src/ttc.py: offset in object heights)
  true risk     = same thresholds as src/ttc.py applied to the true values

dZ/dt and dX/dt come from a least-squares line over a short window centred on each frame.

Our objects are matched to KITTI objects frame by frame by 2D box overlap (IoU >= 0.5,
Hungarian), within compatible classes (vehicle <-> Car/Van/Truck/Tram,
person <-> Pedestrian/Person_sitting/Cyclist).

Usage (repo root):
  python eval/evaluate_kitti.py --pred outputs/kitti_0000_v3/frames.csv \
      --labels ~/datasets/kitti_tracking/training/label_02/0000.txt --fps 10 --out outputs/eval_0000
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

CLASS_MAP = {"vehicle": {"Car", "Van", "Truck", "Tram"},
             "car": {"Car", "Van"},
             "person": {"Pedestrian", "Person_sitting", "Cyclist"}}
RISKS = ["safe", "uncertain", "caution", "danger"]


# ------------------------------------------------------------------ ground truth
def load_labels(path):
    """KITTI tracking label file -> {track_id: [(frame, type, box, dims(h,w,l), loc(x,y,z)), ...]}"""
    tracks = defaultdict(list)
    with open(os.path.expanduser(path)) as f:
        for line in f:
            p = line.split()
            if len(p) < 17 or p[2] == "DontCare":
                continue
            frame, tid, typ = int(p[0]), int(p[1]), p[2]
            box = tuple(float(v) for v in p[6:10])
            h, w, l = (float(v) for v in p[10:13])
            x, y, z = (float(v) for v in p[13:16])
            tracks[tid].append((frame, typ, box, (h, w, l), (x, y, z)))
    for tid in tracks:
        tracks[tid].sort()
    return tracks


def gt_per_frame(tracks, fps, half_window=4, ttc_max=10.0, miss_thresh=1.0,
                 danger_ttc=2.0, caution_ttc=4.0, near_pass_ttc=1.5):
    """-> {frame: [gt object dict, ...]} with true TTC, miss and risk."""
    out = defaultdict(list)
    for tid, rows in tracks.items():
        frames = np.array([r[0] for r in rows])
        X = np.array([r[4][0] for r in rows])
        Z = np.array([r[4][2] for r in rows])
        for i, (fr, typ, box, (h, w, l), (x, y, z)) in enumerate(rows):
            sel = np.abs(frames - fr) <= half_window
            if sel.sum() < 3:
                continue
            t = frames[sel] / fps
            vz = np.polyfit(t, Z[sel], 1)[0]
            vx = np.polyfit(t, X[sel], 1)[0]
            closing = -vz
            ttc = z / closing if closing > 1e-3 and z > 0 else math.inf
            approaching = ttc < ttc_max
            miss, on_course, risk = None, None, "safe"
            if approaching:
                miss = (x + vx * ttc) / max(h, 0.1)
                on_course = abs(miss) < miss_thresh
                if on_course and ttc < danger_ttc:
                    risk = "danger"
                elif (on_course and ttc < caution_ttc) or ttc < near_pass_ttc:
                    risk = "caution"
            out[fr].append({"track": tid, "type": typ, "box": box, "z": z, "x": x,
                            "ttc": ttc, "approaching": approaching, "miss": miss,
                            "on_course": on_course, "risk": risk})
    return out


# ------------------------------------------------------------------ predictions
def load_pred(path):
    per_frame = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("bx0"):
                continue
            num = lambda k: float(r[k]) if r[k] not in ("", "None") else None
            per_frame[int(r["frame"])].append({
                "id": int(r["id"]), "label": r["label"],
                "box": (num("bx0"), num("by0"), num("bx1"), num("by1")),
                "ttc": num("ttc_s"), "miss": num("miss_widths"),
                "on_course": r["on_course"] == "True", "risk": r["risk"],
                "reliable": r["reliable"] == "True", "reasons": r["reasons"]})
    return per_frame


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thresh=0.5):
    if not preds or not gts:
        return []
    M = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        allowed = CLASS_MAP.get(p["label"], set())
        for j, g in enumerate(gts):
            if g["type"] in allowed:
                M[i, j] = iou(p["box"], g["box"])
    rows, cols = linear_sum_assignment(-M)
    return [(preds[i], gts[j], M[i, j]) for i, j in zip(rows, cols) if M[i, j] >= thresh]


# ------------------------------------------------------------------ metrics
def ttc_errors(pairs):
    """Pairs where BOTH true and predicted say approaching -> abs and relative error."""
    e = [(abs(p["ttc"] - g["ttc"]), abs(p["ttc"] - g["ttc"]) / g["ttc"], g["z"], p["reliable"])
         for p, g, _ in pairs if p["ttc"] is not None and g["approaching"]]
    return e


def summarise(errs):
    if not errs:
        return {"n": 0}
    a = np.array([x[0] for x in errs])
    r = np.array([x[1] for x in errs])
    return {"n": len(errs), "median_abs_s": round(float(np.median(a)), 2),
            "mean_abs_s": round(float(a.mean()), 2),
            "median_rel_pct": round(float(np.median(r)) * 100, 1),
            "within_20pct": round(float((r <= 0.2).mean()) * 100, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="frames.csv from analyze.py")
    ap.add_argument("--labels", required=True, help="KITTI label_02/<seq>.txt")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--max_depth", type=float, default=40.0, help="ignore GT objects farther than this (m)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt = gt_per_frame(load_labels(args.labels), args.fps)
    pred = load_pred(args.pred)
    frames = sorted(pred)

    pairs, n_gt, n_gt_matched = [], 0, 0
    for fr in frames:
        gts = [g for g in gt.get(fr, []) if 0 < g["z"] <= args.max_depth]
        m = match(pred[fr], gts)
        n_gt += len(gts)
        n_gt_matched += len(m)
        pairs += [(p, g, s) for p, g, s in m]
        for p, g, s in m:
            p["frame"] = fr

    with open(os.path.join(args.out, "matched.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "pred_id", "label", "gt_track", "gt_type", "iou", "gt_depth_m",
                    "gt_ttc", "pred_ttc", "gt_miss", "pred_miss", "gt_risk", "pred_risk", "reliable", "reasons"])
        for p, g, s in pairs:
            w.writerow([p["frame"], p["id"], p["label"], g["track"], g["type"], round(s, 2), round(g["z"], 1),
                        None if math.isinf(g["ttc"]) else round(g["ttc"], 2), p["ttc"],
                        None if g["miss"] is None else round(g["miss"], 2), p["miss"],
                        g["risk"], p["risk"], p["reliable"], p["reasons"]])

    # 1. TTC accuracy
    errs = ttc_errors(pairs)
    res = {"frames": len(frames), "gt_objects_within_depth": n_gt,
           "gt_matched_pct": round(100 * n_gt_matched / max(n_gt, 1), 1),
           "ttc_all": summarise(errs),
           "ttc_reliable": summarise([e for e in errs if e[3]]),
           "ttc_unreliable": summarise([e for e in errs if not e[3]]),
           "ttc_by_depth": {}}
    for lo, hi in [(0, 10), (10, 20), (20, 40)]:
        res["ttc_by_depth"][f"{lo}-{hi} m"] = summarise([e for e in errs if lo <= e[2] < hi and e[3]])

    # 2. approach detection (on matched objects)
    tp = sum(1 for p, g, _ in pairs if p["ttc"] is not None and g["approaching"])
    fp = sum(1 for p, g, _ in pairs if p["ttc"] is not None and not g["approaching"])
    fn = sum(1 for p, g, _ in pairs if p["ttc"] is None and g["approaching"])
    res["approach"] = {"tp": tp, "fp": fp, "fn": fn,
                       "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}

    # 3. risk confusion and alert precision/recall (alert = caution or danger)
    conf = {g: {p: 0 for p in RISKS} for g in ["safe", "caution", "danger"]}
    for p, g, _ in pairs:
        conf[g["risk"]][p["risk"]] += 1
    alert = lambda r: r in ("caution", "danger")
    a_tp = sum(1 for p, g, _ in pairs if alert(p["risk"]) and alert(g["risk"]))
    a_fp = sum(1 for p, g, _ in pairs if alert(p["risk"]) and not alert(g["risk"]))
    a_fn = sum(1 for p, g, _ in pairs if not alert(p["risk"]) and alert(g["risk"]))
    abstain = sum(1 for p, g, _ in pairs if p["risk"] == "uncertain" and alert(g["risk"]))
    res["risk_confusion_gt_rows_pred_cols"] = conf
    res["alerts"] = {"tp": a_tp, "fp": a_fp, "fn": a_fn,
                     "fn_flagged_uncertain": abstain,
                     "precision": round(a_tp / max(a_tp + a_fp, 1), 3),
                     "recall": round(a_tp / max(a_tp + a_fn, 1), 3)}

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== KITTI evaluation: {res['frames']} frames, {n_gt} GT objects within {args.max_depth:.0f} m, "
          f"{res['gt_matched_pct']}% matched ===")
    print("\nTime to contact (objects approaching in truth and in our estimate):")
    for k in ("ttc_all", "ttc_reliable", "ttc_unreliable"):
        print(f"  {k:<16} {res[k]}")
    print("  reliable, by true depth:")
    for k, v in res["ttc_by_depth"].items():
        print(f"    {k:<8} {v}")
    print(f"\nApproach detection: {res['approach']}")
    print("\nRisk (rows = truth, columns = ours):")
    print(f"  {'':10}" + "".join(f"{r:>10}" for r in RISKS))
    for g, row in conf.items():
        print(f"  {g:<10}" + "".join(f"{row[r]:>10}" for r in RISKS))
    print(f"\nAlerts (caution or danger): {res['alerts']}")
    print(f"\nwritten: {args.out}/metrics.json, {args.out}/matched.csv")


if __name__ == "__main__":
    main()