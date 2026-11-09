"""
Digbodha 1 - turn a video into frames, and the pipeline's frames.csv into labels.json.

labels.json holds every frame and every object with its motion, time to contact, collision
course, risk and reliability, plus a one-line English description per object, so it can be
turned directly into training text for a vision-language model.
"""
import csv
import os

import cv2

RISK_ORDER = {"safe": 0, "uncertain": 1, "caution": 2, "danger": 3}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_frames(video_path, out_dir, target_fps=10.0, max_seconds=0):
    """Video -> numbered JPG frames at about target_fps, first max_seconds only (0 = all).
    Returns (number of frames written, frame rate of the written frames)."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / target_fps)) if target_fps and target_fps > 0 else 1
    limit = int(max_seconds * src_fps) if max_seconds and max_seconds > 0 else None
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (limit is not None and i >= limit):
            break
        if i % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
        i += 1
    cap.release()
    return n, src_fps / step


def describe(o):
    """One plain-English line for one object in one frame."""
    side = {"left": "on the left", "right": "on the right", "centre": "ahead"}.get(o["zone"], "")
    s = f"{o['label']} #{o['id']} {side}".strip()
    motion = (o.get("motion") or "").replace("_", " ")
    if motion and motion != "unknown":
        s += f", {motion}"
    if o["approaching"]:
        s += f", approaching, contact in about {o['ttc_s']:.1f} s"
        if o["on_course"] is True:
            s += ", on a collision course"
        elif o["on_course"] is False:
            s += f", will pass on the {'left' if (o['miss'] or 0) < 0 else 'right'}"
    else:
        s += ", not approaching"
    s += f" -> {o['risk']}"
    if not o["reliable"] and o["reasons"]:
        s += f" (unsure: {', '.join(o['reasons'])})"
    return s


def build_labels(frames_csv, meta=None):
    """frames.csv from analyze.py -> {"meta", "frames": [{"frame", "time_s", "frame_risk", "objects": [...]}]}"""
    frames = {}
    with open(frames_csv) as f:
        for r in csv.DictReader(f):
            ttc = num(r["ttc_s"])
            o = {"id": int(r["id"]), "label": r["label"],
                 "box_xyxy": [int(float(r[k])) for k in ("bx0", "by0", "bx1", "by1")],
                 "zone": r["zone"], "motion": r["motion"],
                 "own_flow_px": [num(r["own_dx"]), num(r["own_dy"])],
                 "approaching": ttc is not None,
                 "ttc_s": ttc, "ttc_range_s": [num(r["ttc_lo"]), num(r["ttc_hi"])],
                 "miss": num(r["miss_widths"]),
                 "on_course": {"True": True, "False": False}.get(r["on_course"]),
                 "risk": r["risk"], "reliable": r["reliable"] == "True",
                 "reasons": [x for x in (r["reasons"] or "").split(";") if x]}
            o["text"] = describe(o)
            fr = frames.setdefault(int(r["frame"]), {"frame": int(r["frame"]), "time_s": num(r["time_s"]),
                                                     "objects": []})
            fr["objects"].append(o)
    for fr in frames.values():
        fr["frame_risk"] = max((o["risk"] for o in fr["objects"]), key=RISK_ORDER.get, default="safe")
    meta = dict(meta or {})
    meta.setdefault("units", {"ttc_s": "seconds",
                              "miss": "sideways offset at contact, in object heights (|miss| < 1 = on course)",
                              "own_flow_px": "pixels per frame, camera motion removed"})
    meta.setdefault("risk_rules", {"danger": "on course and TTC < 2 s",
                                   "caution": "on course and TTC < 4 s, or TTC < 1.5 s",
                                   "uncertain": "would alert, but the estimate failed a reliability check"})
    return {"meta": meta, "frames": [frames[k] for k in sorted(frames)]}
