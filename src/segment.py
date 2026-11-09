"""
Digbodha 1 - chunked SAM 3.1 segmentation with object IDs kept consistent across chunks.

Why chunks: SAM 3.1's GPU memory grows ~0.21-0.26 GB per 16 frames and never levels off,
so one session crashes an 8 GB GPU at about frame 92. We run 60-frame chunks (each a
fresh session), overlap them by a few frames, and link object IDs across each seam by
mask IoU with Hungarian matching.

Usage:
    from src.segment import build_predictor, segment_video
    predictor = build_predictor()
    for fr in segment_video(predictor, "data/kitti_0000", ["car", "person"]):
        fr["frame"], fr["objects"]   # objects: id, label, prob, box_xywh, mask
"""
import functools
import gc
import glob
import inspect
import os
import shutil
import tempfile
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sam3.model_builder import build_sam3_multiplex_video_predictor


# ----------------------------------------------------------------------------- model
def build_predictor(max_objects=8, low_mem=True, verbose=True):
    """Build SAM 3.1 once, with the fixes needed on an 8 GB laptop GPU."""
    predictor = build_sam3_multiplex_video_predictor(use_fa3=False, max_num_objects=max_objects)
    m = predictor.model

    # 1. init_state() rejects some session arguments (e.g. offload_state_to_cpu): drop them.
    original = m.init_state
    accepted = inspect.signature(original).parameters

    @functools.wraps(original)
    def init_state_compat(*a, **kw):
        kw = {k: v for k, v in kw.items() if k in accepted}
        if low_mem and "offload_video_to_cpu" in accepted:
            kw["offload_video_to_cpu"] = True  # decoded frames stay in system RAM
        return original(*a, **kw)

    m.init_state = init_state_compat

    # 2. The builder batches 16 frames through the detector: needs >13 GB. One at a time.
    m.use_batched_grounding = False
    m.batched_grounding_batch_size = 1

    # 3. Low-memory options that are off in Meta's demo config.
    if low_mem and hasattr(m, "tracker"):
        for opt in ("trim_past_non_cond_mem_for_eval", "offload_output_to_cpu_for_eval"):
            if hasattr(m.tracker, opt):
                setattr(m.tracker, opt, True)

    if verbose:
        print(f"[segment] SAM 3.1 ready: max_objects={max_objects}, low_mem={low_mem}")
    return predictor


# ----------------------------------------------------------------------------- helpers
def list_frames(folder):
    """Sorted list of .jpg frames in a folder (SAM 3.1's folder loader reads .jpg only)."""
    frames = glob.glob(os.path.join(folder, "*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames in {folder}")
    try:
        frames.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        frames.sort()
    return frames


def chunk_ranges(n_frames, chunk=60, overlap=2):
    """[(start, end), ...] covering 0..n_frames, consecutive chunks sharing `overlap` frames."""
    if overlap >= chunk:
        raise ValueError("overlap must be smaller than chunk")
    ranges, start = [], 0
    while True:
        end = min(start + chunk, n_frames)
        ranges.append((start, end))
        if end >= n_frames:
            return ranges
        start = end - overlap


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return (xs.mean(), ys.mean()) if len(xs) else None


def _shift_mask(mask, dx, dy):
    """Translate a boolean mask by (dx, dy) pixels; pixels shifted out are dropped."""
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    if xs0 < xs1 and ys0 < ys1:
        out[ys0:ys1, xs0:xs1] = mask[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _to_objects(out, label):
    """SAM 3.1 per-frame output -> list of plain dicts. Arrays are COPIED because
    SAM 3.1 reuses its CPU output buffers between frames."""
    ids = np.array(out["out_obj_ids"], copy=True)
    probs = np.array(out["out_probs"], copy=True)
    boxes = np.array(out["out_boxes_xywh"], copy=True)
    masks = np.array(out["out_binary_masks"], copy=True)
    return [
        {"local_id": int(ids[k]), "label": label, "prob": float(probs[k]),
         "box_xywh": boxes[k].tolist(), "mask": masks[k]}
        for k in range(len(ids))
    ]


# ----------------------------------------------------------------------------- one chunk
def _run_chunk(predictor, frame_paths, prompts):
    """Run SAM 3.1 on one chunk. Returns a list (one entry per frame) of object lists.
    Each prompt is tracked separately; local IDs are made unique per label as (label, id)."""
    tmp = tempfile.mkdtemp(prefix="digbodha_chunk_")
    per_frame = [[] for _ in frame_paths]
    try:
        for i, p in enumerate(frame_paths):
            os.symlink(os.path.abspath(p), os.path.join(tmp, f"{i:05d}.jpg"))

        sid = predictor.handle_request(
            request=dict(type="start_session", resource_path=tmp))["session_id"]
        try:
            for k, label in enumerate(prompts):
                if k > 0:  # a new text prompt needs a reset, otherwise results are wrong
                    predictor.handle_request(request=dict(type="reset_session", session_id=sid))
                predictor.handle_request(request=dict(
                    type="add_prompt", session_id=sid, frame_index=0, text=label))
                for r in predictor.handle_stream_request(
                        request=dict(type="propagate_in_video", session_id=sid)):
                    per_frame[r["frame_index"]].extend(_to_objects(r["outputs"], label))
        finally:
            predictor.handle_request(request=dict(type="close_session", session_id=sid))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return per_frame


# ----------------------------------------------------------------------------- ID linking
def _link_ids(prev_tail, new_head, next_id, iou_thresh=0.3):
    """Map this chunk's (label, local_id) to global IDs.

    prev_tail: overlap frames from the previous chunk, each {(label, gid): mask}
    new_head : the same frames from this chunk, each a list of objects (with local_id)
    Returns (mapping {(label, local_id): gid}, next_id, n_matched).
    """
    mapping, n_matched = {}, 0
    labels = {o["label"] for fr in new_head for o in fr}
    for label in labels:
        prev_ids = sorted({gid for fr in prev_tail for (lab, gid) in fr if lab == label})
        new_ids = sorted({o["local_id"] for fr in new_head for o in fr if o["label"] == label})
        if prev_ids and new_ids:
            # mean IoU over the overlap frames where BOTH objects are present
            # (a frame where one side missed the object is skipped, not counted as 0)
            iou_sum = np.zeros((len(prev_ids), len(new_ids)))
            count = np.zeros((len(prev_ids), len(new_ids)))
            for f_prev, f_new in zip(prev_tail, new_head):
                new_masks = {o["local_id"]: o["mask"] for o in f_new if o["label"] == label}
                for i, gid in enumerate(prev_ids):
                    m_prev = f_prev.get((label, gid))
                    if m_prev is None:
                        continue
                    for j, lid in enumerate(new_ids):
                        if lid in new_masks:
                            iou_sum[i, j] += mask_iou(m_prev, new_masks[lid])
                            count[i, j] += 1
            iou = np.divide(iou_sum, count, out=np.zeros_like(iou_sum), where=count > 0)
            rows, cols = linear_sum_assignment(-iou)
            for i, j in zip(rows, cols):
                if iou[i, j] >= iou_thresh:
                    mapping[(label, new_ids[j])] = prev_ids[i]
                    n_matched += 1
        for lid in new_ids:
            if (label, lid) not in mapping:
                mapping[(label, lid)] = next_id
                next_id += 1
    return mapping, next_id, n_matched


# ----------------------------------------------------------------------------- main entry
def segment_video(predictor, frames_dir, prompts, chunk=60, overlap=2,
                  iou_thresh=0.3, grace=10, stats=None, verbose=True, min_chunk=15):
    """Generator over frames in order. Each item:
        {"frame": global index, "path": jpg path,
         "objects": [{"id", "label", "prob", "box_xywh", "mask"}, ...]}
    If `stats` is a list, one dict per chunk is appended (timing, VRAM, ID matches).
    """
    if isinstance(prompts, str):
        prompts = [p.strip() for p in prompts.split(",") if p.strip()]
    paths = list_frames(frames_dir)
    n_frames = len(paths)
    if verbose:
        print(f"[segment] {n_frames} frames, chunks of <= {chunk} (overlap {overlap}, "
              f"halved on out-of-memory down to {min_chunk}), prompts {prompts}")

    # Chunks are planned on the fly: memory use depends on how many objects are tracked, so a
    # crowded scene can overflow the GPU at a chunk length that is safe for a quiet one. On
    # out-of-memory the same chunk is retried at half the length, and that length is kept.
    prev_tail, next_id = None, 1
    start, c, cur_chunk = 0, 0, chunk
    while start < n_frames:
        end = min(start + cur_chunk, n_frames)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            per_frame = _run_chunk(predictor, paths[start:end], prompts)
        except RuntimeError as err:  # torch.OutOfMemoryError and "CUDA driver error: out of memory"
            if "out of memory" not in str(err).lower() or cur_chunk <= min_chunk:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            new_len = max(min_chunk, cur_chunk // 2)
            if verbose:
                print(f"[segment] out of GPU memory on frames {start}-{end - 1} "
                      f"({cur_chunk}-frame chunk) -> retrying with {new_len}-frame chunks")
            cur_chunk = new_len
            continue
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9

        n_head = 0 if prev_tail is None else min(overlap, end - start)
        if prev_tail is None:
            mapping, n_matched = {}, 0
            for fr in per_frame:
                for o in fr:
                    key = (o["label"], o["local_id"])
                    if key not in mapping:
                        mapping[key] = next_id
                        next_id += 1
        else:
            mapping, next_id, n_matched = _link_ids(prev_tail, per_frame[:n_head], next_id, iou_thresh)

            # Grace window: SAM sometimes picks an object up several frames into the new chunk.
            # Compare such late objects with previous objects that are still unmatched. A fast
            # object has moved by then, so its last mask is first shifted by its velocity
            # (constant-velocity prediction from the last two overlap frames).
            last_prev = prev_tail[-1]
            used = set(mapping.values())
            velocity = {}
            if len(prev_tail) >= 2:
                for key_p, m_last in last_prev.items():
                    m_before = prev_tail[-2].get(key_p)
                    c1, c0 = _centroid(m_last), _centroid(m_before) if m_before is not None else None
                    if c1 is not None and c0 is not None:
                        velocity[key_p] = (c1[0] - c0[0], c1[1] - c0[1])
            for i in range(n_head, min(n_head + grace, len(per_frame))):
                dt = i - (n_head - 1)  # frames since the last overlap frame
                for o in per_frame[i]:
                    key = (o["label"], o["local_id"])
                    if key in mapping:
                        continue
                    best_gid, best_iou = None, iou_thresh
                    for (lab, gid), m_prev in last_prev.items():
                        if lab != o["label"] or gid in used:
                            continue
                        vx, vy = velocity.get((lab, gid), (0.0, 0.0))
                        v = mask_iou(_shift_mask(m_prev, vx * dt, vy * dt), o["mask"])
                        if v >= best_iou:
                            best_gid, best_iou = gid, v
                    if best_gid is not None:
                        mapping[key] = best_gid
                        used.add(best_gid)
                        n_matched += 1

            for fr in per_frame[n_head:]:  # objects that first appear after the overlap
                for o in fr:
                    key = (o["label"], o["local_id"])
                    if key not in mapping:
                        mapping[key] = next_id
                        next_id += 1

        for fr in per_frame:
            for o in fr:
                o["id"] = mapping[(o["label"], o.pop("local_id"))]

        if stats is not None:
            stats.append({"chunk": c, "start": start, "end": end, "frames": end - start,
                          "chunk_len": cur_chunk,
                          "seconds": round(elapsed, 2), "fps": round((end - start) / elapsed, 2),
                          "peak_vram_gb": round(peak, 2),
                          "objects": len(mapping), "linked_from_prev": n_matched})
        if verbose:
            print(f"[segment] chunk {c}: frames {start}-{end - 1}, {elapsed:.1f} s "
                  f"({(end - start) / elapsed:.2f} FPS incl. setup), peak {peak:.2f} GB, "
                  f"{len(mapping)} objects, {n_matched} linked to previous chunk")

        # the last `overlap` frames are kept to link the next chunk
        prev_tail = [{(o["label"], o["id"]): o["mask"] for o in fr} for fr in per_frame[-overlap:]]

        # frames already yielded by the previous chunk are skipped
        for i in range(n_head, end - start):
            yield {"frame": start + i, "path": paths[start + i], "objects": per_frame[i]}

        del per_frame
        gc.collect()
        torch.cuda.empty_cache()
        if end >= n_frames:
            break
        start, c = end - overlap, c + 1