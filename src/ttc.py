"""
Digbodha 1 - time to contact, collision course and risk, from a single camera (2D only).

Two classical results do all the work; neither needs depth or the object's real size.

1. Time to contact (Lee, 1976, "tau").  An object of real width W at distance Z appears
   with image size s = f*W/Z.  If it closes at speed V, then
           d(ln s)/dt = V/Z     so     TTC = Z/V = 1 / (d ln s / dt).
   W, Z, V and the focal length f all cancel.  We fit ln(s) against time over a short
   sliding window with least squares, which smooths mask noise and gives an error bar.

2. Collision course (predicted miss distance, in object widths).  Lateral offset X from
   the camera's line of travel appears as x = f*X/Z (measured from the image centre,
   which is the heading for a camera fixed to a body or car facing forward).  Hence
           x / s = X / W        (no f, no Z).
   Fitting q = x/s over the window and extrapolating to the moment of contact gives
           m = q_now + (dq/dt) * TTC = predicted offset at contact, in object widths.
   |m| small -> the object will pass through where the camera is going: on course.

The image size s is sqrt(mask area): steadier than box height or width, and scale-linear.
Uses RAW image measurements, not ego-compensated ones: "will it hit me" is about motion
relative to the camera (a cyclist pacing the car ahead is moving in the world but not
approaching). Ego-motion is used only for the "moving vs parked" label.

Usage:
    ttc = TTCEstimator(fps=10)
    res = ttc.update(frame_idx, fr["objects"], ego_res)   # ego_res from EgoMotion.step
    res[obj_id]["ttc_s"], res[obj_id]["risk"]
"""
from collections import deque

import cv2
import numpy as np

# "uncertain": the object seems close/approaching but the estimate is not trustworthy, so no
# caution/danger alert is raised from it (flag uncertainty rather than be confidently wrong).
RISK_ORDER = {"safe": 0, "uncertain": 1, "caution": 2, "danger": 3}


def clean_box(mask):
    """Largest connected piece of a mask -> (x_centre, width, height, (x0, y0, x1, y1)). Stray fragments far from
    the object would otherwise drag its position and size. Box width/height are used instead of
    area because thin objects (bicycles: spokes, frames) change area from frame to frame while
    their extent stays steady. Returns None for an empty mask."""
    m = mask.astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[k]
    return float(x + w / 2.0), float(w), float(h), (int(x), int(y), int(x + w), int(y + h))


def _linfit(t, y):
    """Least-squares slope, intercept and the slope's standard error."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    tm = t.mean()
    sxx = ((t - tm) ** 2).sum()
    if n < 3 or sxx == 0:
        return 0.0, float(y.mean()), float("inf")
    slope = ((t - tm) * (y - y.mean())).sum() / sxx
    intercept = y.mean() - slope * tm
    resid = y - (intercept + slope * t)
    se = np.sqrt((resid ** 2).sum() / (n - 2) / sxx)
    return float(slope), float(intercept), float(se)


class TTCEstimator:
    def __init__(self, fps, window_s=0.8, min_frames=5, ttc_max=10.0,
                 miss_thresh=1.0, danger_ttc=2.0, caution_ttc=4.0, near_pass_ttc=1.5,
                 still_px_per_s=15.0, min_size_px=30.0, edge_px=2, jump_ratio=1.25,
                 jump_max_frac=0.25):
        """
        fps           : frame rate of the video (KITTI: 10).
        window_s      : length of the fitting window in seconds.
        min_frames    : fewer points than this -> no estimate yet.
        ttc_max       : TTC above this (s) is treated as "not approaching".
        miss_thresh   : |predicted miss| below this many object widths -> on collision course.
        danger_ttc    : on course and TTC below this (s) -> danger.
        caution_ttc   : on course and TTC below this (s) -> caution.
        near_pass_ttc : NOT on course but TTC below this (s) -> caution (a close pass).
        still_px_per_s: own (ego-compensated) motion below this -> "still" (parked/standing).
        min_size_px   : box height (px) below this -> estimate marked unreliable.
        edge_px       : box within this many px of the top/bottom edge -> cut (unreliable).
        jump_ratio    : box height changing by more than this factor between frames ...
        jump_max_frac : ... while shorter than this fraction of the image -> "size_jump".
        """
        self.min_size_px = min_size_px
        self.edge_px = edge_px
        self.jump_ratio = jump_ratio
        self.jump_max_frac = jump_max_frac
        self.img_h = None
        self.fps = float(fps)
        self.window = max(min_frames, int(round(window_s * fps)))
        self.min_frames = min_frames
        self.ttc_max = ttc_max
        self.miss_thresh = miss_thresh
        self.danger_ttc = danger_ttc
        self.caution_ttc = caution_ttc
        self.near_pass_ttc = near_pass_ttc
        self.still = still_px_per_s
        self.hist = {}  # id -> deque of (t, x_from_centre, s, border)

    def update(self, frame_idx, objects, ego_res=None):
        """objects: [{"id", "label", "mask"}, ...] for this frame.
        ego_res : output of EgoMotion.step for this frame (optional; for border and motion labels).
        Returns {id: estimate dict}."""
        t = frame_idx / self.fps
        out = {}
        seen = set()
        for o in objects:
            oid, mask = o["id"], o["mask"]
            seen.add(oid)
            h, w = mask.shape
            self.img_h = h
            box = clean_box(mask)
            if box is None:
                continue
            xc, bw, bh, bbox = box
            x = xc - w / 2.0                       # offset from the heading (image centre)
            ego_o = (ego_res or {}).get("objects", {}).get(oid, {})
            # Which image edge cuts the object matters: a TOP/BOTTOM cut corrupts its height (and
            # so the TTC); a LEFT/RIGHT cut leaves the height intact and only shifts the visible
            # centre toward the image middle, which biases the miss distance toward "on course"
            # (the safe direction for a warning system), so it does not make the estimate unreliable.
            x0, y0, x1, y1 = bbox
            m = self.edge_px
            cut_tb = y0 <= m or y1 >= h - m
            cut_side = x0 <= m or x1 >= w - m
            dq = self.hist.setdefault(oid, deque(maxlen=self.window))
            dq.append((t, x, bw, bh, cut_tb, cut_side))
            out[oid] = self._estimate(oid, o.get("label", ""), dq, w, ego_o, ego_res)
            out[oid]["box"] = bbox
        for oid in list(self.hist):  # forget objects not seen for a while
            if oid not in seen and self.hist[oid] and t - self.hist[oid][-1][0] > 1.0:
                del self.hist[oid]
        return out

    # ------------------------------------------------------------------------------
    def _estimate(self, oid, label, dq, img_w, ego_o, ego_res):
        ts = [p[0] for p in dq]
        xs = np.array([p[1] for p in dq])
        ws = np.array([p[2] for p in dq])
        hs = np.array([p[3] for p in dq])
        # Image size = box HEIGHT. Height depends only on distance; width also changes with viewing
        # angle (swinging around a parked car shows more of its side) and grows when an object
        # emerges from behind an occluder. With height, an emerging object simply isn't approaching.
        ss = hs
        x_now, s_now = xs[-1], ss[-1]
        zone = "left" if x_now < -img_w / 6 else ("right" if x_now > img_w / 6 else "centre")

        res = {"label": label, "zone": zone, "size_px": round(float(s_now), 1),
               "n_frames": len(dq), "expansion_per_s": None, "ttc_s": None,
               "ttc_lo": None, "ttc_hi": None, "approaching": None, "miss_widths": None,
               "on_course": None, "motion": self._motion_label(ego_o, ego_res),
               "risk": "safe", "reliable": False, "reasons": [], "cut_side": False}

        if len(dq) < self.min_frames:
            res["reasons"].append("warming_up")
            return res

        # 1/s is proportional to distance (1/s = Z / (f W)), so under constant closing speed it
        # falls in a STRAIGHT line and reaches 0 at contact. Fitting that line and reading where it
        # hits zero gives TTC *now*. (Fitting d ln s/dt instead lags: it gives TTC at the window's
        # middle, and an approaching object's expansion rate keeps rising.)
        t_now = ts[-1]
        u = 1.0 / ss
        b, a, b_se = _linfit(ts, u)                        # u(t) = a + b t ;  b < 0 when approaching
        u_now = a + b * t_now
        q = xs / ss                                         # = X / W, linear in t if X is
        q_slope, q_icpt, _ = _linfit(ts, q)
        q_now = q_icpt + q_slope * t_now
        res["expansion_per_s"] = round(-b / u_now, 4) if u_now > 0 else None

        ttc = u_now / -b if (b < 0 and u_now > 0) else float("inf")
        approaching = ttc < self.ttc_max
        res["approaching"] = bool(approaching)
        significant = -b > 2 * b_se                         # shrinking distance clearly above noise

        if approaching:
            res["ttc_s"] = round(ttc, 2)
            b_fast, b_slow = b - b_se, b + b_se             # steeper / shallower closing
            res["ttc_lo"] = round(u_now / -b_fast, 2)
            res["ttc_hi"] = round(u_now / -b_slow, 2) if b_slow < 0 else None  # None = could be infinite
            m = float(q_now + q_slope * ttc)
            res["miss_widths"] = round(m, 2)
            res["on_course"] = bool(abs(m) < self.miss_thresh)

            if res["on_course"] and ttc < self.danger_ttc:
                res["risk"] = "danger"
            elif (res["on_course"] and ttc < self.caution_ttc) or ttc < self.near_pass_ttc:
                res["risk"] = "caution"

        # reliability: enough history, real expansion (when approaching), no edge cut, steady camera
        if any(p[4] for p in dq):
            res["reasons"].append("cut_top_bottom")
        res["cut_side"] = bool(dq[-1][5])
        # A sudden height jump between consecutive frames on an object that is still small in the
        # image is a mask glitch or occlusion, not motion (a real 25% jump in 0.1 s would mean
        # contact within ~0.4 s, which only happens to objects that already fill much of the view).
        jumps = np.maximum(hs[1:] / hs[:-1], hs[:-1] / hs[1:]) if len(hs) > 1 else np.array([1.0])
        if jumps.max() > self.jump_ratio and hs[-1] < self.jump_max_frac * self.img_h:
            res["reasons"].append("size_jump")
        if approaching and ws[-1] < 0.85 * ws[0]:
            # height says "closer" but the width shrank by >15%: physically odd for an approaching
            # object (usually a mask that lost part of the object) -> don't trust it
            res["reasons"].append("inconsistent_growth")
        if s_now < self.min_size_px:          # far/small: a few px of mask noise swamps the growth
            res["reasons"].append("too_small")
        if approaching and not significant:
            res["reasons"].append("noisy_expansion")
        if ego_res is not None and not ego_res.get("camera", {}).get("reliable", True):
            res["reasons"].append("camera_unreliable")
        res["reliable"] = len(res["reasons"]) == 0
        if not res["reliable"] and res["risk"] in ("caution", "danger"):
            res["risk"] = "uncertain"    # would alert, but the estimate is not trustworthy
        return res

    @staticmethod
    def _ttc_of(ts, sizes):
        """TTC (s) from one size series via the 1/size line; inf if not approaching."""
        b, a, _ = _linfit(ts, 1.0 / np.asarray(sizes, float))
        u_now = a + b * ts[-1]
        return u_now / -b if (b < 0 and u_now > 0) else float("inf")

    def _motion_label(self, ego_o, ego_res):
        """Moving or still in the world, from ego-compensated flow (px/frame -> px/s)."""
        own = ego_o.get("own_flow") if ego_o else None
        if own is None:
            return "unknown"
        vx, vy = own[0] * self.fps, own[1] * self.fps
        if np.hypot(vx, vy) < self.still:
            return "still"
        return "moving_left" if vx < 0 and abs(vx) >= abs(vy) else \
               "moving_right" if vx > 0 and abs(vx) >= abs(vy) else "moving"