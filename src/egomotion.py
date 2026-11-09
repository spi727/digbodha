"""
Digbodha 1 - camera (ego) motion estimation and per-object motion with the camera removed.

Idea: when the camera moves, everything in the image moves. The background (everything
outside the object masks) moves only because of the camera, so we:

  1. track corner points on the background from the previous frame to this one
     (Shi-Tomasi corners + pyramidal Lucas-Kanade optical flow, forward-backward checked),
  2. fit a homography H (prev -> curr) to those tracks with RANSAC  = the camera's motion,
  3. track points inside each object's mask the same way; the object's OWN motion is what
     is left after removing what H predicts:   own = p_curr - H(p_prev).

A homography is exact for camera rotation and for a flat background; for a camera that
translates through a 3D scene it is an approximation (parallax), which the report notes.

Usage:
    ego = EgoMotion()
    for fr in segment_video(...):
        img = cv2.imread(fr["path"])
        res = ego.step(img, fr["objects"])
        res["camera"], res["objects"][obj_id]["own_flow"]
"""
import cv2
import numpy as np

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def mask_centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def touches_border(mask, margin=2):
    """True if the mask reaches within `margin` pixels of any image edge."""
    return bool(mask[:margin].any() or mask[-margin:].any()
                or mask[:, :margin].any() or mask[:, -margin:].any())


def apply_h(H, pts):
    """Apply homography to Nx2 points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


class EgoMotion:
    def __init__(self, work_width=640, bg_corners=800, obj_corners=60, mask_dilate=15,
                 ransac_thresh=1.5, fb_thresh=1.0, min_bg_inliers=30, min_obj_points=5):
        """
        work_width     : frames are downscaled to this width for flow (speed); results are
                         returned in full-resolution pixels.
        bg_corners     : max background corners tracked per frame.
        obj_corners    : max corners tracked inside each object.
        mask_dilate    : object masks are grown by this many work-pixels before being removed
                         from the background, so object edges don't pollute the camera fit.
        ransac_thresh  : RANSAC reprojection threshold (work-pixels).
        fb_thresh      : max forward-backward error (work-pixels) for a track to be kept.
        min_bg_inliers : fewer RANSAC inliers than this -> camera estimate marked unreliable.
        min_obj_points : fewer tracked points in an object -> fall back to its mask centroid.
        """
        self.work_width = work_width
        self.bg_corners = bg_corners
        self.obj_corners = obj_corners
        self.kernel = np.ones((mask_dilate, mask_dilate), np.uint8)
        self.ransac_thresh = ransac_thresh
        self.fb_thresh = fb_thresh
        self.min_bg_inliers = min_bg_inliers
        self.min_obj_points = min_obj_points
        self.reset()

    def reset(self):
        self.prev_gray = None      # downscaled grey frame
        self.prev_masks = {}       # id -> downscaled uint8 mask
        self.prev_centroids = {}   # id -> full-res centroid
        self.prev_full_masks = {}  # id -> full-res bool mask (for the border check)
        self.scale = None          # work / full

    # ------------------------------------------------------------------ helpers
    def _track(self, p0, gray0, gray1):
        """LK tracking with a forward-backward check. Returns (p0_kept, p1_kept)."""
        if p0 is None or len(p0) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        p0 = p0.reshape(-1, 1, 2).astype(np.float32)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **LK_PARAMS)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **LK_PARAMS)
        fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_thresh)
        return p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]

    def _small_mask(self, mask, shape):
        return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------------ main step
    def step(self, frame_bgr, objects):
        """
        frame_bgr : full-resolution BGR image of the current frame.
        objects   : list of {"id", "mask", ...} for the current frame (from segment_video).

        Returns {
          "camera": {"H": 3x3 list (full-res, prev->curr) or None, "inliers": int,
                     "inlier_ratio": float, "shift_center": (dx, dy) px, "reliable": bool},
          "objects": {id: {"centroid": (x, y), "raw_flow": (dx, dy), "own_flow": (dx, dy),
                           "n_points": int, "method": "points"|"centroid"|"new",
                           "border": bool, "reliable": bool}}
        }   (flows in full-resolution pixels per frame)
        """
        h_full, w_full = frame_bgr.shape[:2]
        if self.scale is None:
            self.scale = self.work_width / w_full
        s = self.scale
        small = cv2.resize(frame_bgr, (int(round(w_full * s)), int(round(h_full * s))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        cur_full_masks = {o["id"]: o["mask"] for o in objects}
        cur_masks = {o["id"]: self._small_mask(o["mask"], gray.shape) for o in objects}
        cur_centroids = {o["id"]: mask_centroid(o["mask"]) for o in objects}

        result = {"camera": {"H": None, "inliers": 0, "inlier_ratio": 0.0,
                             "shift_center": (0.0, 0.0), "reliable": False},
                  "objects": {}}

        if self.prev_gray is None:  # first frame: nothing to compare with
            for oid, c in cur_centroids.items():
                result["objects"][oid] = {"centroid": c, "raw_flow": None, "own_flow": None,
                                          "n_points": 0, "method": "new", "border": False,
                                          "reliable": False}
            self._remember(gray, cur_masks, cur_centroids, cur_full_masks)
            return result

        # ---- 1. camera motion from the background -------------------------------
        fg = np.zeros_like(self.prev_gray)
        for m in self.prev_masks.values():
            fg |= m
        bg_mask = (cv2.dilate(fg, self.kernel) == 0).astype(np.uint8) * 255
        corners = cv2.goodFeaturesToTrack(self.prev_gray, self.bg_corners, 0.01, 7, mask=bg_mask)
        b0, b1 = self._track(corners, self.prev_gray, gray)

        H_small = None
        if len(b0) >= 8:
            H_small, inl = cv2.findHomography(b0, b1, cv2.RANSAC, self.ransac_thresh)
            n_inl = int(inl.sum()) if inl is not None else 0
            result["camera"]["inliers"] = n_inl
            result["camera"]["inlier_ratio"] = round(n_inl / len(b0), 3)
            result["camera"]["reliable"] = H_small is not None and n_inl >= self.min_bg_inliers
        if H_small is None:
            H_small = np.eye(3)  # no estimate: assume a still camera, flagged unreliable

        # convert H to full-resolution pixel coordinates: H_full = S^-1 H_small S
        S = np.diag([s, s, 1.0])
        H_full = np.linalg.inv(S) @ H_small @ S
        result["camera"]["H"] = H_full.tolist()
        c = np.array([[w_full / 2, h_full / 2]])
        shift = apply_h(H_full, c)[0] - c[0]
        result["camera"]["shift_center"] = (round(float(shift[0]), 2), round(float(shift[1]), 2))

        # ---- 2. each object's own motion ------------------------------------------
        for oid, c_cur in cur_centroids.items():
            if oid not in self.prev_masks or c_cur is None or self.prev_centroids.get(oid) is None:
                result["objects"][oid] = {"centroid": c_cur, "raw_flow": None, "own_flow": None,
                                          "n_points": 0, "method": "new", "border": False,
                                          "reliable": False}
                continue
            pm = self.prev_masks[oid]
            pts = cv2.goodFeaturesToTrack(self.prev_gray, self.obj_corners, 0.01, 3,
                                          mask=(pm * 255).astype(np.uint8))
            p0, p1 = self._track(pts, self.prev_gray, gray)
            if len(p0) >= self.min_obj_points:
                raw = np.median(p1 - p0, axis=0) / s
                own = np.median(p1 - apply_h(H_small, p0), axis=0) / s
                method, n = "points", len(p0)
            else:  # too little texture: use the mask centroid instead
                c_prev = np.array([self.prev_centroids[oid]])
                raw = np.array(c_cur) - c_prev[0]
                own = np.array(c_cur) - apply_h(H_full, c_prev)[0]
                method, n = "centroid", len(p0)
            # an object cut by the image edge has a sliding mask centre: not trustworthy
            border = bool(touches_border(cur_full_masks[oid]) or touches_border(self.prev_full_masks[oid]))
            result["objects"][oid] = {
                "centroid": (round(c_cur[0], 2), round(c_cur[1], 2)),
                "raw_flow": (round(float(raw[0]), 3), round(float(raw[1]), 3)),
                "own_flow": (round(float(own[0]), 3), round(float(own[1]), 3)),
                "n_points": int(n), "method": method, "border": border,
                "reliable": bool(result["camera"]["reliable"] and method == "points" and not border),
            }

        self._remember(gray, cur_masks, cur_centroids, cur_full_masks)
        return result

    def _remember(self, gray, masks, centroids, full_masks):
        self.prev_gray, self.prev_masks, self.prev_centroids = gray, masks, centroids
        self.prev_full_masks = full_masks