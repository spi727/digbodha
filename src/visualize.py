"""
Digbodha 1 - draw results from frames.csv onto the frames and write a browser-playable video.

The video is H.264 (via the ffmpeg binary bundled with imageio-ffmpeg), so it plays in Jupyter,
Colab and web browsers, and needs no system ffmpeg.
"""
import csv
import glob
import os

import cv2
import numpy as np

from src.labels import num

COLORS_RGB = {"safe": (0, 180, 0), "uncertain": (180, 0, 180), "caution": (255, 170, 0), "danger": (255, 0, 0)}


def load_rows(frames_csv):
    by_frame = {}
    with open(frames_csv) as f:
        for r in csv.DictReader(f):
            by_frame.setdefault(int(r["frame"]), []).append(r)
    return by_frame


def list_frames(frames_dir):
    return sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))


def draw(img_path, rows):
    """RGB image with each object's box in its risk colour and 'id label TTC risk'."""
    img = np.ascontiguousarray(cv2.imread(img_path)[:, :, ::-1])
    for r in rows:
        c = COLORS_RGB.get(r["risk"], (200, 200, 200))
        x0, y0, x1, y1 = (int(float(r[k])) for k in ("bx0", "by0", "bx1", "by1"))
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
        ttc = num(r["ttc_s"])
        txt = f"#{r['id']} {r['label']} " + (f"{ttc:.1f}s " if ttc is not None and ttc < 10 else "") + r["risk"]
        cv2.putText(img, txt, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return img


def write_video(frames_dir, frames_csv, out_path, fps, speed=0.25, width=960, quality=5):
    """Annotated H.264 video of the whole run at `speed` x real time. Returns out_path."""
    try:
        import imageio_ffmpeg  # noqa: F401  (ffmpeg binary; without it imageio silently picks another writer)
    except ImportError as e:
        raise ImportError("pip install imageio imageio-ffmpeg") from e
    import imageio.v2 as imageio

    by_frame = load_rows(frames_csv)
    paths = list_frames(frames_dir)
    h0, w0 = cv2.imread(paths[0]).shape[:2]
    W = width // 8 * 8
    H = int(round(h0 * W / w0)) // 8 * 8
    writer = imageio.get_writer(out_path, format="FFMPEG", fps=fps * speed, codec="libx264",
                                quality=quality, macro_block_size=8, ffmpeg_log_level="error")
    for i, p in enumerate(paths):
        img = cv2.resize(draw(p, by_frame.get(i, [])), (W, H), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"frame {i}  t={i / fps:.1f}s  ({speed:g}x)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.append_data(img)
    writer.close()
    return out_path
