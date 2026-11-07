"""
Digbodha 1 - SAM 3.1 smoke test.

Runs text-prompted video tracking with SAM 3.1, prints the structure of the
model's outputs, measures FPS and peak GPU memory, and saves a few overlay frames.

Usage (from the repo root, with the cvproj env active):
  python scripts/test_sam3.py
      -> Meta's sample video, prompt "person"
  python scripts/test_sam3.py --video data/test_frames --prompt car
      -> your own clip (JPEG folder or .mp4)
"""
import argparse
import glob
import os
import time

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed in WSL
import matplotlib.pyplot as plt
import torch

import sam3
from sam3.model_builder import build_sam3_multiplex_video_predictor
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)


def describe(name, value, indent=2):
    """Recursively print the type/shape of whatever SAM 3.1 returns."""
    pad = " " * indent
    if torch.is_tensor(value):
        print(f"{pad}{name}: Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
        print(f"{pad}{name}: ndarray shape={value.shape} dtype={value.dtype}")
    elif isinstance(value, dict):
        print(f"{pad}{name}: dict, keys={list(value.keys())}")
        for k, v in value.items():
            describe(str(k), v, indent + 2)
    elif isinstance(value, (list, tuple)):
        print(f"{pad}{name}: {type(value).__name__}, len={len(value)}")
        if len(value) > 0:
            describe(f"{name}[0]", value[0], indent + 2)
    else:
        text = f"{pad}{name}: {type(value).__name__} = {value!r}"
        print(text[:200])


def load_frames_for_vis(video_path):
    """Frames used only for drawing overlays (the model reads the video itself)."""
    if video_path.endswith(".mp4"):
        import cv2

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    frames = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        frames.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        frames.sort()
    return frames


def main():
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=os.path.join(sam3_root, "assets", "videos", "0001"),
                        help="JPEG folder or .mp4 file")
    parser.add_argument("--prompt", default="person", help="text prompt, e.g. car, bike, person")
    parser.add_argument("--out", default="outputs/sam3_test", help="folder for overlay images")
    parser.add_argument("--save_every", type=int, default=30, help="save an overlay every N frames")
    # Speed / memory knobs for experiments
    parser.add_argument("--max_objects", type=int, default=8, help="max objects tracked at once")
    parser.add_argument("--bf16", action="store_true", help="run inference in bfloat16 (less memory, faster)")
    parser.add_argument("--grounding_batch", type=int, default=0,
                        help="detector frames per batch; 0 = off (one frame at a time)")
    parser.add_argument("--compile", action="store_true", help="use torch.compile (slow first run)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Build SAM 3.1 (first run downloads the checkpoint from Hugging Face)
    print("Building SAM 3.1 predictor...")
    t0 = time.perf_counter()
    # use_fa3=False: FlashAttention 3 targets H100 GPUs and isn't installed;
    # fall back to PyTorch's built-in attention on the RTX 5060.
    predictor = build_sam3_multiplex_video_predictor(
        use_fa3=False, max_num_objects=args.max_objects, compile=args.compile
    )
    print(f"  built in {time.perf_counter() - t0:.1f} s")

    # Compatibility patch: the session code passes keyword arguments (e.g.
    # offload_state_to_cpu) that the SAM 3.1 multiplex model's init_state()
    # does not accept. Drop any argument init_state doesn't know, and report it.
    import functools
    import inspect

    original_init_state = predictor.model.init_state
    accepted = inspect.signature(original_init_state).parameters

    @functools.wraps(original_init_state)
    def init_state_compat(*a, **kw):
        dropped = [k for k in kw if k not in accepted]
        if dropped:
            print(f"  [compat] ignoring unsupported init_state args: {dropped}")
        return original_init_state(*a, **{k: v for k, v in kw.items() if k in accepted})

    predictor.model.init_state = init_state_compat

    # Memory fix for 8 GB GPUs: during propagation SAM 3.1 runs its detector on
    # a batch of frames at once (16 by default), which overflows 8 GB of VRAM.
    # Run detection one frame at a time instead.
    m = predictor.model
    print(f"  batched grounding was: use={getattr(m, 'use_batched_grounding', 'n/a')}, "
          f"batch_size={getattr(m, 'batched_grounding_batch_size', 'n/a')}")
    if args.grounding_batch > 0:
        m.use_batched_grounding = True
        m.batched_grounding_batch_size = args.grounding_batch
    else:
        m.use_batched_grounding = False
        m.batched_grounding_batch_size = 1
    print(f"  batched grounding now: use={m.use_batched_grounding}, "
          f"batch_size={m.batched_grounding_batch_size}")

    # Optional bfloat16: tensor cores run bf16 much faster and it halves activation memory
    import contextlib

    def amp():
        if args.bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    print(f"  settings: max_objects={args.max_objects}, bf16={args.bf16}, compile={args.compile}")

    frames_vis = load_frames_for_vis(args.video)
    print(f"Video: {args.video}  ({len(frames_vis)} frames)")

    # 2. Open a session on the video
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=args.video)
    )
    session_id = response["session_id"]

    # 3. Text prompt on frame 0, and inspect what comes back
    with amp():
        response = predictor.handle_request(
            request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=args.prompt)
        )
    print("\n=== Output format on frame 0 ===")
    describe("outputs", response["outputs"])

    # 4. Propagate through the whole video, timing it
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    outputs_per_frame = {}
    with amp():
        for r in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs_per_frame[r["frame_index"]] = r["outputs"]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_frames = len(outputs_per_frame)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print("\n=== Performance ===")
    print(f"  prompt         : {args.prompt!r}")
    print(f"  frames tracked : {n_frames}")
    print(f"  time           : {elapsed:.2f} s")
    print(f"  FPS            : {n_frames / elapsed:.2f}")
    print(f"  peak VRAM      : {peak_gb:.2f} GB")

    # 5. Save a few overlay frames to check the masks by eye
    vis = prepare_masks_for_visualization(outputs_per_frame)
    for idx in range(0, n_frames, args.save_every):
        plt.close("all")
        visualize_formatted_frame_output(
            idx,
            frames_vis,
            outputs_list=[vis],
            titles=[f"SAM 3.1 '{args.prompt}' - frame {idx}"],
            figsize=(8, 5),
        )
        path = os.path.join(args.out, f"frame_{idx:05d}.png")
        plt.savefig(path, bbox_inches="tight")
    print(f"\nOverlays saved to {args.out}/")

    # 6. Close the session
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))


if __name__ == "__main__":
    main()