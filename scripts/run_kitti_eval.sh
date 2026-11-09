#!/usr/bin/env bash
# Digbodha 1 - run the full pipeline + ground-truth evaluation on several KITTI tracking sequences.
#
# For each sequence: extract its images from the big zip (only that sequence), convert the first
# N frames to JPG, run analyze.py, run eval/evaluate_kitti.py. Finally pool all sequences.
#
# Usage (repo root, cvproj env active):
#   bash scripts/run_kitti_eval.sh                 # default sequences below
#   bash scripts/run_kitti_eval.sh 0013:340 0001:426
# Each argument is SEQ:MAX_FRAMES.

set -euo pipefail

KITTI=${KITTI:-$HOME/datasets/kitti_tracking}
ZIP=${ZIP:-/mnt/c/Users/prave/Downloads/data_tracking_image_2.zip}
PROMPTS=${PROMPTS:-vehicle,person}
SEQS=("$@")
if [ ${#SEQS[@]} -eq 0 ]; then
  SEQS=(0013:340 0001:426 0020:300)
fi

mkdir -p data outputs
EVAL_DIRS=()
for item in "${SEQS[@]}"; do
  seq=${item%%:*}
  n=${item##*:}
  img_dir="$KITTI/training/image_02/$seq"
  frames_dir="data/kitti_${seq}_${n}"
  out_dir="outputs/kitti_${seq}"
  eval_dir="outputs/eval_${seq}"
  echo "=================== sequence $seq (first $n frames) ==================="

  if [ ! -d "$img_dir" ] || [ -z "$(ls -A "$img_dir" 2>/dev/null)" ]; then
    echo "[1/4] extracting $seq from $ZIP"
    (cd "$KITTI" && unzip -q -o "$ZIP" "training/image_02/$seq/*")
  else
    echo "[1/4] images already extracted"
  fi

  if [ ! -d "$frames_dir" ] || [ -z "$(ls -A "$frames_dir" 2>/dev/null)" ]; then
    echo "[2/4] converting first $n frames to JPG"
    python - "$img_dir" "$frames_dir" "$n" <<'EOF'
import cv2, glob, os, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
files = sorted(glob.glob(os.path.join(src, "*.png")))[:n]
for f in files:
    cv2.imwrite(os.path.join(dst, os.path.basename(f).replace(".png", ".jpg")), cv2.imread(f),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"  {len(files)} frames")
EOF
  else
    echo "[2/4] JPG frames already there"
  fi

  echo "[3/4] analyze.py"
  python analyze.py --video "$frames_dir" --fps 10 --prompts "$PROMPTS" --out "$out_dir" --no_video

  echo "[4/4] evaluate_kitti.py"
  python eval/evaluate_kitti.py --pred "$out_dir/frames.csv" \
    --labels "$KITTI/training/label_02/$seq.txt" --fps 10 --out "$eval_dir"
  EVAL_DIRS+=("$eval_dir")
done

echo "=================== pooled over: ${EVAL_DIRS[*]} ==================="
python eval/aggregate.py "${EVAL_DIRS[@]}" --out outputs/eval_pooled