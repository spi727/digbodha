#!/usr/bin/env bash
# Digbodha 1 - re-score saved KITTI runs with the current src/ttc.py (no SAM, takes seconds),
# then evaluate, pool and analyse errors.
#
# Usage (repo root):  bash scripts/rescore_all.sh [TAG] [extra rescore options]
#   bash scripts/rescore_all.sh v5
#   bash scripts/rescore_all.sh nodebounce --debounce 1
set -euo pipefail
TAG=${1:-rescored}; shift || true
KITTI=${KITTI:-$HOME/datasets/kitti_tracking}
SEQS=(${SEQS:-0013:340 0001:426 0020:300})

DIRS=()
for item in "${SEQS[@]}"; do
  seq=${item%%:*}; n=${item##*:}
  python eval/rescore.py --pred "outputs/kitti_${seq}/frames.csv" --frames "data/kitti_${seq}_${n}" \
    --fps 10 --out "outputs/kitti_${seq}_${TAG}/frames.csv" "$@"
  python eval/evaluate_kitti.py --pred "outputs/kitti_${seq}_${TAG}/frames.csv" \
    --labels "$KITTI/training/label_02/${seq}.txt" --fps 10 --out "outputs/eval_${seq}_${TAG}" > /dev/null
  DIRS+=("outputs/eval_${seq}_${TAG}")
done
python eval/aggregate.py "${DIRS[@]}" --out "outputs/eval_pooled_${TAG}"
python eval/analyze_errors.py "${DIRS[@]}"