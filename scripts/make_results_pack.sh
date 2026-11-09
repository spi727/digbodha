#!/usr/bin/env bash
# Digbodha 1 - build the results pack that the Colab notebook downloads in "results" mode.
#
# Contents (KITTI Tracking, CC BY-NC-SA 3.0, Geiger et al. 2012):
#   data/kitti_0007_300/*.jpg                 the held-out sequence's frames (JPEG quality 85)
#   kitti_labels/0007.txt                     its KITTI labels
#   outputs/kitti_0007/frames.csv             saved SAM 3.1 + ego-motion results for it
#   outputs/eval_<seq>_<tag>/{matched.csv,metrics.json}   saved evaluations of all 5 sequences
#
# Usage (repo root):  bash scripts/make_results_pack.sh
# Then upload digbodha_results_pack.zip to a GitHub release tagged v1.0 (see the README).
set -euo pipefail
KITTI=${KITTI:-$HOME/datasets/kitti_tracking}
PACK=$(mktemp -d)
mkdir -p "$PACK/data/kitti_0007_300" "$PACK/kitti_labels" "$PACK/outputs/kitti_0007"

python - "$PACK/data/kitti_0007_300" <<'EOF'
import cv2, glob, os, sys
dst = sys.argv[1]
for f in sorted(glob.glob("data/kitti_0007_300/*.jpg")):
    cv2.imwrite(os.path.join(dst, os.path.basename(f)), cv2.imread(f), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("frames:", len(os.listdir(dst)))
EOF
cp "$KITTI/training/label_02/0007.txt" "$PACK/kitti_labels/"
cp outputs/kitti_0007/frames.csv "$PACK/outputs/kitti_0007/"
for d in eval_0013_v5_nodebounce eval_0001_v5_nodebounce eval_0020_v5_nodebounce \
         eval_0019_heldout_nodebounce eval_0007_heldout_nodebounce; do
  mkdir -p "$PACK/outputs/$d"
  cp "outputs/$d/matched.csv" "$PACK/outputs/$d/"
  cp "outputs/$d/metrics.json" "$PACK/outputs/$d/" 2>/dev/null || true
done
echo "Digbodha 1 results pack. KITTI data: CC BY-NC-SA 3.0, Geiger, Lenz, Urtasun, CVPR 2012." > "$PACK/PACK_README.txt"
rm -f digbodha_results_pack.zip
(cd "$PACK" && python -m zipfile -c "$OLDPWD/digbodha_results_pack.zip" data kitti_labels outputs PACK_README.txt)
rm -rf "$PACK"
ls -lh digbodha_results_pack.zip
