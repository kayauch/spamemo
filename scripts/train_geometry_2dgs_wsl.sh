#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root"
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
task_output="${1:-data/2dgs-room/train-geometry-7000}"
if [ -e "$task_output" ]; then echo "Output already exists: $task_output" >&2; exit 1; fi
third_party/2dgs-env/bin/python scripts/refine_2dgs.py \
  -s data/2dgs-room/colmap -m "$task_output" --eval -r 1 --data_device cpu --sh_degree 2 \
  --iterations 7000 --depth-weight 0.3 --normal-prior-weight 0.02 --confidence-photo-weighting \
  --geometry-npz data/scenes-v2/d93ad097-c9d1-466c-b6fa-a8bad5fff538/reconstruction/reconstruction.npz \
  > "$task_output.log" 2>&1
