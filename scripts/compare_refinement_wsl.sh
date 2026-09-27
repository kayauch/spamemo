#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root"
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
task_model="${1:-data/2dgs-room/refine-depth-3000}"
task_steps="${2:-3000}"
third_party/2dgs-env/bin/python scripts/evaluate_2dgs.py \
  -s "${3:-data/2dgs-room/colmap}" -m "$task_model" --iteration "$task_steps" \
  --output-name "${4:-evaluation}" \
  --sh_degree 2 --eval --data_device cpu -r 1 \
  --geometry-npz data/scenes-v2/d93ad097-c9d1-466c-b6fa-a8bad5fff538/reconstruction/reconstruction.npz \
  > "$task_model-${4:-evaluation}.log" 2>&1
