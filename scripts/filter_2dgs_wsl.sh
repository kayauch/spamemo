#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
third_party/2dgs-env/bin/python scripts/filter_2dgs.py \
  "${1:-data/2dgs-room/refine-depth-3000}" "${2:-data/2dgs-room/refine-filtered-3000}" \
  data/scenes-v2/d93ad097-c9d1-466c-b6fa-a8bad5fff538/reconstruction/reconstruction.npz \
  --iteration 3000 --tolerance "${3:-0.08}"
