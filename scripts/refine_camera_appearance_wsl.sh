#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root"
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
task_output="${1:-data/2dgs-room/refine-cameras-3000}"
if [ -e "$task_output" ]; then echo "Output already exists: $task_output" >&2; exit 1; fi
third_party/2dgs-env/bin/python scripts/refine_2dgs.py \
  -s "${2:-data/2dgs-room/colmap-cameras}" -m "$task_output" --eval -r 1 --data_device cpu --sh_degree 2 \
  --iterations 3000 --depth-weight 0 --normal-prior-weight 0 --freeze-positions \
  --structure-sh-penalty "${4:-0}" \
  --initial-ply "${3:-data/2dgs-room/refine-hires-3000/point_cloud/iteration_3000/point_cloud.ply}" \
  --geometry-npz data/scenes-v2/d93ad097-c9d1-466c-b6fa-a8bad5fff538/reconstruction/reconstruction.npz \
  > "$task_output.log" 2>&1
