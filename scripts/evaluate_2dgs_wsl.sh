#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
cd "$task_root"
third_party/2dgs-env/bin/python scripts/evaluate_2dgs.py \
  -s "${1:-data/2dgs-room/colmap}" -m "${2:-data/2dgs-room/train-7000}" \
  --iteration "${3:-7000}" --sh_degree 2 --eval --data_device cpu -r 1
