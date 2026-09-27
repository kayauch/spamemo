#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_data="${1:-$task_root/data/2dgs-room/colmap}"
task_output="${2:-$task_root/data/2dgs-room/train-7000}"
task_iterations="${3:-7000}"
if [ -e "$task_output" ]; then
  echo "Choose a new output directory; refusing to overwrite: $task_output" >&2
  exit 1
fi
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$task_output")"
echo "Training log: $task_output.log"
cd "$task_root/third_party/2dgs"
"$task_root/third_party/2dgs-env/bin/python" train.py -s "$task_data" -m "$task_output" \
  --iterations "$task_iterations" --position_lr_max_steps "$task_iterations" \
  --densify_until_iter 5000 --sh_degree 2 --eval --data_device cpu -r 1 \
  --test_iterations 1 1000 "$task_iterations" --save_iterations "$task_iterations" \
  --checkpoint_iterations "$task_iterations" --port 6019 > "$task_output.log" 2>&1
echo "Training complete: $task_output"
