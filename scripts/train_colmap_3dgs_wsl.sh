#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_run="$(realpath "${1:?COLMAP run directory required}")"
task_output="$task_run/3dgs-7000"
export PYTHONPATH="$task_root/third_party/3dgs-packages"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
task_python="$task_root/third_party/2dgs-env/bin/python"
"$task_python" - "$task_run" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
report = json.loads((run/'report.json').read_text())
if report['status'] != 'ready' or report['registered_fraction'] < .8:
    raise SystemExit('COLMAP quality check failed; refusing GS training')
if (run/'3dgs-7000').exists():
    raise SystemExit('Existing output will not be overwritten')
PY
cd "$task_root/third_party/3dgs"
"$task_python" train.py -s "$task_run/colmap" -m "$task_output" \
    --iterations 7000 --position_lr_max_steps 7000 --densify_until_iter 5000 \
    --sh_degree 2 --eval --data_device cpu -r 2 --disable_viewer \
    --test_iterations 1000 7000 --save_iterations 7000 --checkpoint_iterations 7000 \
    > "$task_run/3dgs-7000.log" 2>&1
