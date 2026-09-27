#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_HOME="$task_root/third_party/cuda-12.8"
export PATH="$CUDA_HOME/bin:$task_root/third_party/2dgs-env/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
python "$task_root/scripts/patch_3dgs_cuda.py"
# Reuse tested Torch, but keep the 3D rasterizer outside the existing environment.
python -m pip install --no-deps --no-build-isolation --target "$task_root/third_party/3dgs-packages" "$task_root/third_party/3dgs/submodules/diff-gaussian-rasterization"
PYTHONPATH="$task_root/third_party/3dgs-packages" python -c 'from diff_gaussian_rasterization import _C; print("3DGS rasterizer ready")'
