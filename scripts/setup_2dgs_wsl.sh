#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="$task_root/third_party/2dgs-env/bin/python"
mkdir -p "$task_root/third_party"
if [ ! -d "$task_root/third_party/2dgs/.git" ]; then
  git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git "$task_root/third_party/2dgs"
  git -C "$task_root/third_party/2dgs" checkout f3e3b9fa67bbd1c75e05167ff37391d8dab2a678
  git -C "$task_root/third_party/2dgs" submodule update --init --recursive
fi
if [ ! -x "$task_python" ]; then
  python3 -m venv "$task_root/third_party/2dgs-env"
fi
if ! "$task_python" -c 'import torch; assert torch.__version__ == "2.7.1+cu128"' 2>/dev/null; then
  "$task_python" -m pip install --no-compile torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
fi
if [ ! -x "$task_root/third_party/cuda-12.8/bin/nvcc" ]; then
  python3 "$task_root/scripts/setup_2dgs_cuda.py"
fi
export CUDA_HOME="$task_root/third_party/cuda-12.8"
export PATH="$CUDA_HOME/bin:$task_root/third_party/2dgs-env/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
"$task_python" -m pip install --no-compile numpy==1.26.4 plyfile==1.1.3 tqdm==4.70.1 matplotlib==3.11.2 opencv-python-headless==4.11.0.86 ninja==1.13.2 wheel==0.48.0
"$task_python" "$task_root/scripts/patch_2dgs_cuda.py"
"$task_python" -m pip install --no-build-isolation "$task_root/third_party/2dgs/submodules/diff-surfel-rasterization" "$task_root/third_party/2dgs/submodules/simple-knn" > "$task_root/third_party/2dgs-build.log" 2>&1
"$task_python" -c 'import torch; from diff_surfel_rasterization import _C; from simple_knn._C import distCUDA2; print(torch.__version__, torch.cuda.get_device_name()); print(distCUDA2(torch.rand(100,3,device="cuda")).shape)'
