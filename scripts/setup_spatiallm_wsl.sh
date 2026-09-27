#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_env="$task_root/third_party/spatiallm-env"
if [ ! -e "$task_env/bin/python" ]; then
  "$task_root/third_party/2dgs-env/bin/python" -m venv "$task_env"
  "$task_env/bin/python" - "$task_root" <<'PY'
import pathlib,sys,sysconfig
root=pathlib.Path(sys.argv[1])
shared=next((root/'third_party/2dgs-env/lib').glob('python*/site-packages'))
(pathlib.Path(sysconfig.get_paths()['purelib'])/'shared_torch_runtime.pth').write_text(str(shared)+'\n')
PY
fi
"$task_env/bin/python" -c "import torch; assert torch.__version__ == '2.7.1+cu128', 'This setup expects the tested 2DGS torch 2.7.1+cu128 runtime'"
"$task_env/bin/python" -m pip install -r "$task_root/scripts/spatiallm-requirements.txt"
"$task_env/bin/python" -m pip install 'torch-scatter==2.1.2+pt27cu128' -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
"$task_env/bin/python" - <<'PY'
import torch,transformers,spconv.pytorch,torch_scatter,open3d
print('torch',torch.__version__,'GPU',torch.cuda.get_device_name(0),'transformers',transformers.__version__)
print(torch_scatter.scatter_add(torch.ones(4,device='cuda'),torch.tensor([0,0,1,1],device='cuda')))
PY
