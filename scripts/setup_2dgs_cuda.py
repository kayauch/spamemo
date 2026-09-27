"""Install NVIDIA redistributable compiler into a local directory (Linux only)."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

root = Path(__file__).resolve().parents[1] / 'third_party'
root.mkdir(exist_ok=True)
if not (root / 'cuda-redistrib.json').exists():
    urllib.request.urlretrieve('https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json', root / 'cuda-redistrib.json')
manifest = json.loads((root / 'cuda-redistrib.json').read_text())
target = root / 'cuda-12.8'
target.mkdir(exist_ok=True)
for name in ('cuda_nvcc', 'cuda_cudart', 'cuda_cccl'):
    spec = manifest[name]['linux-x86_64']
    archive = root / Path(spec['relative_path']).name
    if not archive.exists():
        urllib.request.urlretrieve('https://developer.download.nvidia.com/compute/cuda/redist/' + spec['relative_path'], archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != spec['sha256']:
        raise RuntimeError(f'Hash mismatch: {name}')
    with tarfile.open(archive) as package:
        package.extractall(root, filter='data')
    folder = root / archive.name.removesuffix('.tar.xz')
    shutil.copytree(folder, target, dirs_exist_ok=True)
lib = target / 'lib64'
if not lib.exists():
    lib.symlink_to('lib', target_is_directory=True)
print(target)
