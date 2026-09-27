"""Repair DLL discovery using VC++ DLLs already placed in this venv."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

if sys.prefix == sys.base_prefix:
    raise SystemExit("Run this using .venv/Scripts/python.exe.")
scripts = Path(sysconfig.get_path("scripts"))
site = Path(sysconfig.get_path("purelib"))
if os.name == "nt" and (scripts / "msvcp140.dll").exists():
    (site / "_spatial_local_runtime.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "_dll_handle = None\n"
        "if os.name == 'nt':\n"
        "    _runtime = Path(sys.prefix) / 'Scripts'\n"
        "    if (_runtime / 'msvcp140.dll').exists():\n"
        "        _dll_handle = os.add_dll_directory(str(_runtime))\n",
        encoding="utf-8")
    (site / "spatial_local_runtime.pth").write_text("import _spatial_local_runtime\n", encoding="utf-8")
import imageio_ffmpeg
target = scripts / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
if not target.exists():
    shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), target)
subprocess.run([sys.executable, "-c", "import torch, open3d; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'open3d', open3d.__version__)"], check=True)
print("Runtime discovery and local ffmpeg are configured.")
