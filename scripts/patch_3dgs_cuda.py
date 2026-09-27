"""Explicit standard integer header for CUDA 12.8/GCC 13; no algorithm changes."""
from pathlib import Path

path = Path(__file__).resolve().parents[1]/'third_party/3dgs/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h'
source = path.read_text()
if '#include <cstdint>' not in source:
    path.write_text(source.replace('#include <iostream>', '#include <cstdint>\n#include <iostream>'))
