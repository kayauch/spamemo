"""Make legacy headers explicit for CUDA 12.8 / GCC 13. No algorithm changes."""
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'third_party/2dgs/submodules'
patches={
    'diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h': ['cstdint'],
    'simple-knn/simple_knn.cu': ['cstdint','cfloat'],
}
for name,headers in patches.items():
    path=root/name
    text=path.read_text()
    for header in headers:
        line=f'#include <{header}>'
        if line not in text:
            index=text.index('#include')
            text=text[:index]+line+'\n'+text[index:]
    path.write_text(text)
    print(name)
