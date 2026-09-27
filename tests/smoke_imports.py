import torch
import open3d
import vggt_omega
from depth_anything_3.api import DepthAnything3

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("vram_gib", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print("open3d", open3d.__version__)
print("DepthAnything3", DepthAnything3.__name__)
print("vggt_omega", vggt_omega.__name__)
