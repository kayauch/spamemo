"""Small real-model CPU smoke; does not use the reconstruction fallback."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from spatial_recon.adapters.vggt import VGGTAdapter
from spatial_recon.io import save_result
from spatial_recon.geometry import save_point_cloud

root = Path(__file__).resolve().parents[1]
output = root / "data" / "vggt-cpu-smoke"
output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
video = cv2.VideoCapture(str(root / "test-assets" / "fake-camera.y4m"))
frames = []
for index in (0, 30):
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = video.read()
    assert ok, f"Cannot read fixture frame {index}"
    path = output / f"frame-{index}.jpg"
    assert cv2.imwrite(str(path), frame)
    frames.append(path)
video.release()
adapter = VGGTAdapter(root / "checkpoints/vggt/model.pt", root / "third_party/vggt", device="cpu")
print("Loading VGGT-1B and inferring two fixture frames on CPU...", flush=True)
result = adapter.infer(frames)
assert np.isfinite(result.depth).all() and float(np.std(result.depth)) > 0
save_result(result, output)
count = save_point_cloud(result, output / "scene.ply", stride=8)
print(json.dumps({"metrics": result.metrics, "depth_shape": list(result.depth.shape), "points": count}, indent=2), flush=True)
