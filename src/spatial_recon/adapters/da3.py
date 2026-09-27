import time
from pathlib import Path

import numpy as np
import torch

from spatial_recon.types import ReconstructionResult


class DA3Adapter:
    def __init__(self, model_id: str = "depth-anything/DA3-LARGE-1.1", device: str = "cuda"):
        self.model_id = model_id
        self.device = torch.device(device)
        self.model = None
        self.load_metrics: dict[str, float | str] = {}

    def load(self) -> None:
        from depth_anything_3.api import DepthAnything3

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.model = DepthAnything3.from_pretrained(self.model_id).to(self.device).eval()
        torch.cuda.synchronize()
        self.load_metrics = {
            "model": self.model_id,
            "load_seconds": time.perf_counter() - started,
            "load_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }

    def infer(self, image_paths: list[Path]) -> ReconstructionResult:
        if self.model is None:
            self.load()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = self.model.inference([str(p) for p in image_paths])
        torch.cuda.synchronize()
        metrics = dict(self.load_metrics)
        metrics.update({
            "frames": len(image_paths),
            "inference_seconds": time.perf_counter() - started,
            "inference_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "inference_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        })
        return ReconstructionResult(
            depth=np.asarray(prediction.depth),
            confidence=np.asarray(prediction.conf) if prediction.conf is not None else None,
            extrinsics=np.asarray(prediction.extrinsics),
            intrinsics=np.asarray(prediction.intrinsics),
            images=np.asarray(prediction.processed_images),
            metrics=metrics,
        )
