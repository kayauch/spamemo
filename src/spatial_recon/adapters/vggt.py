from __future__ import annotations

import sys
from contextlib import nullcontext
import time
from pathlib import Path

import numpy as np
import torch

from spatial_recon.types import ReconstructionResult


class VGGTAdapter:
    def __init__(
        self,
        checkpoint: str | Path,
        repo_path: str | Path = "/workspace/third_party/vggt",
        device: str = "cuda",
    ):
        self.checkpoint = Path(checkpoint)
        self.repo_path = Path(repo_path)
        self.device = torch.device(device)
        self.model = None
        self.load_metrics: dict[str, float | str] = {}

    def _imports(self):
        repo = str(self.repo_path)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        return VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri

    def load(self) -> None:
        VGGT, _, _ = self._imports()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"VGGT checkpoint not found: {self.checkpoint}")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()

        model = VGGT()
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_metrics = {
            "model": "VGGT-1B",
            "device": str(self.device),
            "checkpoint": str(self.checkpoint),
            "load_seconds": time.perf_counter() - started,
            "load_peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30 if self.device.type == "cuda" else 0.0,
            "load_peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0.0,
        }

    def infer(self, image_paths: list[Path]) -> ReconstructionResult:
        _, load_and_preprocess_images, pose_encoding_to_extri_intri = self._imports()
        if self.model is None:
            self.load()

        images = load_and_preprocess_images(
            [str(path) for path in image_paths]
        ).to(self.device)

        precision_context = nullcontext()
        if self.device.type == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability(self.device)[0] >= 8
                else torch.float16
            )
            precision_context = torch.autocast(device_type="cuda", dtype=dtype)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()

        with torch.inference_mode():
            with precision_context:
                pred = self.model(images)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pred["pose_enc"],
                images.shape[-2:],
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        metrics = dict(self.load_metrics)
        metrics.update({
            "frames": len(image_paths),
            "inference_seconds": time.perf_counter() - started,
            "inference_peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30 if self.device.type == "cuda" else 0.0,
            "inference_peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0.0,
            "input_height": int(images.shape[-2]),
            "input_width": int(images.shape[-1]),
        })

        depth = pred["depth"].detach().float().cpu().numpy()
        confidence = pred.get("depth_conf")
        if confidence is not None:
            confidence = confidence.detach().float().cpu().numpy()
        frame_images = images.detach().float().cpu().numpy()
        extrinsics = extrinsics.detach().float().cpu().numpy()
        intrinsics = intrinsics.detach().float().cpu().numpy()

        if depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if confidence is not None and confidence.shape[0] == 1:
            confidence = confidence[0]
        if confidence is not None and confidence.ndim == 4 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        if frame_images.shape[0] == 1:
            frame_images = frame_images[0]
        if extrinsics.shape[0] == 1:
            extrinsics = extrinsics[0]
        if intrinsics.shape[0] == 1:
            intrinsics = intrinsics[0]

        return ReconstructionResult(
            depth=np.asarray(depth),
            confidence=np.asarray(confidence) if confidence is not None else None,
            extrinsics=np.asarray(extrinsics),
            intrinsics=np.asarray(intrinsics),
            images=np.asarray(frame_images),
            metrics=metrics,
        )
