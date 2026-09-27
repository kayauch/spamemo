import time
from pathlib import Path

import numpy as np
import torch

from spatial_recon.types import ReconstructionResult


class VGGTOmegaAdapter:
    def __init__(self, checkpoint: str, image_resolution: int = 512, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.image_resolution = image_resolution
        self.device = torch.device(device)
        self.model = None
        self.load_metrics: dict[str, float | str] = {}

    def load(self) -> None:
        from vggt_omega.models import VGGTOmega

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = VGGTOmega().eval()
        model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
        self.model = model.to(self.device)
        torch.cuda.synchronize()
        self.load_metrics = {
            "model": "VGGT-Omega-1B",
            "load_seconds": time.perf_counter() - started,
            "load_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }

    def infer(self, image_paths: list[Path]) -> ReconstructionResult:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        if self.model is None:
            self.load()
        images = load_and_preprocess_images(
            [str(p) for p in image_paths],
            image_resolution=self.image_resolution,
        ).to(self.device)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            pred = self.model(images)
            extrinsics, intrinsics = encoding_to_camera(
                pred["pose_enc"], pred["images"].shape[-2:]
            )
        torch.cuda.synchronize()
        metrics = dict(self.load_metrics)
        metrics.update({
            "frames": len(image_paths),
            "inference_seconds": time.perf_counter() - started,
            "inference_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "inference_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        })

        depth = pred["depth"].detach().float().cpu().numpy()
        confidence = pred["depth_conf"].detach().float().cpu().numpy()
        frame_images = pred["images"].detach().float().cpu().numpy()
        extrinsics = extrinsics.detach().float().cpu().numpy()
        intrinsics = intrinsics.detach().float().cpu().numpy()
        for name, value in (
            ("depth", depth),
            ("confidence", confidence),
            ("images", frame_images),
            ("extrinsics", extrinsics),
            ("intrinsics", intrinsics),
        ):
            if value.shape[0] == 1:
                locals()[name] = value[0]
        return ReconstructionResult(
            depth=depth[0] if depth.shape[0] == 1 else depth,
            confidence=confidence[0] if confidence.shape[0] == 1 else confidence,
            images=frame_images[0] if frame_images.shape[0] == 1 else frame_images,
            extrinsics=extrinsics[0] if extrinsics.shape[0] == 1 else extrinsics,
            intrinsics=intrinsics[0] if intrinsics.shape[0] == 1 else intrinsics,
            metrics=metrics,
        )
