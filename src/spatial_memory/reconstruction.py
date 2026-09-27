from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import UploadFile
from PIL import Image

from spatial_recon.adapters import DA3Adapter, VGGTAdapter, VGGTOmegaAdapter
from spatial_recon.geometry import save_point_cloud
from spatial_recon.io import save_result
from spatial_recon.types import ReconstructionResult
from spatial_recon.video import extract_video_frames_with_metadata

from .models import Scene
from .spatial_model import build_spatial_model
from .storage import SpatialMemoryDB
from .capture_inputs import CaptureInput, constrain_panorama_cameras, inherited_inputs, mask_vggt_padding, prepare_capture


class ReconstructionService:
    def __init__(
        self,
        db: SpatialMemoryDB,
        root_dir: str | Path = "/results/scenes",
        backend: str = "vggt",
        model_id: str = "depth-anything/DA3-LARGE-1.1",
        vggt_checkpoint: str | Path = "/results/checkpoints/vggt/model.pt",
        vggt_repo: str | Path = "/workspace/third_party/vggt",
        vggt_omega_checkpoint: str | Path | None = None,
        vggt_omega_resolution: int = 512,
    ):
        self.db = db
        self.root_dir = Path(root_dir)
        self.backend = backend
        self.model_id = model_id
        self.vggt_checkpoint = Path(vggt_checkpoint)
        self.vggt_repo = Path(vggt_repo)
        self.vggt_omega_checkpoint = (
            Path(vggt_omega_checkpoint)
            if vggt_omega_checkpoint
            else None
        )
        self.vggt_omega_resolution = vggt_omega_resolution
        self._adapter = None
        self._lock = threading.Lock()

    @property
    def model_label(self) -> str:
        if self.backend == "vggt":
            return "VGGT-1B"
        if self.backend == "vggt-omega":
            return "VGGT-Omega-1B-512"
        return self.model_id

    @property
    def adapter(self):
        if self._adapter is not None:
            return self._adapter

        if self.backend == "vggt":
            self._adapter = VGGTAdapter(
                checkpoint=self.vggt_checkpoint,
                repo_path=self.vggt_repo,
                device=os.environ.get("RECONSTRUCTION_DEVICE", "cuda"),
            )
        elif self.backend == "da3":
            self._adapter = DA3Adapter(model_id=self.model_id)
        elif self.backend == "vggt-omega":
            if self.vggt_omega_checkpoint is None:
                raise ValueError(
                    "VGGT-Omega backend requires a checkpoint path"
                )
            self._adapter = VGGTOmegaAdapter(
                checkpoint=str(self.vggt_omega_checkpoint),
                image_resolution=self.vggt_omega_resolution,
            )
        else:
            raise ValueError(
                f"Unsupported reconstruction backend: {self.backend}"
            )
        return self._adapter

    def _fallback_infer(self, frames: list[Path]) -> ReconstructionResult:
        imgs = []
        for p in frames:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                if w > 640 or h > 480:
                    img_rgb.thumbnail((640, 480))
                imgs.append(np.asarray(img_rgb))

        imgs_arr = np.stack(imgs, axis=0)
        n, h, w, _ = imgs_arr.shape

        depth = np.full((n, h, w), 2.0, dtype=np.float32)
        fx = float(w)
        fy = float(w)
        cx = float(w / 2.0)
        cy = float(h / 2.0)
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        intrinsics = np.repeat(k[None], n, axis=0)

        extrinsics = np.zeros((n, 3, 4), dtype=np.float32)
        for i in range(n):
            angle = (i / max(n - 1, 1) - 0.5) * 0.4
            r = np.array([
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ], dtype=np.float32)
            t = np.array([(i / max(n - 1, 1) - 0.5) * 0.6, 0.0, 0.0], dtype=np.float32)
            extrinsics[i, :3, :3] = r
            extrinsics[i, :3, 3] = t

        confidence = np.full((n, h, w), 0.9, dtype=np.float32)
        metrics = {
            "model": "local-fallback",
            "frames": n,
            "backend": "synthetic_pinhole",
        }
        return ReconstructionResult(
            depth=depth,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            images=imgs_arr,
            confidence=confidence,
            metrics=metrics,
        )

    def reconstruct_upload(
        self,
        upload: UploadFile,
        name: str,
        description: str | None = None,
        sample_fps: float = 2.0,
        max_frames: int = 60,
        point_stride: int = 4,
        quality_filter: bool = False,
    ) -> tuple[Scene, dict]:
        scene_id = str(uuid4())
        scene_dir = self.root_dir / scene_id
        input_dir = scene_dir / "input"
        frames_dir = scene_dir / "frames"
        reconstruction_dir = scene_dir / "reconstruction"
        input_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(upload.filename or "scene.mp4").suffix or ".mp4"
        video_path = input_dir / f"scene{suffix}"
        upload.file.seek(0)
        with video_path.open("wb") as out:
            shutil.copyfileobj(upload.file, out)

        extracted = extract_video_frames_with_metadata(
            video_path,
            frames_dir,
            sample_fps=sample_fps,
            max_frames=max_frames,
            quality_filter=quality_filter,
        )
        frames = [item.path for item in extracted]
        if len(frames) < 2:
            raise ValueError("At least two usable video frames are required")

        manifest_path = frames_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "source_video_path": str(video_path),
                    "sample_fps": sample_fps,
                    "quality_filter": quality_filter,
                    "frames": [
                        {
                            "index": index,
                            "name": item.path.name,
                            "source_frame_index": item.source_frame_index,
                            "timestamp_seconds": item.timestamp_seconds,
                            "sharpness": item.sharpness,
                            "motion_score": item.motion_score,
                        }
                        for index, item in enumerate(extracted)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        with self._lock:
            try:
                result = self.adapter.infer(frames)
            except Exception as exc:
                logging.getLogger("spatial_memory").warning(
                    "Reconstruction adapter %s unavailable (%s); falling back to local lightweight reconstruction.",
                    self.backend,
                    exc,
                )
                result = self._fallback_infer(frames)
                result.metrics["fallback"] = True
                result.metrics["fallback_reason"] = str(exc)

        npz_path = save_result(result, reconstruction_dir)
        ply_path = reconstruction_dir / "scene.ply"
        point_count = save_point_cloud(result, ply_path, stride=point_stride)
        result.metrics["point_count"] = point_count

        spatial_model_path = None
        try:
            spatial_model = build_spatial_model(
                point_cloud_path=ply_path,
                output_dir=scene_dir / "spatial_model",
                scene_id=scene_id,
                target_triangles=20_000,
            )
            spatial_model_path = spatial_model["manifest_path"]
            if result.metrics.get("model") == "VGGT-1B" and not result.metrics.get("fallback"):
                from .texturing import build_textured_scene
                try:
                    result.metrics["texture"] = build_textured_scene(spatial_model_path)
                except (ValueError, RuntimeError, OSError) as exc:
                    result.metrics["texture_error"] = str(exc)
            result.metrics["spatial_model"] = {
                "manifest_path": spatial_model_path,
                "component_count": spatial_model["component_count"],
                "shell_vertices": spatial_model["shell"]["vertices"],
                "shell_triangles": spatial_model["shell"]["triangles"],
                "total_seconds": spatial_model["metrics"]["total_seconds"],
            }
        except Exception as exc:
            result.metrics["spatial_model_error"] = str(exc)

        save_result(result, reconstruction_dir)

        scene = Scene(
            id=scene_id,
            name=name,
            description=description,
            point_cloud_path=str(ply_path),
            spatial_model_path=spatial_model_path,
            reconstruction_npz=str(npz_path),
            frames_dir=str(frames_dir),
            source_video_path=str(video_path),
            reconstruction_model=str(result.metrics.get("model", self.model_label)),
            frame_count=len(frames),
        )
        self.db.upsert_scene(scene)

        metrics = dict(result.metrics)
        metrics.update({
            "scene_id": scene_id,
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "frame_count": len(frames),
            "point_stride": point_stride,
            "quality_filter": quality_filter,
            "frame_manifest": str(manifest_path),
        })
        return scene, metrics

    def reconstruct_capture(
        self, capture_inputs: list[CaptureInput | dict | Path], name: str,
        description: str | None = None, *, parent_scene_id: str | None = None,
        sample_fps: float = 4.0, max_frames: int = 60, point_stride: int = 4,
        quality_filter: bool = True, capture_metadata: dict | None = None,
    ) -> tuple[Scene, dict]:
        """Reconstruct a new revision from all inputs in one coordinate system.

        No result, files, or saved-object positions in the parent are modified.
        Inference failure is reported instead of generating synthetic geometry.
        """
        if not 1 <= point_stride <= 32:
            raise ValueError("point_stride must be between 1 and 32")
        inputs = list(capture_inputs)
        if parent_scene_id:
            parent = self.db.get_scene(parent_scene_id)
            if parent is None:
                raise ValueError("Parent scene does not exist")
            if getattr(parent, "reconstruction_model", None) == "calibrated-rgbd":
                raise ValueError("Calibrated depth scenes require an aligned sensor bundle for revision; uncalibrated videos cannot be merged into metric coordinates")
            inputs = inherited_inputs(parent, max_frames) + inputs
        scene_id = str(uuid4())
        scene_dir = self.root_dir / scene_id
        frames, records, capture = prepare_capture(
            inputs, scene_dir, max_frames=max_frames, sample_fps=sample_fps,
            quality_filter=quality_filter, parent_scene_id=parent_scene_id,
            capture_metadata=capture_metadata,
        )
        with self._lock:
            result = self.adapter.infer(frames)
        if result.metrics.get("fallback") or result.metrics.get("backend") == "synthetic_pinhole":
            raise RuntimeError("Capture reconstruction requires real model inference; synthetic fallback is disabled")
        if result.frame_count != len(frames) or not np.isfinite(result.extrinsics).all():
            raise ValueError("Reconstruction returned invalid cameras or a mismatched frame count")
        if self.backend == "vggt":
            mask_vggt_padding(result, frames, records)
            frame_manifest = scene_dir / "frames" / "manifest.json"
            frame_payload = json.loads(frame_manifest.read_text(encoding="utf-8"))
            frame_payload["frames"] = records
            frame_manifest.write_text(json.dumps(frame_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        constrain_panorama_cameras(result, records)
        result.metrics.setdefault("model", self.model_label)
        result.metrics["capture"] = capture
        result.metrics["parent_scene_id"] = parent_scene_id
        result.metrics["joint_inference"] = True
        result.metrics["fallback"] = False
        if (capture_metadata or {}).get("refine_cameras"):
            from .camera_refinement import refine_capture_cameras
            result.metrics["camera_refinement"] = refine_capture_cameras(result, records)
        return self._save_capture_result(result, scene_id, name, description, frames, capture, point_stride, parent_scene_id)

    def reconstruct_sensor_capture(
        self, bundle_path: str | Path, name: str, description: str | None = None,
        *, point_stride: int = 4, max_frames: int = 60, capture_metadata: dict | None = None,
    ) -> tuple[Scene, dict]:
        """Import an already calibrated metric RGB-D capture without inference."""
        from .sensor_import import import_sensor_bundle
        if not 1 <= point_stride <= 32:
            raise ValueError("point_stride must be between 1 and 32")
        scene_id = str(uuid4())
        result, frames, capture = import_sensor_bundle(Path(bundle_path), self.root_dir / scene_id, max_frames=max_frames, capture_metadata=capture_metadata)
        return self._save_capture_result(result, scene_id, name, description, frames, capture, point_stride, None)

    def _save_capture_result(
        self, result: ReconstructionResult, scene_id: str, name: str,
        description: str | None, frames: list[Path], capture: dict,
        point_stride: int, parent_scene_id: str | None,
    ) -> tuple[Scene, dict]:
        scene_dir = self.root_dir / scene_id
        reconstruction_dir = scene_dir / "reconstruction"
        reconstruction_dir.mkdir(parents=True, exist_ok=True)
        # The geometric consensus filter counts frame IDs. Do not use it for
        # virtual panorama faces, which share a physical camera center.
        has_panorama = any(item.get("kind") == "panorama" for item in capture["sources"])
        ply_path = reconstruction_dir / "scene.ply"
        result.metrics["point_count"] = save_point_cloud(result, ply_path, stride=point_stride, dynamic_filter=not has_panorama)
        npz_path = save_result(result, reconstruction_dir)
        spatial_model_path = None
        try:
            spatial_model = build_spatial_model(point_cloud_path=ply_path, output_dir=scene_dir / "spatial_model", scene_id=scene_id, target_triangles=20_000)
            spatial_model_path = spatial_model["manifest_path"]
            result.metrics["spatial_model"] = {"manifest_path": spatial_model_path, "component_count": spatial_model["component_count"]}
        except Exception as exc:
            result.metrics["spatial_model_error"] = str(exc)
        save_result(result, reconstruction_dir)
        manifest_path = scene_dir / "input" / "capture.json"
        first_video = next((item["path"] for item in capture["sources"] if item.get("kind") == "video"), None)
        scene = Scene(
            id=scene_id, name=name, description=description, point_cloud_path=str(ply_path),
            spatial_model_path=spatial_model_path, reconstruction_npz=str(npz_path),
            frames_dir=str(scene_dir / "frames"), source_video_path=first_video,
            reconstruction_model=str(result.metrics.get("model", self.model_label)), frame_count=len(frames),
            parent_scene_id=parent_scene_id, capture_manifest_path=str(manifest_path),
        )
        self.db.upsert_scene(scene)
        metrics = {**result.metrics, "scene_id": scene_id, "frame_count": len(frames), "point_stride": point_stride, "frame_manifest": str(scene_dir / "frames" / "manifest.json"), "capture_manifest": str(manifest_path)}
        return scene, metrics
