"""Portable calibrated RGB-D ZIP import; no device SDK or cloud service."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

import cv2
import numpy as np
from PIL import Image

from spatial_recon.types import ReconstructionResult


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = {}
    total = 0
    if len(archive.infolist()) > 4096:
        raise ValueError("Sensor bundle contains too many files")
    for member in archive.infolist():
        name = member.filename
        original_name = member.orig_filename
        path = PurePosixPath(name)
        if "\\" in original_name or "\x00" in original_name or ":" in name or path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe path in sensor bundle")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError("Symbolic links are not allowed in sensor bundles")
        if member.flag_bits & 1:
            raise ValueError("Encrypted sensor bundles are not supported")
        if member.is_dir():
            continue
        if name in members:
            raise ValueError("Duplicate sensor bundle member")
        total += member.file_size
        if member.file_size > 128 * 1024**2 or total > 1024**3:
            raise ValueError("Sensor bundle exceeds decompressed size limits")
        members[name] = member
    if "manifest.json" not in members or members["manifest.json"].file_size > 2 * 1024**2:
        raise ValueError("A root manifest.json (at most 2 MB) is required")
    return members


def _intrinsics(value: object) -> np.ndarray:
    intrinsic = np.asarray(value, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
        raise ValueError("Sensor intrinsics must be a finite 3 x 3 pinhole matrix")
    if not np.allclose(intrinsic[2], [0, 0, 1]) or not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], 0):
        raise ValueError("Sensor intrinsics require zero skew and homogeneous last row")
    return intrinsic


def _load_bounded_array(stream) -> np.ndarray:
    """Validate the NPY header before numpy can allocate an attacker-sized array."""
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Sensor arrays require NPY format version 1 or 2")
    if len(shape) != 2 or any(size <= 0 for size in shape) or shape[0] * shape[1] > 16_000_000 or not np.issubdtype(dtype, np.floating):
        raise ValueError("Sensor arrays must be bounded H x W floating-point arrays")
    if dtype.itemsize > 8:
        raise ValueError("Sensor arrays require float16, float32 or float64")
    stream.seek(0)
    return np.load(stream, allow_pickle=False)


def _extrinsics(value: object) -> np.ndarray:
    extrinsic = np.asarray(value, dtype=np.float64)
    if extrinsic.shape == (4, 4):
        if not np.allclose(extrinsic[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous sensor pose")
        extrinsic = extrinsic[:3]
    if extrinsic.shape != (3, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("Sensor extrinsics must be a finite 3 x 4 or 4 x 4 matrix")
    rotation = extrinsic[:, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError("Sensor pose rotation must be orthonormal and right-handed")
    return extrinsic


def import_sensor_bundle(bundle_path: Path, scene_dir: Path, *, max_frames: int = 60, capture_metadata: dict | None = None) -> tuple[ReconstructionResult, list[Path], dict]:
    """Accept schema v1, aligned RGB + depth NPY, meter world-to-camera poses.

    manifest.json:
      {"schema_version":1,"depth_unit":"meters",
       "pose_convention":"world_to_camera_opencv","frames":[
        {"rgb":"rgb/000.jpg","depth":"depth/000.npy",
         "intrinsics":[[fx,0,cx],[0,fy,cy],[0,0,1]],
         "extrinsics":[[...],[...],[...]],"confidence":"conf/000.npy"}]}

    Depth is optical-axis z, RGB-aligned and already undistorted. Confidence,
    if supplied, is [0,1]; invalid/no-return depth is zero. All poses share one
    metric coordinate frame. Native phone depth exports require conversion.
    """
    if not 2 <= max_frames <= 240:
        raise ValueError("max_frames must be between 2 and 240")
    input_dir, frames_dir = scene_dir / "input", scene_dir / "frames"
    with zipfile.ZipFile(bundle_path) as archive:
        members = _safe_members(archive)
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != 1 or manifest.get("depth_unit") != "meters" or manifest.get("pose_convention") != "world_to_camera_opencv":
            raise ValueError("Sensor manifest requires schema_version 1, depth_unit meters, pose_convention world_to_camera_opencv")
        frame_records = manifest.get("frames")
        if not isinstance(frame_records, list) or not 2 <= len(frame_records) <= 1000:
            raise ValueError("Sensor manifest must contain 2 to 1000 frames")
        indexes = np.linspace(0, len(frame_records) - 1, min(max_frames, len(frame_records)), dtype=int)
        input_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        saved_bundle = input_dir / "sensor-capture.zip"
        shutil.copy2(bundle_path, saved_bundle)
        depths, colors, confidences, intrinsics, extrinsics, records = [], [], [], [], [], []
        common_shape = None
        for output_index, source_index in enumerate(indexes):
            frame = frame_records[int(source_index)]
            for key in ("rgb", "depth"):
                if frame.get(key) not in members:
                    raise ValueError(f"Sensor frame is missing its {key} file")
            if not frame["depth"].lower().endswith(".npy"):
                raise ValueError("Sensor depth must use non-pickled NPY float arrays in meters")
            intrinsic, extrinsic = _intrinsics(frame.get("intrinsics")), _extrinsics(frame.get("extrinsics"))
            with archive.open(frame["depth"]) as stream:
                depth = _load_bounded_array(stream)
            if depth.ndim != 2 or depth.size > 16_000_000 or not np.issubdtype(depth.dtype, np.floating):
                raise ValueError("Sensor depth must be a bounded H x W floating-point array")
            if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth > 100):
                raise ValueError("Sensor depths must be finite meters in [0,100]; use zero for missing depth")
            with archive.open(frame["rgb"]) as stream, Image.open(stream) as image:
                if image.size != (depth.shape[1], depth.shape[0]):
                    raise ValueError("Sensor depth must already be aligned to RGB resolution")
                # EXIF rotations would invalidate supplied intrinsics; forbid
                # them until the exporter rotates RGB, depth and calibration.
                if image.getexif().get(274, 1) != 1:
                    raise ValueError("Sensor RGB orientation must be normalized together with its calibration")
                rgb = np.asarray(image.convert("RGB"))
            if not (0 <= intrinsic[0, 2] < depth.shape[1] and 0 <= intrinsic[1, 2] < depth.shape[0]):
                raise ValueError("Sensor principal point must lie inside the RGB image")
            confidence = np.ones(depth.shape, dtype=np.float32)
            if frame.get("confidence"):
                if frame["confidence"] not in members:
                    raise ValueError("Missing sensor confidence file")
                with archive.open(frame["confidence"]) as stream:
                    confidence = _load_bounded_array(stream)
                if confidence.shape != depth.shape or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
                    raise ValueError("Sensor confidence must match depth and contain values in [0,1]")
            height, width = depth.shape
            scale = min(1.0, 640 / max(width, height))
            target = (max(1, round(width * scale)), max(1, round(height * scale)))
            if common_shape is None:
                common_shape = target
            elif target != common_shape:
                raise ValueError("Sensor frames must use the same image dimensions")
            intrinsic[0] *= target[0] / width
            intrinsic[1] *= target[1] / height
            rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth.astype(np.float32), target, interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence.astype(np.float32), target, interpolation=cv2.INTER_NEAREST)
            valid = (depth > 0) & (confidence > 0)
            if int(valid.sum()) < 16:
                raise ValueError("Sensor frame has fewer than 16 valid depth pixels")
            # Existing VGGT geometry consumers use confidence >1.2. Preserve
            # ordering while exposing the conversion in metrics and manifest.
            confidence = np.where(valid, 1 + 2 * confidence, 0).astype(np.float32)
            depth = np.where(valid, depth, 0).astype(np.float32)
            output = frames_dir / f"{output_index:06d}.jpg"
            Image.fromarray(rgb).save(output, quality=96)
            colors.append(rgb)
            depths.append(depth)
            confidences.append(confidence)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            records.append({"index": output_index, "name": output.name, "kind": "sensor", "source_kind": "sensor", "source_id": f"sensor-{source_index:06d}", "source_frame_index": int(source_index), "timestamp_seconds": frame.get("timestamp_seconds"), "source_path": str(saved_bundle)})
    capture = {"schema_version": 1, "sources": [{"id": "sensor-bundle", "path": str(saved_bundle), "kind": "sensor"}], "options": capture_metadata or {}, "frame_count": len(records), "coordinate_units": "meters", "pose_convention": "world_to_camera_opencv", "confidence_mapping": "valid input c in (0,1] -> 1+2c; invalid -> 0", "limitations": ["RGB/depth registration and shared metric poses are supplied by the exporter; they cannot be inferred or verified from the ZIP alone."]}
    (input_dir / "capture.json").write_text(json.dumps(capture, ensure_ascii=False, indent=2), encoding="utf-8")
    (frames_dir / "manifest.json").write_text(json.dumps({"schema_version": 2, "frames": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    result = ReconstructionResult(depth=np.stack(depths), extrinsics=np.asarray(extrinsics, dtype=np.float32), intrinsics=np.asarray(intrinsics, dtype=np.float32), confidence=np.stack(confidences), images=np.stack(colors), metrics={"model": "calibrated-rgbd", "fallback": False, "coordinate_units": "meters", "capture": capture})
    return result, [frames_dir / record["name"] for record in records], capture
