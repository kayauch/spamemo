"""Prepare immutable multi-source captures for a single reconstruction call."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import hashlib
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from spatial_recon.types import ReconstructionResult
from spatial_recon.video import extract_video_frames_with_metadata

from .capture_options import MAX_CAPTURE_FRAMES

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class CaptureInput:
    path: Path
    kind: str = "photo"
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_input(value: CaptureInput | dict | Path | str) -> CaptureInput:
    if isinstance(value, CaptureInput):
        result = CaptureInput(Path(value.path), value.kind, dict(value.metadata))
    elif isinstance(value, dict):
        result = CaptureInput(Path(value["path"]), str(value.get("kind", "photo")), dict(value.get("metadata") or {}))
    else:
        path = Path(value)
        result = CaptureInput(path, "video" if path.suffix.lower() in VIDEO_SUFFIXES else "photo")
    if result.kind not in {"video", "photo", "panorama"}:
        raise ValueError(f"Unsupported capture kind: {result.kind}")
    if not result.path.is_file():
        raise ValueError(f"Capture input is missing: {result.path.name}")
    expected = VIDEO_SUFFIXES if result.kind == "video" else PHOTO_SUFFIXES
    if result.path.suffix.lower() not in expected:
        raise ValueError(f"Unsupported file type for {result.kind}: {result.path.suffix}")
    return result


def inherited_inputs(parent: Any, max_frames: int) -> list[CaptureInput]:
    """Read parent originals; never change its files or entity coordinates."""
    manifest_path = getattr(parent, "capture_manifest_path", None)
    if not manifest_path and parent.frames_dir:
        manifest_path = str(Path(parent.frames_dir).parent / "input" / "capture.json")
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        sources = manifest.get("sources", [])
        if sources and all(Path(item["path"]).is_file() for item in sources):
            return [CaptureInput(Path(item["path"]), item["kind"], {**item.get("metadata", {}), "parent_scene_id": parent.id}) for item in sources]
    paths = sorted(Path(parent.frames_dir).glob("*.jpg")) if parent.frames_dir else []
    # Still photos remain mandatory even if their original source manifest was
    # lost. Only video-derived/legacy frames are eligible for temporal sampling.
    photo_names = set()
    frame_manifest = Path(parent.frames_dir) / "manifest.json" if parent.frames_dir else None
    if frame_manifest and frame_manifest.is_file():
        records = json.loads(frame_manifest.read_text(encoding="utf-8")).get("frames", [])
        photo_names = {record.get("name") for record in records
                       if record.get("source_kind", record.get("kind")) == "photo"}
    photos = [CaptureInput(path, "photo", {"parent_scene_id": parent.id})
              for path in paths if path.name in photo_names]
    if parent.source_video_path and Path(parent.source_video_path).is_file():
        return [CaptureInput(Path(parent.source_video_path), "video", {"parent_scene_id": parent.id})] + photos
    sequence = [path for path in paths if path.name not in photo_names]
    if not sequence and not photos:
        raise ValueError("Parent scene has no reusable video or saved frames")
    indexes = np.linspace(0, len(sequence) - 1, min(len(sequence), max(2, max_frames // 2)), dtype=int) if sequence else []
    return photos + [CaptureInput(sequence[int(i)], "photo", {"parent_scene_id": parent.id, "reused_frame": True}) for i in indexes]


def minimum_capture_frames(inputs: list[CaptureInput]) -> int:
    """Each still is mandatory; only legacy video frames can be resampled."""
    legacy = sum(bool(item.metadata.get("reused_frame")) for item in inputs)
    required = sum(6 if item.kind == "panorama" else 2 if item.kind == "video" else 1
                   for item in inputs if not item.metadata.get("reused_frame"))
    return required + min(2, legacy)


def effective_capture_frame_budget(inputs: list[CaptureInput], requested: int) -> int:
    if not 2 <= requested <= MAX_CAPTURE_FRAMES:
        raise ValueError(f"max_frames must be between 2 and {MAX_CAPTURE_FRAMES}")
    required = minimum_capture_frames(inputs)
    if required > MAX_CAPTURE_FRAMES:
        raise ValueError(f"以前の写真と今回の写真をすべて使うには{required}枚分が必要ですが、1つの部屋で処理できる上限は{MAX_CAPTURE_FRAMES}枚です。今回の写真を減らして再送信してください。撮影済みの写真は画面に残っています。")
    return max(requested, required)


def fit_reused_frames(inputs: list[CaptureInput], max_frames: int) -> list[CaptureInput]:
    legacy = [index for index, item in enumerate(inputs) if item.metadata.get("reused_frame")]
    mandatory = [item for item in inputs if not item.metadata.get("reused_frame")]
    available = max_frames - minimum_capture_frames(mandatory)
    if len(legacy) <= available:
        return inputs
    selected = {legacy[int(i)] for i in np.linspace(0, len(legacy) - 1, available, dtype=int)}
    return [item for index, item in enumerate(inputs) if index not in legacy or index in selected]


def allocate_frame_budget(inputs: list[CaptureInput], max_frames: int) -> list[int]:
    if not 2 <= max_frames <= 240:
        raise ValueError("max_frames must be between 2 and 240")
    quotas = [6 if item.kind == "panorama" else 2 if item.kind == "video" else 1 for item in inputs]
    if sum(quotas) > max_frames:
        raise ValueError("Frame budget is too small to cover every capture; reduce inputs or increase max_frames")
    videos = [i for i, item in enumerate(inputs) if item.kind == "video"]
    remaining = max_frames - sum(quotas)
    if videos:
        for i, index in enumerate(videos):
            quotas[index] += remaining // len(videos) + (i < remaining % len(videos))
    return quotas


def _rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError("Image exceeds the 40 megapixel capture limit")
        return ImageOps.exif_transpose(source).convert("RGB")


def panorama_faces(path: Path, output_dir: Path, face_size: int = 768) -> list[tuple[Path, str, np.ndarray]]:
    """Project explicit 2:1 equirectangular imagery to six 90-degree views.

    Coordinates: camera x right, y down, z forward; face rotations map a
    face-camera ray into the panorama's camera coordinate system.
    """
    image = _rgb_image(path)
    width, height = image.size
    if width < 512 or height < 256 or abs(width / height - 2.0) > 0.02:
        raise ValueError("Panorama must be a full 360-degree equirectangular 2:1 image (at least 512 x 256); ordinary wide photos are not supported")
    pixels = np.asarray(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:face_size, :face_size].astype(np.float32)
    rays = np.stack(((xx + 0.5 - face_size / 2) / (face_size / 2), (yy + 0.5 - face_size / 2) / (face_size / 2), np.ones_like(xx)), -1)
    rotations = {
        "front": np.eye(3),
        "right": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        "back": np.diag([-1, 1, -1]),
        "left": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        "up": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        "down": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    }
    results = []
    for name, rotation in rotations.items():
        world = rays @ rotation.T
        longitude = np.arctan2(world[..., 0], world[..., 2])
        latitude = np.arctan2(world[..., 1], np.linalg.norm(world[..., [0, 2]], axis=-1))
        map_x = ((longitude / (2 * np.pi) + 0.5) * width - 0.5).astype(np.float32)
        map_y = np.clip((latitude / np.pi + 0.5) * height - 0.5, 0, height - 1).astype(np.float32)
        face = cv2.remap(pixels, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        output = output_dir / f"{name}.jpg"
        Image.fromarray(face).save(output, quality=96)
        results.append((output, name, rotation.astype(np.float32)))
    return results


def prepare_capture(
    inputs: list[CaptureInput | dict | Path], scene_dir: Path, *, max_frames: int = 60,
    sample_fps: float = 4.0, quality_filter: bool = True,
    parent_scene_id: str | None = None, capture_metadata: dict | None = None,
) -> tuple[list[Path], list[dict], dict]:
    normalized = [normalize_input(item) for item in inputs]
    if not normalized:
        raise ValueError("At least one capture input is required")
    unique, duplicates, seen = [], [], {}
    for item in normalized:
        with item.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in seen:
            previous = unique[seen[digest]]
            if previous.kind != item.kind:
                raise ValueError("Identical capture content has conflicting photo/panorama types")
            duplicates.append({"original_name": item.metadata.get("original_name", item.path.name),
                               "kept_source_index": seen[digest], "sha256": digest})
            continue
        seen[digest] = len(unique)
        unique.append(item)
    normalized = unique
    if len(normalized) == 1 and normalized[0].kind != "video":
        raise ValueError("A single photo or panorama cannot establish multiple physical viewpoints; add an overlapping video or another capture position")
    max_frames = effective_capture_frame_budget(normalized, max_frames)
    normalized = fit_reused_frames(normalized, max_frames)
    quotas = allocate_frame_budget(normalized, max_frames)
    input_dir, frames_dir = scene_dir / "input", scene_dir / "frames"
    input_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    records, sources = [], []
    for source_index, (item, quota) in enumerate(zip(normalized, quotas)):
        source_id = f"capture-{source_index:03d}"
        stored = input_dir / f"{source_id}{item.path.suffix.lower()}"
        shutil.copy2(item.path, stored)
        sources.append({"id": source_id, "path": str(stored), "kind": item.kind, "original_name": item.path.name, "metadata": item.metadata})
        common = {"source_id": source_id, "capture_group": source_id, "source_path": str(stored), "kind": item.kind, "source_kind": item.kind}
        prepared = []
        if item.kind == "video":
            extracted = extract_video_frames_with_metadata(stored, input_dir / f"{source_id}-frames", sample_fps, quota, quality_filter)
            for frame in extracted:
                prepared.append((frame.path, {"source_frame_index": frame.source_frame_index, "timestamp_seconds": frame.timestamp_seconds, "sharpness": frame.sharpness, "motion_score": frame.motion_score}))
        elif item.kind == "panorama":
            for face_path, face_name, rotation in panorama_faces(stored, input_dir / f"{source_id}-faces"):
                prepared.append((face_path, {"panorama_group": source_id, "panorama_source_id": source_id, "projection": "perspective_90deg", "panorama_face": face_name, "face_camera_to_panorama": rotation.tolist(), "independent_baseline": False}))
        else:
            photo_path = input_dir / f"{source_id}-normalized.jpg"
            _rgb_image(stored).save(photo_path, quality=96)
            prepared.append((photo_path, {}))
        for frame_path, metadata in prepared:
            index = len(records)
            destination = frames_dir / f"{index:06d}.jpg"
            shutil.copy2(frame_path, destination)
            records.append({"index": index, "name": destination.name, **common, **metadata})
    if len(records) < 2:
        raise ValueError("At least two usable frames are required")
    limitations = ["All inputs must show the same room with overlapping views; joint inference does not guarantee alignment for disconnected captures."]
    if any(item.kind == "panorama" for item in normalized):
        limitations.append("Panorama faces share one physical camera center. Single-position panoramic imagery adds appearance and layout evidence, not translational depth observations. Panorama camera constraints remain approximate.")
    manifest = {"schema_version": 1, "parent_scene_id": parent_scene_id, "sources": sources, "options": capture_metadata or {}, "sample_fps": sample_fps, "max_frames": max_frames, "quality_filter": quality_filter, "frame_count": len(records), "selection": "full_duration_temporal_quality_bins", "duplicates_removed": duplicates, "limitations": limitations}
    (input_dir / "capture.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (frames_dir / "manifest.json").write_text(json.dumps({"schema_version": 2, "parent_scene_id": parent_scene_id, "capture_manifest": str(input_dir / "capture.json"), "frames": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return [frames_dir / record["name"] for record in records], records, manifest


def constrain_panorama_cameras(result: ReconstructionResult, records: list[dict]) -> None:
    """Fit one camera rig to inferred panorama views, eliminating fake baselines."""
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if record.get("panorama_group"):
            groups.setdefault(record["panorama_group"], []).append(index)
    if not groups:
        return
    height, width = result.depth.shape[1:3]
    details = []
    for group, indexes in groups.items():
        rotations, centers = [], []
        for index in indexes:
            extrinsic = result.extrinsics[index]
            face_rotation = np.asarray(records[index]["face_camera_to_panorama"])
            rotations.append(face_rotation @ extrinsic[:3, :3])
            centers.append(-extrinsic[:3, :3].T @ extrinsic[:3, 3])
        u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(u @ vt)
        base_rotation = u @ correction @ vt
        center = np.median(centers, axis=0)
        for index in indexes:
            face_rotation = np.asarray(records[index]["face_camera_to_panorama"])
            rotation = face_rotation.T @ base_rotation
            result.extrinsics[index, :3, :3] = rotation
            result.extrinsics[index, :3, 3] = -rotation @ center
            # Square perspective faces are never cropped by VGGT preprocessing.
            result.intrinsics[index] = np.array([[width / 2, 0, (width - 1) / 2], [0, height / 2, (height - 1) / 2], [0, 0, 1]])
        details.append({"source_id": group, "face_count": len(indexes), "physical_viewpoints": 1, "center_spread_before_constraint": float(np.max(np.linalg.norm(np.asarray(centers) - center, axis=1)))})
    result.metrics["panorama_camera_constraints"] = details


def mask_vggt_padding(result: ReconstructionResult, frames: list[Path], records: list[dict]) -> None:
    """Suppress batch padding introduced when videos and stills differ in aspect."""
    height, width = result.depth.shape[1:3]
    if width != 518:
        return
    masked = 0
    for index, path in enumerate(frames):
        with Image.open(path) as image:
            native_height = min(round(image.height * (518 / image.width) / 14) * 14, 518)
        if native_height > height:
            raise ValueError("VGGT output dimensions do not match capture preprocessing")
        top = (height - native_height) // 2
        bottom = top + native_height
        records[index]["valid_image_rect"] = [0, top, width, bottom]
        result.depth[index, :top] = 0
        result.depth[index, bottom:] = 0
        if result.confidence is not None:
            result.confidence[index, :top] = 0
            result.confidence[index, bottom:] = 0
        masked += (height - native_height) * width
    result.metrics["batch_padding_masked_pixels"] = masked
