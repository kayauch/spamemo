from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import BoundingBox3D, Vec3


def load_reconstruction(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def pixel_to_world(
    u: float,
    v: float,
    depth: float,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> Vec3:
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera = np.array([
        (u - cx) / fx * depth,
        (v - cy) / fy * depth,
        depth,
    ], dtype=np.float64)

    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    world = rotation.T @ (camera - translation)
    return Vec3(x=float(world[0]), y=float(world[1]), z=float(world[2]))


def bbox_center_to_world(
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> Vec3:
    if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    x1, y1, x2, y2 = bbox_xyxy
    h, w = depth_map.shape[-2:]
    x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
    y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
    crop = depth_map[y1:y2 + 1, x1:x2 + 1]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    if valid.size == 0:
        raise ValueError("No valid depth inside bbox")
    depth = float(np.median(valid))
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    return pixel_to_world(u, v, depth, intrinsic, extrinsic)


def project_bbox_from_npz(
    reconstruction_npz: str | Path,
    frame_index: int,
    bbox_xyxy: tuple[int, int, int, int],
    source_size: tuple[int, int] | None = None,
) -> Vec3:
    data = load_reconstruction(reconstruction_npz)
    depth = data["depth"]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    bbox = bbox_xyxy
    if source_size is not None:
        source_w, source_h = source_size
        target_h, target_w = depth[frame_index].shape[-2:]
        sx = target_w / source_w
        sy = target_h / source_h
        x1, y1, x2, y2 = bbox_xyxy
        bbox = (
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round(x2 * sx)),
            int(round(y2 * sy)),
        )

    return bbox_center_to_world(
        depth[frame_index],
        data["intrinsics"][frame_index],
        data["extrinsics"][frame_index],
        bbox,
    )


def _scaled_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    source_size: tuple[int, int] | None,
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    bbox = bbox_xyxy
    if source_size is not None:
        source_w, source_h = source_size
        target_h, target_w = target_shape
        sx = target_w / max(float(source_w), 1.0)
        sy = target_h / max(float(source_h), 1.0)
        x1, y1, x2, y2 = bbox_xyxy
        bbox = (
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round(x2 * sx)),
            int(round(y2 * sy)),
        )
    return bbox


def bbox_geometry_to_world(
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    max_samples: int = 256,
    trim_quantile: float = 0.10,
) -> tuple[Vec3, BoundingBox3D]:
    """Project a 2D bbox into a robust 3D center and axis-aligned world bbox."""
    if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    if depth_map.ndim != 2:
        raise ValueError("depth_map must be HxW")

    h, w = depth_map.shape
    x1, y1, x2, y2 = bbox_xyxy
    x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
    y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
    if x2 < x1 or y2 < y1:
        raise ValueError("Invalid bbox")

    crop = depth_map[y1:y2 + 1, x1:x2 + 1]
    yy, xx = np.indices(crop.shape)
    valid_mask = np.isfinite(crop) & (crop > 0)
    if not np.any(valid_mask):
        raise ValueError("No valid depth inside bbox")

    depths = crop[valid_mask].astype(np.float64)
    us = (xx[valid_mask] + x1).astype(np.float64)
    vs = (yy[valid_mask] + y1).astype(np.float64)

    # Suppress foreground/background contamination near bbox borders.
    if len(depths) >= 10 and 0.0 < trim_quantile < 0.45:
        low, high = np.quantile(
            depths,
            [trim_quantile, 1.0 - trim_quantile],
        )
        keep = (depths >= low) & (depths <= high)
        if int(np.count_nonzero(keep)) >= 4:
            depths = depths[keep]
            us = us[keep]
            vs = vs[keep]

    if len(depths) > max_samples:
        indices = np.linspace(0, len(depths) - 1, max_samples).astype(int)
        depths = depths[indices]
        us = us[indices]
        vs = vs[indices]

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        raise ValueError("Invalid camera intrinsics")

    camera = np.column_stack([
        (us - cx) / fx * depths,
        (vs - cy) / fy * depths,
        depths,
    ])
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float64)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float64)
    world = (camera - translation[None, :]) @ rotation

    if len(world) >= 10:
        lower = np.quantile(world, trim_quantile, axis=0)
        upper = np.quantile(world, 1.0 - trim_quantile, axis=0)
    else:
        lower = world.min(axis=0)
        upper = world.max(axis=0)

    center = (lower + upper) * 0.5
    size = np.maximum(upper - lower, 1e-6)
    center_vec = Vec3(
        x=float(center[0]),
        y=float(center[1]),
        z=float(center[2]),
    )
    bbox3d = BoundingBox3D(
        center=center_vec,
        size=Vec3(
            x=float(size[0]),
            y=float(size[1]),
            z=float(size[2]),
        ),
    )
    return center_vec, bbox3d


def project_bbox_geometry(
    reconstruction: dict[str, np.ndarray],
    frame_index: int,
    bbox_xyxy: tuple[int, int, int, int],
    source_size: tuple[int, int] | None = None,
) -> tuple[Vec3, BoundingBox3D]:
    depth = reconstruction["depth"]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_map = depth[frame_index]
    bbox = _scaled_bbox(
        bbox_xyxy,
        source_size=source_size,
        target_shape=depth_map.shape[-2:],
    )
    return bbox_geometry_to_world(
        depth_map,
        reconstruction["intrinsics"][frame_index],
        reconstruction["extrinsics"][frame_index],
        bbox,
    )


def project_bbox_geometry_from_npz(
    reconstruction_npz: str | Path,
    frame_index: int,
    bbox_xyxy: tuple[int, int, int, int],
    source_size: tuple[int, int] | None = None,
) -> tuple[Vec3, BoundingBox3D]:
    return project_bbox_geometry(
        load_reconstruction(reconstruction_npz),
        frame_index,
        bbox_xyxy,
        source_size=source_size,
    )
