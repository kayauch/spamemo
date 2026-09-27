from pathlib import Path

import numpy as np
import open3d as o3d

from .types import ReconstructionResult


def _depth3(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 4 and depth.shape[-1] == 1:
        return depth[..., 0]
    return depth


def unproject_to_world(result: ReconstructionResult) -> np.ndarray:
    depth = _depth3(result.depth)
    n, h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xx = np.broadcast_to(xx[None], (n, h, w))
    yy = np.broadcast_to(yy[None], (n, h, w))
    k = result.intrinsics
    fx, fy = k[:, 0, 0, None, None], k[:, 1, 1, None, None]
    cx, cy = k[:, 0, 2, None, None], k[:, 1, 2, None, None]
    camera = np.stack(((xx - cx) / fx * depth, (yy - cy) / fy * depth, depth), -1)
    r = result.extrinsics[:, :3, :3]
    t = result.extrinsics[:, :3, 3]
    return np.einsum(
        "nij,nhwj->nhwi",
        np.transpose(r, (0, 2, 1)),
        camera - t[:, None, None, :],
    )


def _colors_nhwc(images: np.ndarray | None) -> np.ndarray | None:
    if images is None:
        return None
    arr = images
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    elif arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    return np.clip(arr, 0.0, 1.0)


def _confidence3(confidence: np.ndarray | None) -> np.ndarray | None:
    if confidence is None:
        return None
    arr = np.asarray(confidence)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def robust_scene_diagonal(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    lower = np.quantile(points, 0.01, axis=0)
    upper = np.quantile(points, 0.99, axis=0)
    return float(np.linalg.norm(upper - lower))


def multi_frame_support_mask(
    points: np.ndarray,
    frame_ids: np.ndarray,
    voxel_size: float,
    min_frame_support: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    if min_frame_support < 1:
        raise ValueError("min_frame_support must be >= 1")

    origin = np.min(points, axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxel_frame_pairs = np.unique(
        np.column_stack([inverse, frame_ids.astype(np.int32)]),
        axis=0,
    )
    support_by_voxel = np.bincount(
        voxel_frame_pairs[:, 0],
        minlength=int(inverse.max()) + 1,
    )
    support_per_point = support_by_voxel[inverse].astype(np.int32)
    return support_per_point >= min_frame_support, support_per_point
def save_point_cloud(
    result: ReconstructionResult,
    path: str | Path,
    stride: int = 4,
    dynamic_filter: bool = True,
    support_voxel_size: float | None = None,
    min_frame_support: int | None = None,
    min_keep_ratio: float = 0.60,
) -> int:
    world = unproject_to_world(result)[:, ::stride, ::stride]
    depth = _depth3(result.depth)[:, ::stride, ::stride]
    n, h, w = depth.shape

    points = world.reshape(-1, 3)
    sampled_depth = depth.reshape(-1)
    frame_ids = np.broadcast_to(
        np.arange(n, dtype=np.int32)[:, None, None],
        (n, h, w),
    ).reshape(-1)

    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(sampled_depth)
        & (sampled_depth > 0)
    )
    points = points[valid]
    frame_ids = frame_ids[valid]

    colors = _colors_nhwc(result.images)
    sampled_colors = None
    if colors is not None:
        sampled_colors = colors[:, ::stride, ::stride].reshape(-1, 3)[valid]

    confidence = _confidence3(result.confidence)
    sampled_confidence = None
    if confidence is not None:
        sampled_confidence = confidence[:, ::stride, ::stride].reshape(-1)[valid]

    input_count = int(len(points))
    scene_diagonal = robust_scene_diagonal(points)
    filter_metrics: dict[str, float | int | str | bool] = {
        "mode": "disabled",
        "input_points": input_count,
        "output_points": input_count,
        "removed_points": 0,
        "keep_ratio": 1.0,
        "scene_diagonal_robust": scene_diagonal,
        "stride": int(stride),
        "frame_count": int(n),
    }

    if sampled_confidence is not None and sampled_confidence.size:
        finite_conf = sampled_confidence[np.isfinite(sampled_confidence)]
        if finite_conf.size:
            q05, q50, q95 = np.quantile(finite_conf, [0.05, 0.50, 0.95])
            filter_metrics.update({
                "confidence_q05": float(q05),
                "confidence_q50": float(q50),
                "confidence_q95": float(q95),
            })

    if dynamic_filter and n >= 3 and input_count:
        voxel_size = (
            float(support_voxel_size)
            if support_voxel_size is not None
            else float(np.clip(scene_diagonal * 0.015, 0.02, 0.06))
        )
        requested_support = (
            int(min_frame_support)
            if min_frame_support is not None
            else (3 if n >= 12 else 2)
        )
        requested_support = max(2, min(requested_support, n))

        keep, support = multi_frame_support_mask(
            points,
            frame_ids,
            voxel_size=voxel_size,
            min_frame_support=requested_support,
        )
        applied_support = requested_support
        keep_ratio = float(keep.mean()) if keep.size else 1.0

        if keep_ratio < min_keep_ratio and requested_support > 2:
            keep, support = multi_frame_support_mask(
                points,
                frame_ids,
                voxel_size=voxel_size,
                min_frame_support=2,
            )
            applied_support = 2
            keep_ratio = float(keep.mean()) if keep.size else 1.0

        if keep_ratio >= min_keep_ratio:
            points = points[keep]
            frame_ids = frame_ids[keep]
            if sampled_colors is not None:
                sampled_colors = sampled_colors[keep]
            if sampled_confidence is not None:
                sampled_confidence = sampled_confidence[keep]
            filter_metrics.update({
                "mode": "multi_frame_voxel",
                "voxel_size": voxel_size,
                "min_frame_support": int(applied_support),
                "requested_min_frame_support": int(requested_support),
                "median_frame_support": float(np.median(support)),
                "p90_frame_support": float(np.quantile(support, 0.90)),
                "output_points": int(len(points)),
                "removed_points": int(input_count - len(points)),
                "keep_ratio": float(len(points) / max(input_count, 1)),
            })
        else:
            filter_metrics.update({
                "mode": "safety_fallback_unfiltered",
                "voxel_size": voxel_size,
                "requested_min_frame_support": int(requested_support),
                "candidate_keep_ratio": keep_ratio,
            })
    elif dynamic_filter:
        filter_metrics["mode"] = "disabled_short_sequence"

    result.metrics["point_cloud_filter"] = filter_metrics

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if sampled_colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(
            sampled_colors.astype(np.float64)
        )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output), cloud, write_ascii=False)
    return len(points)
