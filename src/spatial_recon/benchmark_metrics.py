from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .types import ReconstructionResult


def _depth3(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsics, dtype=np.float64)
    r = ext[:, :3, :3]
    t = ext[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(r, (0, 2, 1)), t)


def reconstruction_health(result: ReconstructionResult) -> dict[str, float | int]:
    depth = _depth3(result.depth).astype(np.float64)
    positive = np.isfinite(depth) & (depth > 0)
    depth_valid_fraction = float(positive.mean()) if depth.size else 0.0
    valid_depth = depth[positive]
    median_depth = float(np.median(valid_depth)) if valid_depth.size else math.nan

    ext = np.asarray(result.extrinsics)
    intr = np.asarray(result.intrinsics)
    pose_valid = np.isfinite(ext).all(axis=(1, 2))
    intrinsics_valid = np.isfinite(intr).all(axis=(1, 2))
    centers = camera_centers(ext[pose_valid]) if pose_valid.any() else np.empty((0, 3))

    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.array([])
    return {
        "frame_count": int(depth.shape[0]) if depth.ndim >= 3 else 0,
        "depth_valid_fraction": depth_valid_fraction,
        "median_depth_model_units": median_depth,
        "invalid_pose_frames": int((~pose_valid).sum()),
        "invalid_intrinsics_frames": int((~intrinsics_valid).sum()),
        "camera_step_median_model_units": float(np.median(steps)) if steps.size else math.nan,
        "camera_step_p95_model_units": float(np.percentile(steps, 95)) if steps.size else math.nan,
    }


def _pixel_to_world(
    u: float,
    v: float,
    image_size: tuple[int, int],
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray | None:
    image_w, image_h = image_size
    depth_h, depth_w = depth.shape[-2:]
    du = u * depth_w / max(image_w, 1)
    dv = v * depth_h / max(image_h, 1)
    x = int(round(du))
    y = int(round(dv))
    if x < 0 or x >= depth_w or y < 0 or y >= depth_h:
        return None

    y1, y2 = max(0, y - 2), min(depth_h, y + 3)
    x1, x2 = max(0, x - 2), min(depth_w, x + 3)
    crop = depth[y1:y2, x1:x2]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    if not valid.size:
        return None
    z = float(np.median(valid))

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        return None

    camera = np.array(
        [(du - cx) / fx * z, (dv - cy) / fy * z, z],
        dtype=np.float64,
    )
    r = np.asarray(extrinsic[:3, :3], dtype=np.float64)
    t = np.asarray(extrinsic[:3, 3], dtype=np.float64)
    return r.T @ (camera - t)


def matched_world_consistency(
    result: ReconstructionResult,
    image_paths: list[Path],
    max_pairs: int = 8,
    ratio_test: float = 0.75,
    max_matches_per_pair: int = 250,
) -> dict[str, float | int]:
    depth = _depth3(result.depth)
    n = min(len(image_paths), len(depth), len(result.extrinsics), len(result.intrinsics))
    if n < 2:
        return {
            "matched_world_samples": 0,
            "matched_world_median_model_units": math.nan,
            "matched_world_p90_model_units": math.nan,
            "matched_world_median_normalized": math.nan,
            "matched_world_p90_normalized": math.nan,
        }

    pair_indices = np.linspace(0, n - 2, min(max_pairs, n - 1), dtype=int)
    sift = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    cache: dict[int, tuple[list, np.ndarray | None, tuple[int, int]]] = {}

    def features(index: int):
        if index in cache:
            return cache[index]
        gray = cv2.imread(str(image_paths[index]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            value = ([], None, (1, 1))
        else:
            kp, desc = sift.detectAndCompute(gray, None)
            value = (kp or [], desc, (gray.shape[1], gray.shape[0]))
        cache[index] = value
        return value

    distances: list[float] = []
    for i in pair_indices.tolist():
        j = i + 1
        kp1, d1, size1 = features(i)
        kp2, d2, size2 = features(j)
        if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
            continue
        pairs = matcher.knnMatch(d1, d2, k=2)
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < ratio_test * second.distance
        ]
        good.sort(key=lambda m: m.distance)
        for match in good[:max_matches_per_pair]:
            p1 = kp1[match.queryIdx].pt
            p2 = kp2[match.trainIdx].pt
            w1 = _pixel_to_world(
                p1[0], p1[1], size1, depth[i], result.intrinsics[i], result.extrinsics[i]
            )
            w2 = _pixel_to_world(
                p2[0], p2[1], size2, depth[j], result.intrinsics[j], result.extrinsics[j]
            )
            if w1 is None or w2 is None:
                continue
            distance = float(np.linalg.norm(w1 - w2))
            if math.isfinite(distance):
                distances.append(distance)

    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    scale = float(np.median(valid_depth)) if valid_depth.size else math.nan
    arr = np.asarray(distances, dtype=np.float64)
    if not arr.size:
        return {
            "matched_world_samples": 0,
            "matched_world_median_model_units": math.nan,
            "matched_world_p90_model_units": math.nan,
            "matched_world_median_normalized": math.nan,
            "matched_world_p90_normalized": math.nan,
        }

    median = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    return {
        "matched_world_samples": int(arr.size),
        "matched_world_median_model_units": median,
        "matched_world_p90_model_units": p90,
        "matched_world_median_normalized": median / scale if scale > 0 else math.nan,
        "matched_world_p90_normalized": p90 / scale if scale > 0 else math.nan,
    }



def _scale_intrinsics_to_image(
    intrinsic: np.ndarray,
    depth_shape: tuple[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    depth_h, depth_w = depth_shape
    image_w, image_h = image_size
    sx = image_w / max(depth_w, 1)
    sy = image_h / max(depth_h, 1)
    k = np.asarray(intrinsic, dtype=np.float64).copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def heldout_relocalization_consistency(
    result: ReconstructionResult,
    image_paths: list[Path],
    holdout_count: int = 5,
    max_reference_frames: int = 12,
    ratio_test: float = 0.75,
    min_inliers: int = 10,
) -> dict[str, float | int]:
    depth = _depth3(result.depth)
    n = min(len(image_paths), len(depth), len(result.extrinsics), len(result.intrinsics))
    if n < 3:
        return {
            "relocalization_attempts": 0,
            "relocalization_successes": 0,
            "relocalization_success_rate": 0.0,
            "relocalization_position_error_median_model_units": math.nan,
            "relocalization_position_error_p90_model_units": math.nan,
            "relocalization_position_error_median_normalized": math.nan,
            "relocalization_reprojection_error_median_px": math.nan,
            "relocalization_inliers_median": math.nan,
        }

    sift = cv2.SIFT_create(nfeatures=3000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    cache: dict[int, tuple[list, np.ndarray | None, tuple[int, int]]] = {}

    def features(index: int):
        if index in cache:
            return cache[index]
        gray = cv2.imread(str(image_paths[index]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            value = ([], None, (1, 1))
        else:
            kp, desc = sift.detectAndCompute(gray, None)
            value = (kp or [], desc, (gray.shape[1], gray.shape[0]))
        cache[index] = value
        return value

    holdouts = np.linspace(1, n - 2, min(holdout_count, n - 2), dtype=int)
    target_centers = camera_centers(np.asarray(result.extrinsics))
    position_errors: list[float] = []
    reprojection_errors: list[float] = []
    inlier_counts: list[int] = []
    successes = 0

    for query_index in holdouts.tolist():
        query_kp, query_desc, query_size = features(query_index)
        if query_desc is None or len(query_kp) < min_inliers:
            continue

        candidates = [idx for idx in range(n) if idx != query_index]
        candidates.sort(key=lambda idx: abs(idx - query_index))
        candidates = candidates[:max_reference_frames]

        best = None
        camera_matrix = _scale_intrinsics_to_image(
            result.intrinsics[query_index],
            depth[query_index].shape[-2:],
            query_size,
        )

        for ref_index in candidates:
            ref_kp, ref_desc, ref_size = features(ref_index)
            if ref_desc is None or len(ref_kp) < 2:
                continue

            pairs = matcher.knnMatch(ref_desc, query_desc, k=2)
            good = [
                first
                for pair in pairs
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < ratio_test * second.distance
            ]
            if len(good) < min_inliers:
                continue

            world_points: list[np.ndarray] = []
            image_points: list[tuple[float, float]] = []
            for match in good:
                ref_pt = ref_kp[match.queryIdx].pt
                query_pt = query_kp[match.trainIdx].pt
                world = _pixel_to_world(
                    ref_pt[0],
                    ref_pt[1],
                    ref_size,
                    depth[ref_index],
                    result.intrinsics[ref_index],
                    result.extrinsics[ref_index],
                )
                if world is None or not np.isfinite(world).all():
                    continue
                world_points.append(world)
                image_points.append(query_pt)

            if len(world_points) < min_inliers:
                continue

            object_points = np.asarray(world_points, dtype=np.float64)
            image_points_arr = np.asarray(image_points, dtype=np.float64)
            ok, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
                object_points,
                image_points_arr,
                camera_matrix,
                None,
                flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=200,
                reprojectionError=6.0,
                confidence=0.999,
            )
            if not ok or inlier_idx is None or len(inlier_idx) < min_inliers:
                continue

            inliers = inlier_idx.reshape(-1)
            cv2.solvePnP(
                object_points[inliers],
                image_points_arr[inliers],
                camera_matrix,
                None,
                rvec,
                tvec,
                True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            projected, _ = cv2.projectPoints(
                object_points[inliers],
                rvec,
                tvec,
                camera_matrix,
                None,
            )
            reproj = float(
                np.mean(
                    np.linalg.norm(
                        projected.reshape(-1, 2) - image_points_arr[inliers],
                        axis=1,
                    )
                )
            )
            rotation, _ = cv2.Rodrigues(rvec)
            center = -rotation.T @ tvec.reshape(3)
            candidate = {
                "center": center,
                "inliers": int(len(inliers)),
                "reprojection_error": reproj,
            }
            if best is None or (
                candidate["inliers"],
                -candidate["reprojection_error"],
            ) > (
                best["inliers"],
                -best["reprojection_error"],
            ):
                best = candidate

        if best is None:
            continue

        successes += 1
        position_errors.append(
            float(np.linalg.norm(best["center"] - target_centers[query_index]))
        )
        reprojection_errors.append(float(best["reprojection_error"]))
        inlier_counts.append(int(best["inliers"]))

    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    scale = float(np.median(valid_depth)) if valid_depth.size else math.nan
    pos = np.asarray(position_errors, dtype=np.float64)
    reproj = np.asarray(reprojection_errors, dtype=np.float64)
    inliers = np.asarray(inlier_counts, dtype=np.float64)
    attempts = int(len(holdouts))

    return {
        "relocalization_attempts": attempts,
        "relocalization_successes": int(successes),
        "relocalization_success_rate": successes / attempts if attempts else 0.0,
        "relocalization_position_error_median_model_units": float(np.median(pos)) if pos.size else math.nan,
        "relocalization_position_error_p90_model_units": float(np.percentile(pos, 90)) if pos.size else math.nan,
        "relocalization_position_error_median_normalized": (
            float(np.median(pos)) / scale if pos.size and scale > 0 else math.nan
        ),
        "relocalization_reprojection_error_median_px": float(np.median(reproj)) if reproj.size else math.nan,
        "relocalization_inliers_median": float(np.median(inliers)) if inliers.size else math.nan,
    }
