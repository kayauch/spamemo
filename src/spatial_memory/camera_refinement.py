"""Conservative image/depth pose refinement, with an independent acceptance gate.

This is fixed-depth PnP, not COLMAP or bundle adjustment. Reference cameras,
depth maps, calibration, scale and gauge remain fixed. It can only accept small
pose corrections supported by withheld image matches and cross-view depth.
"""
from __future__ import annotations

import cv2
import numpy as np

from spatial_recon.types import ReconstructionResult


def _project(points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    pixels = camera @ intrinsic.T
    return pixels[:, :2] / np.maximum(pixels[:, 2:3], 1e-9), camera[:, 2]


def refine_pose_from_correspondences(
    world_points: np.ndarray, image_points: np.ndarray, observed_depth: np.ndarray,
    intrinsic: np.ndarray, initial_extrinsic: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit on 80% of matches; accept only on the other 20%, never on train error."""
    original = np.asarray(initial_extrinsic).copy()
    report = {"accepted": False, "reason": "insufficient_matches", "matches": len(world_points)}
    points = np.asarray(world_points, dtype=np.float64)
    pixels = np.asarray(image_points, dtype=np.float64)
    depths = np.asarray(observed_depth, dtype=np.float64)
    if points.shape != (len(points), 3) or pixels.shape != (len(points), 2) or depths.shape != (len(points),):
        raise ValueError("Pose correspondences must have shapes N x 3, N x 2 and N")
    valid = np.isfinite(points).all(axis=1) & np.isfinite(pixels).all(axis=1) & np.isfinite(depths) & (depths > 0)
    points, pixels, depths = points[valid], pixels[valid], depths[valid]
    report["matches"] = len(points)
    if len(points) < 40:
        return original, report
    heldout = np.arange(len(points)) % 5 == 0
    train = ~heldout
    before_uv, before_z = _project(points[heldout], original, intrinsic)
    before_error = float(np.median(np.linalg.norm(before_uv - pixels[heldout], axis=1)))
    before_depth = float(np.median(np.abs(before_z - depths[heldout]) / depths[heldout]))
    report.update({"heldout_matches": int(heldout.sum()), "before_reprojection_px": before_error, "before_relative_depth_error": before_depth})
    if before_error < 0.5:
        report["reason"] = "already_aligned"
        return original, report
    try:
        rvec, _ = cv2.Rodrigues(original[:, :3].astype(np.float64))
        tvec = original[:, 3].astype(np.float64).copy().reshape(3, 1)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            points[train], pixels[train], np.asarray(intrinsic, dtype=np.float64), None,
            rvec=rvec.copy(), tvec=tvec, useExtrinsicGuess=True,
            iterationsCount=150, reprojectionError=3.0, confidence=0.995,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < 20:
            report["reason"] = "insufficient_pnp_inliers"
            return original, report
        indexes = inliers.reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(points[train][indexes], pixels[train][indexes], np.asarray(intrinsic, dtype=np.float64), None, rvec, tvec)
        rotation, _ = cv2.Rodrigues(rvec)
    except cv2.error:
        report["reason"] = "pnp_failed"
        return original, report
    proposed = np.column_stack([rotation, tvec.reshape(3)])
    if not np.isfinite(proposed).all() or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        report["reason"] = "invalid_rigid_pose"
        return original, report
    old_center = -original[:, :3].T @ original[:, 3]
    new_center = -rotation.T @ proposed[:, 3]
    scale = max(float(np.median(np.linalg.norm(points - old_center, axis=1))), 1e-6)
    translation_fraction = float(np.linalg.norm(new_center - old_center) / scale)
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation @ original[:, :3].T) - 1) / 2, -1, 1))))
    report.update({"translation_fraction_of_depth": translation_fraction, "rotation_degrees": angle, "training_inliers": int(len(inliers))})
    if translation_fraction > 0.08 or angle > 5:
        report["reason"] = "correction_exceeds_local_bounds"
        return original, report
    after_uv, after_z = _project(points[heldout], proposed, intrinsic)
    after_error = float(np.median(np.linalg.norm(after_uv - pixels[heldout], axis=1)))
    after_depth = float(np.median(np.abs(after_z - depths[heldout]) / depths[heldout]))
    report.update({"after_reprojection_px": after_error, "after_relative_depth_error": after_depth})
    if np.mean(after_z > 0) < 0.98:
        report["reason"] = "points_behind_camera"
    elif after_error > 3.0 or after_error > before_error * 0.85:
        report["reason"] = "heldout_reprojection_did_not_improve"
    elif after_depth > 0.12 or after_depth > before_depth + 0.005:
        report["reason"] = "cross_view_depth_disagreement"
    else:
        report.update({"accepted": True, "reason": "heldout_reprojection_and_depth_passed"})
        return proposed.astype(original.dtype), report
    return original, report


def _processed_images(result: ReconstructionResult) -> np.ndarray | None:
    if result.images is None:
        return None
    images = np.asarray(result.images)
    if images.ndim != 4:
        return None
    if images.shape[1] == 3 and images.shape[-1] != 3:
        images = images.transpose(0, 2, 3, 1)
    if images.shape[:3] != result.depth.shape[:3] or images.shape[-1] != 3:
        return None
    if np.issubdtype(images.dtype, np.floating):
        if float(np.nanmin(images)) < 0:
            images = (images + 1) / 2
        if float(np.nanmax(images)) <= 1.5:
            images = images * 255
    return np.clip(images, 0, 255).astype(np.uint8)


def refine_capture_cameras(result: ReconstructionResult, records: list[dict], *, max_features: int = 1200) -> dict:
    report = {"method": "fixed_depth_orb_pnp_with_heldout_validation", "accepted_count": 0, "frames": [], "depth_modified": False, "scale_modified": False, "limitations": "Local pose correction uses predicted depth as a fixed reference; it does not guarantee improved real-world dimensions and is not bundle adjustment."}
    if result.metrics.get("model") == "calibrated-rgbd":
        report["status"] = "skipped_calibrated_sensor"
        return report
    images = _processed_images(result)
    if images is None:
        report["status"] = "skipped_missing_processed_images"
        return report
    count = result.frame_count
    if len(records) != count:
        raise ValueError("Camera refinement records must match reconstruction frames")
    usable = [i for i, record in enumerate(records) if not record.get("panorama_group")]
    if len(usable) < 2:
        report["status"] = "skipped_insufficient_independent_views"
        return report
    anchor_count = min(3, max(1, len(usable) // 10))
    anchors = [usable[int(i)] for i in np.linspace(0, len(usable) - 1, anchor_count, endpoint=False)]
    report["fixed_reference_frames"] = anchors
    original_poses = result.extrinsics.copy()
    depth = np.asarray(result.depth)
    if depth.ndim == 4:
        depth = depth[..., 0]
    detector = cv2.ORB_create(nfeatures=max(100, min(max_features, 3000)), fastThreshold=12)
    features = [detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None) for image in images]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for target in range(count):
        if target not in usable:
            report["frames"].append({"index": target, "accepted": False, "reason": "panorama_rig_preserved"})
            continue
        if target in anchors:
            report["frames"].append({"index": target, "accepted": False, "reason": "fixed_reference"})
            continue
        target_keypoints, target_descriptors = features[target]
        if target_descriptors is None:
            report["frames"].append({"index": target, "accepted": False, "reason": "insufficient_texture"})
            continue
        correspondences = {}
        # Local references supply overlap in long room videos. All references
        # use original poses, so accepted updates cannot propagate drift.
        references = sorted(set(anchors + [j for j in (target-6, target-3, target-1, target+1, target+3, target+6) if j in usable and j != target]))
        for anchor in references:
            keypoints, descriptors = features[anchor]
            if descriptors is None or len(target_descriptors) < 2:
                continue
            if len(descriptors) < 2:
                continue
            reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(target_descriptors, descriptors, k=2)
                       if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance}
            for pair in matcher.knnMatch(descriptors, target_descriptors, k=2):
                if len(pair) != 2:
                    continue
                match, runner_up = pair
                if reverse.get(match.trainIdx) != match.queryIdx or match.distance >= 0.72 * runner_up.distance or match.distance > 64:
                    continue
                pixel = np.asarray(keypoints[match.queryIdx].pt)
                target_pixel = np.asarray(target_keypoints[match.trainIdx].pt)
                x, y = np.rint(pixel).astype(int)
                tx, ty = np.rint(target_pixel).astype(int)
                z, tz = float(depth[anchor, y, x]), float(depth[target, ty, tx])
                if not np.isfinite([z, tz]).all() or z <= 0 or tz <= 0:
                    continue
                ray = np.linalg.solve(result.intrinsics[anchor], np.r_[pixel, 1.0])
                camera_point = ray * z
                pose = original_poses[anchor]
                world_point = pose[:, :3].T @ (camera_point - pose[:, 3])
                if match.trainIdx not in correspondences or match.distance < correspondences[match.trainIdx][0]:
                    correspondences[match.trainIdx] = (match.distance, world_point, target_pixel, tz)
        rows = [correspondences[index] for index in sorted(correspondences)]
        if len(rows) < 40:
            report["frames"].append({"index": target, "accepted": False, "reason": "insufficient_consistent_matches", "matches": len(rows)})
            continue
        pose, frame_report = refine_pose_from_correspondences(np.asarray([row[1] for row in rows]), np.asarray([row[2] for row in rows]), np.asarray([row[3] for row in rows]), result.intrinsics[target], original_poses[target])
        frame_report["index"] = target
        report["frames"].append(frame_report)
        if frame_report["accepted"]:
            result.extrinsics[target] = pose
            report["accepted_count"] += 1
    report["status"] = "completed"
    return report
