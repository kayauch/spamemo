from pathlib import Path

import cv2
import numpy as np

from spatial_recon.benchmark_metrics import (
    camera_centers,
    heldout_relocalization_consistency,
    matched_world_consistency,
    reconstruction_health,
)
from spatial_recon.types import ReconstructionResult


def synthetic_result(n: int = 5, h: int = 96, w: int = 128) -> ReconstructionResult:
    yy, xx = np.meshgrid(
        np.linspace(0, 1, h, dtype=np.float32),
        np.linspace(0, 1, w, dtype=np.float32),
        indexing="ij",
    )
    base_depth = 2.0 + 0.4 * xx + 0.2 * yy
    depth = np.repeat(base_depth[None], n, axis=0)

    intrinsics = np.repeat(
        np.array([[
            [90.0, 0.0, w / 2],
            [0.0, 90.0, h / 2],
            [0.0, 0.0, 1.0],
        ]], dtype=np.float32),
        n,
        axis=0,
    )
    extrinsics = np.repeat(
        np.array([[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]], dtype=np.float32),
        n,
        axis=0,
    )
    return ReconstructionResult(
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )


def write_textured_frames(tmp_path: Path, n: int = 5) -> list[Path]:
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    for _ in range(20):
        x1, y1 = rng.integers([0, 0], [100, 70])
        x2, y2 = x1 + int(rng.integers(8, 28)), y1 + int(rng.integers(8, 24))
        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            tuple(int(x) for x in rng.integers(0, 255, 3)),
            2,
        )
    paths = []
    for index in range(n):
        path = tmp_path / f"{index:06d}.jpg"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def test_camera_centers_and_health() -> None:
    result = synthetic_result(n=3)
    result.extrinsics[1, 0, 3] = -1.0
    result.extrinsics[2, 0, 3] = -2.0

    centers = camera_centers(result.extrinsics)
    assert np.allclose(centers[:, 0], [0.0, 1.0, 2.0])

    health = reconstruction_health(result)
    assert health["depth_valid_fraction"] == 1.0
    assert health["invalid_pose_frames"] == 0
    assert abs(health["camera_step_median_model_units"] - 1.0) < 1e-6


def test_multiview_consistency_metrics_on_identical_views(tmp_path: Path) -> None:
    frames = write_textured_frames(tmp_path)
    result = synthetic_result(n=len(frames))

    consistency = matched_world_consistency(
        result,
        frames,
        max_pairs=3,
    )
    assert consistency["matched_world_samples"] > 20
    assert consistency["matched_world_median_normalized"] < 1e-5

    relocalization = heldout_relocalization_consistency(
        result,
        frames,
        holdout_count=3,
        max_reference_frames=3,
        min_inliers=8,
    )
    assert relocalization["relocalization_success_rate"] > 0.5
    assert relocalization["relocalization_position_error_median_normalized"] < 0.05
