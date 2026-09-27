from pathlib import Path

import numpy as np

from spatial_recon.geometry import (
    multi_frame_support_mask,
    save_point_cloud,
)
from spatial_recon.types import ReconstructionResult


def test_multi_frame_support_removes_transient_voxel() -> None:
    points = np.array([
        [0.00, 0.00, 0.00],
        [0.01, 0.00, 0.00],
        [0.02, 0.00, 0.00],
        [1.00, 1.00, 1.00],
    ], dtype=np.float32)
    frame_ids = np.array([0, 1, 2, 0], dtype=np.int32)

    keep, support = multi_frame_support_mask(
        points,
        frame_ids,
        voxel_size=0.05,
        min_frame_support=2,
    )

    assert keep.tolist() == [True, True, True, False]
    assert support.tolist() == [3, 3, 3, 1]


def test_save_point_cloud_filters_single_frame_depth_artifact(
    tmp_path: Path,
) -> None:
    depth = np.full((3, 2, 2), 2.0, dtype=np.float32)
    depth[0, 0, 0] = 8.0
    intrinsics = np.repeat(
        np.array([[
            [2.0, 0.0, 0.5],
            [0.0, 2.0, 0.5],
            [0.0, 0.0, 1.0],
        ]], dtype=np.float32),
        3,
        axis=0,
    )
    extrinsics = np.repeat(
        np.array([[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]], dtype=np.float32),
        3,
        axis=0,
    )
    result = ReconstructionResult(
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )

    output = tmp_path / "filtered.ply"
    count = save_point_cloud(
        result,
        output,
        stride=1,
        dynamic_filter=True,
        support_voxel_size=0.20,
        min_frame_support=2,
    )

    assert output.exists()
    assert count == 11
    metrics = result.metrics["point_cloud_filter"]
    assert metrics["mode"] == "multi_frame_voxel"
    assert metrics["input_points"] == 12
    assert metrics["output_points"] == 11
    assert metrics["removed_points"] == 1
