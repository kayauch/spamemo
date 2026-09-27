from pathlib import Path

import numpy as np

from spatial_memory.projection import project_bbox_from_npz, project_bbox_geometry_from_npz


def test_bbox_center_projects_to_world(tmp_path: Path) -> None:
    depth = np.full((1, 5, 5), 2.0, dtype=np.float32)
    intrinsics = np.array([[
        [2.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 1.0],
    ]], dtype=np.float32)
    extrinsics = np.array([[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]], dtype=np.float32)
    path = tmp_path / "reconstruction.npz"
    np.savez(path, depth=depth, intrinsics=intrinsics, extrinsics=extrinsics)

    point = project_bbox_from_npz(path, 0, (1, 1, 3, 3))
    assert abs(point.x) < 1e-6
    assert abs(point.y) < 1e-6
    assert abs(point.z - 2.0) < 1e-6


def test_source_image_bbox_is_scaled_to_depth_resolution(tmp_path: Path) -> None:
    depth = np.full((1, 5, 5), 2.0, dtype=np.float32)
    intrinsics = np.array([[
        [2.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 1.0],
    ]], dtype=np.float32)
    extrinsics = np.array([[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]], dtype=np.float32)
    path = tmp_path / "scaled.npz"
    np.savez(path, depth=depth, intrinsics=intrinsics, extrinsics=extrinsics)

    point = project_bbox_from_npz(
        path,
        0,
        (2, 2, 6, 6),
        source_size=(10, 10),
    )
    assert abs(point.x) < 1e-6
    assert abs(point.y) < 1e-6
    assert abs(point.z - 2.0) < 1e-6


def test_bbox_geometry_projects_center_and_extent(tmp_path: Path) -> None:
    depth = np.full((1, 5, 5), 2.0, dtype=np.float32)
    intrinsics = np.array([[
        [2.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 1.0],
    ]], dtype=np.float32)
    extrinsics = np.array([[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]], dtype=np.float32)
    path = tmp_path / "geometry.npz"
    np.savez(path, depth=depth, intrinsics=intrinsics, extrinsics=extrinsics)

    center, bbox = project_bbox_geometry_from_npz(
        path,
        0,
        (1, 1, 3, 3),
    )

    assert abs(center.x) < 1e-6
    assert abs(center.y) < 1e-6
    assert abs(center.z - 2.0) < 1e-6
    assert bbox.size.x > 1.0
    assert bbox.size.y > 1.0
    assert bbox.size.z <= 1e-5


def test_bbox_geometry_scales_source_coordinates(tmp_path: Path) -> None:
    depth = np.full((1, 5, 5), 2.0, dtype=np.float32)
    intrinsics = np.array([[
        [2.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 1.0],
    ]], dtype=np.float32)
    extrinsics = np.array([[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]], dtype=np.float32)
    path = tmp_path / "geometry-scaled.npz"
    np.savez(path, depth=depth, intrinsics=intrinsics, extrinsics=extrinsics)

    center, bbox = project_bbox_geometry_from_npz(
        path,
        0,
        (2, 2, 6, 6),
        source_size=(10, 10),
    )

    assert abs(center.x) < 1e-6
    assert abs(center.y) < 1e-6
    assert abs(center.z - 2.0) < 1e-6
    assert bbox.size.x > 1.0
