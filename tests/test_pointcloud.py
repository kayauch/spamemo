from pathlib import Path

import numpy as np
import open3d as o3d

from spatial_memory.pointcloud import get_point_cloud_variant


def write_cloud(path: Path, count: int) -> None:
    cloud = o3d.geometry.PointCloud()
    x = np.linspace(0.0, 1.0, count, dtype=np.float64)
    points = np.column_stack([x, x * 0.5, x * 0.25])
    colors = np.column_stack([x, 1.0 - x, np.full(count, 0.5)])
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    assert o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def test_point_cloud_preview_is_downsampled_and_cached(tmp_path: Path) -> None:
    source = tmp_path / "scene.ply"
    write_cloud(source, 1000)

    preview = get_point_cloud_variant(source, max_points=100)
    assert preview != source
    assert preview.exists()

    cloud = o3d.io.read_point_cloud(str(preview))
    assert 1 <= len(cloud.points) <= 100

    first_mtime = preview.stat().st_mtime_ns
    again = get_point_cloud_variant(source, max_points=100)
    assert again == preview
    assert again.stat().st_mtime_ns == first_mtime


def test_small_point_cloud_uses_original_file(tmp_path: Path) -> None:
    source = tmp_path / "small.ply"
    write_cloud(source, 50)

    served = get_point_cloud_variant(source, max_points=100)
    assert served == source
