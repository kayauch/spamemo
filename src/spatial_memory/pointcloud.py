from __future__ import annotations

import math
import os
from pathlib import Path
from threading import Lock

_preview_lock = Lock()


def preview_cache_path(source: Path, max_points: int) -> Path:
    return source.with_name(f"{source.stem}.preview-{max_points}.ply")


def get_point_cloud_variant(source: Path, max_points: int | None = None) -> Path:
    source = Path(source)
    if max_points is None:
        return source
    if max_points < 1:
        raise ValueError("max_points must be positive")

    target = preview_cache_path(source, max_points)
    with _preview_lock:
        if target.exists() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return target

        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(source))
        point_count = len(cloud.points)
        if point_count <= max_points:
            return source

        stride = max(2, math.ceil(point_count / max_points))
        preview = cloud.uniform_down_sample(stride)
        temp = target.with_name(
            f"{target.stem}.tmp-{os.getpid()}.ply"
        )
        try:
            ok = o3d.io.write_point_cloud(
                str(temp),
                preview,
                write_ascii=False,
                compressed=False,
            )
            if not ok:
                raise RuntimeError("failed to write point cloud preview")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)

    return target
