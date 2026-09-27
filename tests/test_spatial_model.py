from pathlib import Path

import numpy as np
import open3d as o3d

from spatial_memory.spatial_model import build_object_layout, build_spatial_model, build_structural_layout, estimate_room_frame


def _synthetic_room_cloud(path: Path) -> None:
    values = np.linspace(-1.0, 1.0, 24)
    a, b = np.meshgrid(values, values, indexing="ij")
    faces = [
        np.column_stack([a.ravel(), np.full(a.size, -1.0), b.ravel()]),
        np.column_stack([a.ravel(), np.full(a.size, 1.0), b.ravel()]),
        np.column_stack([np.full(a.size, -1.0), a.ravel(), b.ravel()]),
        np.column_stack([np.full(a.size, 1.0), a.ravel(), b.ravel()]),
        np.column_stack([a.ravel(), b.ravel(), np.full(a.size, -1.0)]),
        np.column_stack([a.ravel(), b.ravel(), np.full(a.size, 1.0)]),
    ]
    points = np.concatenate(faces, axis=0).astype(np.float64)
    rng = np.random.default_rng(7)
    points += rng.normal(scale=0.002, size=points.shape)

    colors = np.empty_like(points)
    colors[:, 0] = (points[:, 0] + 1.1) / 2.2
    colors[:, 1] = (points[:, 1] + 1.1) / 2.2
    colors[:, 2] = (points[:, 2] + 1.1) / 2.2
    colors = np.clip(colors, 0.0, 1.0)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    assert o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def test_build_spatial_model_persists_shell_and_components(tmp_path: Path) -> None:
    source = tmp_path / "room.ply"
    _synthetic_room_cloud(source)

    manifest = build_spatial_model(
        point_cloud_path=source,
        output_dir=tmp_path / "model",
        scene_id="room",
        target_triangles=2_000,
        max_planes=6,
        max_cluster_components=8,
    )

    manifest_path = Path(manifest["manifest_path"])
    shell_path = Path(manifest["shell"]["path"])
    assert manifest_path.exists()
    assert shell_path.exists()
    assert manifest["version"] == 1
    assert manifest["scene_id"] == "room"
    assert 1 <= manifest["component_count"] <= 14
    assert manifest["shell"]["vertices"] > 0
    assert 100 <= manifest["shell"]["triangles"] <= 2_100
    assert manifest["shell"]["has_vertex_colors"] is True

    kinds = {component["kind"] for component in manifest["components"]}
    assert kinds & {"floor", "ceiling", "wall", "horizontal_surface"}

    for component in manifest["components"]:
        assert len(component["vertices"]) == 8
        assert len(component["triangles"]) == 12
        assert component["id"]
        assert component["point_count"] > 0


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa /= np.linalg.norm(aa)
    bb /= np.linalg.norm(bb)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(aa, bb)), 0.0, 1.0))))


def _plane_component_fixture(
    component_id: str,
    normal: np.ndarray,
    center: np.ndarray,
    area: float,
    point_count: int,
) -> dict:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return {
        "id": component_id,
        "source": "plane",
        "center": {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
        },
        "point_count": point_count,
        "metadata": {
            "normal": normal.tolist(),
            "area": area,
        },
    }


def _tilted_room_fixture(include_diagonal_furniture: bool = True):
    up = np.array([0.08, 0.995, 0.05], dtype=np.float64)
    up /= np.linalg.norm(up)
    global_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    base_x = global_x - up * float(np.dot(global_x, up))
    base_x /= np.linalg.norm(base_x)
    base_z = np.cross(base_x, up)
    base_z /= np.linalg.norm(base_z)

    yaw = np.radians(18.0)
    axis_x = np.cos(yaw) * base_x + np.sin(yaw) * base_z
    axis_x /= np.linalg.norm(axis_x)
    axis_z = np.cross(axis_x, up)
    axis_z /= np.linalg.norm(axis_z)

    origin = np.array([2.5, -0.4, 1.2], dtype=np.float64)
    components = [
        _plane_component_fixture("floor", up, origin, 12.0, 2200),
        _plane_component_fixture("ceiling", -up, origin + up * 2.5, 11.0, 1800),
        _plane_component_fixture("wall-x0", axis_x, origin - axis_x * 2.0 + up * 1.25, 7.5, 1500),
        _plane_component_fixture("wall-x1", -axis_x, origin + axis_x * 2.0 + up * 1.25, 7.5, 1500),
        _plane_component_fixture("wall-z0", axis_z, origin - axis_z * 1.5 + up * 1.25, 10.0, 1700),
        _plane_component_fixture("wall-z1", -axis_z, origin + axis_z * 1.5 + up * 1.25, 10.0, 1700),
    ]
    if include_diagonal_furniture:
        diagonal = np.cos(np.radians(37.0)) * axis_x + np.sin(np.radians(37.0)) * axis_z
        components.append(
            _plane_component_fixture(
                "furniture-diagonal",
                diagonal,
                origin + up * 0.7,
                0.8,
                240,
            )
        )

    xs = np.linspace(-2.0, 2.0, 7)
    ys = np.linspace(0.0, 2.5, 5)
    zs = np.linspace(-1.5, 1.5, 7)
    room_points = []
    for x in xs:
        for y in ys:
            for z in zs:
                room_points.append(origin + axis_x * x + up * y + axis_z * z)
    return components, np.asarray(room_points), up, axis_x, axis_z


def test_room_frame_recovers_floor_up_and_manhattan_axes() -> None:
    components, points, expected_up, expected_x, expected_z = _tilted_room_fixture()

    frame = estimate_room_frame(components, points)

    estimated_up = np.array(list(frame["axis_y_up"].values()))
    estimated_x = np.array(list(frame["axis_x"].values()))
    estimated_z = np.array(list(frame["axis_z"].values()))

    assert frame["mode"] == "manhattan"
    assert frame["snap_enabled"] is True
    assert _angle_degrees(estimated_up, expected_up) < 2.0
    assert _angle_degrees(estimated_x, expected_x) < 3.0
    assert _angle_degrees(estimated_z, expected_z) < 3.0
    assert frame["metrics"]["wall_candidate_count"] >= 4
    assert frame["metrics"]["weighted_residual_mean_degrees"] < 5.0


def test_room_frame_is_not_pulled_by_small_diagonal_furniture_plane() -> None:
    structural, points, _, _, _ = _tilted_room_fixture(include_diagonal_furniture=False)
    with_furniture, _, _, _, _ = _tilted_room_fixture(include_diagonal_furniture=True)

    base = estimate_room_frame(structural, points)
    noisy = estimate_room_frame(with_furniture, points)

    base_x = np.array(list(base["axis_x"].values()))
    noisy_x = np.array(list(noisy["axis_x"].values()))
    assert _angle_degrees(base_x, noisy_x) < 1.5


def test_room_frame_transforms_round_trip_without_modifying_world_points() -> None:
    components, points, _, _, _ = _tilted_room_fixture()
    original = points.copy()

    frame = estimate_room_frame(components, points)
    room_to_world = np.asarray(frame["room_to_world"], dtype=np.float64)
    world_to_room = np.asarray(frame["world_to_room"], dtype=np.float64)

    sample = np.column_stack([points[:20], np.ones(20)])
    room = (world_to_room @ sample.T).T
    restored = (room_to_world @ room.T).T

    np.testing.assert_allclose(restored[:, :3], points[:20], atol=1e-8)
    np.testing.assert_allclose(points, original, atol=0.0)


def test_spatial_model_manifest_contains_room_frame(tmp_path: Path) -> None:
    source = tmp_path / "room-frame.ply"
    _synthetic_room_cloud(source)

    manifest = build_spatial_model(
        point_cloud_path=source,
        output_dir=tmp_path / "model-frame",
        scene_id="room-frame",
        target_triangles=1_500,
        max_planes=6,
        max_cluster_components=4,
    )

    room_frame = manifest["room_frame"]
    assert room_frame["mode"] in {"manhattan", "up_only"}
    assert len(room_frame["room_to_world"]) == 4
    assert len(room_frame["world_to_room"]) == 4
    assert 0.0 <= room_frame["confidence"] <= 1.0
    assert "room_frame_confidence" in manifest["metrics"]


def _box_plane_component(
    component_id: str,
    center: np.ndarray,
    axes: np.ndarray,
    extent: np.ndarray,
    normal: np.ndarray,
    area: float,
    point_count: int,
) -> dict:
    signs = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ], dtype=np.float64)
    vertices = center[None, :] + (signs * (extent * 0.5)[None, :]) @ axes.T
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return {
        "id": component_id,
        "source": "plane",
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "size": {"x": float(extent[0]), "y": float(extent[1]), "z": float(extent[2])},
        "axes": axes.tolist(),
        "vertices": vertices.tolist(),
        "triangles": [],
        "point_count": point_count,
        "metadata": {
            "normal": normal.tolist(),
            "area": area,
        },
    }


def _structural_room_fixture():
    _, points, up, axis_x, axis_z = _tilted_room_fixture(include_diagonal_furniture=False)
    origin = np.array([2.5, -0.4, 1.2], dtype=np.float64)
    thickness = 0.02
    components = [
        _box_plane_component(
            "floor", origin, np.column_stack([axis_x, axis_z, up]),
            np.array([4.0, 3.0, thickness]), up, 12.0, 2200,
        ),
        _box_plane_component(
            "ceiling", origin + up * 2.5, np.column_stack([axis_x, axis_z, up]),
            np.array([4.0, 3.0, thickness]), -up, 12.0, 2000,
        ),
        _box_plane_component(
            "wall-x-min", origin - axis_x * 2.0 + up * 1.25,
            np.column_stack([axis_z, up, axis_x]),
            np.array([3.0, 2.5, thickness]), axis_x, 7.5, 1700,
        ),
        _box_plane_component(
            "wall-x-max", origin + axis_x * 2.0 + up * 1.25,
            np.column_stack([axis_z, up, axis_x]),
            np.array([3.0, 2.5, thickness]), -axis_x, 7.5, 1700,
        ),
        _box_plane_component(
            "wall-z-min", origin - axis_z * 1.5 + up * 1.25,
            np.column_stack([axis_x, up, axis_z]),
            np.array([4.0, 2.5, thickness]), axis_z, 10.0, 1800,
        ),
        _box_plane_component(
            "wall-z-max", origin + axis_z * 1.5 + up * 1.25,
            np.column_stack([axis_x, up, axis_z]),
            np.array([4.0, 2.5, thickness]), -axis_z, 10.0, 1800,
        ),
    ]
    frame = estimate_room_frame(components, points)
    return components, points, frame


def test_structural_layout_regularizes_floor_ceiling_and_four_walls() -> None:
    components, points, frame = _structural_room_fixture()

    layout = build_structural_layout(
        plane_components=components,
        room_frame=frame,
        scene_points=points,
        scene_diagonal=float(np.linalg.norm(np.ptp(points, axis=0))),
    )

    assert layout["enabled"] is True
    assert layout["floor_component_id"] == "struct-floor"
    assert layout["ceiling_component_id"] == "struct-ceiling"
    assert layout["wall_count"] == 4
    assert layout["component_count"] == 6
    assert layout["footprint"]["closed"] is True
    assert len(layout["footprint"]["polygon_room_xz"]) == 4

    room_x = np.array(list(frame["axis_x"].values()))
    room_up = np.array(list(frame["axis_y_up"].values()))
    room_z = np.array(list(frame["axis_z"].values()))

    for component in layout["components"]:
        axes = np.asarray(component["axes"], dtype=np.float64)
        if component["kind"] in {"floor", "ceiling"}:
            assert _angle_degrees(axes[:, 2], room_up) < 1e-6
        elif component["kind"] == "wall":
            target = component["metadata"]["target_axis"]
            expected = room_x if target == "x" else room_z
            assert _angle_degrees(axes[:, 2], expected) < 1e-6
            assert component["metadata"]["regularized"] is True


def test_structural_layout_does_not_force_snap_when_room_frame_is_weak() -> None:
    components, points, frame = _structural_room_fixture()
    weak = dict(frame)
    weak["snap_enabled"] = False
    weak["confidence"] = 0.2

    layout = build_structural_layout(
        plane_components=components,
        room_frame=weak,
        scene_points=points,
        scene_diagonal=float(np.linalg.norm(np.ptp(points, axis=0))),
    )

    assert layout["enabled"] is False
    assert layout["component_count"] == 0
    assert layout["footprint"]["closed"] is False


def _object_component_fixture(
    component_id: str,
    frame: dict,
    center_room: np.ndarray,
    yaw_degrees: float,
    extent: np.ndarray,
    point_count: int = 600,
) -> dict:
    room_to_world = np.asarray(frame["room_to_world"], dtype=np.float64)
    axis_x = np.array(list(frame["axis_x"].values()), dtype=np.float64)
    up = np.array(list(frame["axis_y_up"].values()), dtype=np.float64)
    axis_z = np.array(list(frame["axis_z"].values()), dtype=np.float64)
    yaw = np.radians(yaw_degrees)
    local_x = np.cos(yaw) * axis_x + np.sin(yaw) * axis_z
    local_z = -np.sin(yaw) * axis_x + np.cos(yaw) * axis_z
    axes = np.column_stack([local_x, up, local_z])

    homogeneous = np.append(np.asarray(center_room, dtype=np.float64), 1.0)
    center_world = (room_to_world @ homogeneous)[:3]
    signs = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ], dtype=np.float64)
    vertices = center_world[None, :] + (signs * (extent * 0.5)[None, :]) @ axes.T
    return {
        "id": component_id,
        "kind": "object_region",
        "source": "cluster",
        "center": {
            "x": float(center_world[0]),
            "y": float(center_world[1]),
            "z": float(center_world[2]),
        },
        "size": {
            "x": float(extent[0]),
            "y": float(extent[1]),
            "z": float(extent[2]),
        },
        "axes": axes.tolist(),
        "vertices": vertices.tolist(),
        "triangles": [],
        "point_count": point_count,
        "confidence": 0.8,
        "metadata": {},
    }


def _object_layout_fixture():
    structural_components, points, frame = _structural_room_fixture()
    structural = build_structural_layout(
        plane_components=structural_components,
        room_frame=frame,
        scene_points=points,
        scene_diagonal=float(np.linalg.norm(np.ptp(points, axis=0))),
    )
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    return points, frame, structural, diagonal


def test_object_layout_snaps_small_yaw_and_grounds_near_floor() -> None:
    points, frame, structural, diagonal = _object_layout_fixture()
    extent = np.array([1.2, 0.8, 0.6], dtype=np.float64)
    component = _object_component_fixture(
        "near-axis",
        frame,
        center_room=np.array([0.0, 0.03 + extent[1] * 0.5, 0.0]),
        yaw_degrees=8.0,
        extent=extent,
    )

    layout = build_object_layout(
        components=[component],
        room_frame=frame,
        structural_layout=structural,
        scene_points=points,
        scene_diagonal=diagonal,
    )

    assert layout["component_count"] == 1
    assert layout["snapped_count"] == 1
    assert layout["grounded_count"] == 1
    result = layout["components"][0]
    assert result["metadata"]["snap_applied"] is True
    assert result["metadata"]["grounded"] is True
    assert result["metadata"]["yaw_residual_degrees"] < 10.0
    assert result["metadata"]["volume_expansion"] < 1.50

    world_to_room = np.asarray(frame["world_to_room"], dtype=np.float64)
    room_vertices = _transform_fixture(np.asarray(result["vertices"]), world_to_room)
    assert abs(float(room_vertices[:, 1].min())) < 1e-8


def _transform_fixture(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([values, np.ones(len(values))])
    return (np.asarray(matrix) @ homogeneous.T).T[:, :3]


def test_object_layout_preserves_intentional_diagonal_orientation() -> None:
    points, frame, structural, diagonal = _object_layout_fixture()
    extent = np.array([1.2, 0.8, 0.5], dtype=np.float64)
    component = _object_component_fixture(
        "diagonal",
        frame,
        center_room=np.array([0.0, 0.02 + extent[1] * 0.5, 0.0]),
        yaw_degrees=45.0,
        extent=extent,
    )

    layout = build_object_layout(
        components=[component],
        room_frame=frame,
        structural_layout=structural,
        scene_points=points,
        scene_diagonal=diagonal,
    )

    result = layout["components"][0]
    assert result["metadata"]["snap_applied"] is False
    assert result["metadata"]["yaw_residual_degrees"] > 40.0
    assert result["metadata"]["grounded"] is True

    raw_axes = np.asarray(component["axes"], dtype=np.float64)
    output_axes = np.asarray(result["axes"], dtype=np.float64)
    np.testing.assert_allclose(output_axes, raw_axes, atol=1e-8)


def test_object_layout_does_not_drop_floating_component_to_floor() -> None:
    points, frame, structural, diagonal = _object_layout_fixture()
    extent = np.array([0.8, 0.5, 0.5], dtype=np.float64)
    component = _object_component_fixture(
        "floating",
        frame,
        center_room=np.array([0.2, 0.45 + extent[1] * 0.5, -0.1]),
        yaw_degrees=6.0,
        extent=extent,
    )

    layout = build_object_layout(
        components=[component],
        room_frame=frame,
        structural_layout=structural,
        scene_points=points,
        scene_diagonal=diagonal,
    )

    result = layout["components"][0]
    assert result["metadata"]["snap_applied"] is True
    assert result["metadata"]["grounded"] is False
    assert result["metadata"]["floor_shift_room_y"] == 0.0


def test_spatial_model_manifest_contains_object_layout(tmp_path: Path) -> None:
    source = tmp_path / "room-object-layout.ply"
    _synthetic_room_cloud(source)

    manifest = build_spatial_model(
        point_cloud_path=source,
        output_dir=tmp_path / "model-object-layout",
        scene_id="room-object-layout",
        target_triangles=1_200,
        max_planes=6,
        max_cluster_components=8,
    )

    assert "object_layout" in manifest
    assert "object_component_count" in manifest["metrics"]
    assert "object_snapped_count" in manifest["metrics"]
    assert "object_grounded_count" in manifest["metrics"]
