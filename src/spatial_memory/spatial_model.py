from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


_BOX_TRIANGLES = [
    [0, 1, 2], [0, 2, 3],
    [4, 6, 5], [4, 7, 6],
    [0, 4, 5], [0, 5, 1],
    [1, 5, 6], [1, 6, 2],
    [2, 6, 7], [2, 7, 3],
    [3, 7, 4], [3, 4, 0],
]


def _robust_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise ValueError("Point cloud is empty")
    return (
        np.quantile(points, 0.01, axis=0),
        np.quantile(points, 0.99, axis=0),
    )


def _scene_diagonal(points: np.ndarray) -> float:
    lower, upper = _robust_bounds(points)
    return float(np.linalg.norm(upper - lower))


def _box_vertices(
    center: np.ndarray,
    axes: np.ndarray,
    extent: np.ndarray,
) -> list[list[float]]:
    half = extent * 0.5
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
    local = signs * half[None, :]
    world = center[None, :] + local @ axes.T
    return [[float(value) for value in row] for row in world]


def _component(
    component_id: str,
    kind: str,
    center: np.ndarray,
    axes: np.ndarray,
    extent: np.ndarray,
    point_count: int,
    source: str,
    confidence: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": component_id,
        "kind": kind,
        "center": {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
        },
        "size": {
            "x": float(extent[0]),
            "y": float(extent[1]),
            "z": float(extent[2]),
        },
        "axes": [[float(v) for v in row] for row in axes.tolist()],
        "vertices": _box_vertices(center, axes, extent),
        "triangles": _BOX_TRIANGLES,
        "point_count": int(point_count),
        "source": source,
        "confidence": float(confidence),
        "metadata": metadata or {},
    }


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > 1e-9:
        return value / norm
    if fallback is None:
        raise ValueError("Cannot normalize a near-zero vector")
    return _unit(np.asarray(fallback, dtype=np.float64))


def _axis_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    aa = _unit(a)
    bb = _unit(b)
    cosine = float(np.clip(abs(np.dot(aa, bb)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _plane_records(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for component in components:
        if component.get("source") != "plane":
            continue
        metadata = component.get("metadata") or {}
        normal = metadata.get("normal")
        if normal is None:
            continue
        center_data = component.get("center") or {}
        center = np.array([
            float(center_data.get("x", 0.0)),
            float(center_data.get("y", 0.0)),
            float(center_data.get("z", 0.0)),
        ], dtype=np.float64)
        area = max(float(metadata.get("area", 0.0)), 1e-6)
        point_count = max(int(component.get("point_count", 0)), 1)
        weight = area * math.sqrt(float(point_count))
        records.append({
            "id": component.get("id"),
            "normal": _unit(np.asarray(normal, dtype=np.float64)),
            "center": center,
            "area": area,
            "point_count": point_count,
            "weight": weight,
        })
    return records


def estimate_room_frame(
    plane_components: list[dict[str, Any]],
    scene_points: np.ndarray,
    global_up: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate a display/modeling room frame without changing world coordinates.

    The returned transform maps room-local coordinates into the existing VGGT
    world coordinate system. Normal directions are treated as unsigned because
    plane fitting can flip them arbitrarily.
    """
    points = np.asarray(scene_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("scene_points must be an Nx3 array")

    records = _plane_records(plane_components)
    prior_up = _unit(
        np.array([0.0, 1.0, 0.0]) if global_up is None else np.asarray(global_up),
        fallback=np.array([0.0, 1.0, 0.0]),
    )
    total_weight = sum(record["weight"] for record in records) or 1.0

    up_seed = prior_up.copy()
    up_candidates: list[dict[str, Any]] = []
    up_gate = math.cos(math.radians(35.0))
    for record in records:
        alignment = abs(float(np.dot(record["normal"], prior_up)))
        if alignment >= up_gate:
            scored = dict(record)
            scored["prior_alignment"] = alignment
            up_candidates.append(scored)

    if up_candidates:
        seed = max(
            up_candidates,
            key=lambda item: item["weight"] * item["prior_alignment"] ** 4,
        )
        up_seed = seed["normal"].copy()
        if float(np.dot(up_seed, prior_up)) < 0.0:
            up_seed *= -1.0

        aligned_sum = np.zeros(3, dtype=np.float64)
        aligned_weight = 0.0
        refine_gate = math.cos(math.radians(22.0))
        for record in up_candidates:
            normal = record["normal"].copy()
            if float(np.dot(normal, up_seed)) < 0.0:
                normal *= -1.0
            alignment = float(np.dot(normal, up_seed))
            if alignment < refine_gate:
                continue
            weight = record["weight"] * alignment * alignment
            aligned_sum += normal * weight
            aligned_weight += weight
        up = _unit(aligned_sum, fallback=up_seed) if aligned_weight else up_seed
        if float(np.dot(up, prior_up)) < 0.0:
            up *= -1.0
    else:
        up = prior_up.copy()

    up_support_weight = 0.0
    up_alignment_sum = 0.0
    horizontal_records: list[dict[str, Any]] = []
    wall_dot_limit = math.sin(math.radians(30.0))
    for record in records:
        alignment = abs(float(np.dot(record["normal"], up)))
        if alignment >= math.cos(math.radians(25.0)):
            up_support_weight += record["weight"]
            up_alignment_sum += record["weight"] * alignment
        if alignment <= wall_dot_limit:
            projected = record["normal"] - up * float(np.dot(record["normal"], up))
            projected_norm = float(np.linalg.norm(projected))
            if projected_norm > 1e-6:
                item = dict(record)
                item["horizontal_normal"] = projected / projected_norm
                horizontal_records.append(item)

    world_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    h0 = world_x - up * float(np.dot(world_x, up))
    if float(np.linalg.norm(h0)) < 1e-6:
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        h0 = world_z - up * float(np.dot(world_z, up))
    h0 = _unit(h0)
    h1 = _unit(np.cross(h0, up))

    sum_cos = 0.0
    sum_sin = 0.0
    wall_weight = 0.0
    for record in horizontal_records:
        normal = record["horizontal_normal"]
        angle = math.atan2(float(np.dot(normal, h1)), float(np.dot(normal, h0)))
        weight = float(record["weight"])
        sum_cos += weight * math.cos(4.0 * angle)
        sum_sin += weight * math.sin(4.0 * angle)
        wall_weight += weight

    if wall_weight > 0.0:
        theta = 0.25 * math.atan2(sum_sin, sum_cos)
        axis_x = _unit(math.cos(theta) * h0 + math.sin(theta) * h1)
        wall_alignment = float(math.hypot(sum_cos, sum_sin) / wall_weight)
    else:
        theta = 0.0
        axis_x = h0
        wall_alignment = 0.0

    if float(np.dot(axis_x, h0)) < 0.0:
        axis_x *= -1.0
    axis_z = _unit(np.cross(axis_x, up))
    axis_x = _unit(np.cross(up, axis_z))

    lower, upper = _robust_bounds(points)
    robust_center = (lower + upper) * 0.5

    horizontal_candidates = [
        record
        for record in records
        if abs(float(np.dot(record["normal"], up)))
        >= math.cos(math.radians(25.0))
    ]
    if horizontal_candidates:
        strongest = max(record["weight"] for record in horizontal_candidates)
        structural = [
            record
            for record in horizontal_candidates
            if record["weight"] >= strongest * 0.15
        ]
        floor_record = min(
            structural or horizontal_candidates,
            key=lambda item: float(np.dot(item["center"], up)),
        )
        floor_height = float(np.dot(floor_record["center"], up))
        floor_source = floor_record.get("id")
    else:
        projected_heights = points @ up
        floor_height = float(np.quantile(projected_heights, 0.02))
        floor_source = None

    origin = robust_center + up * (
        floor_height - float(np.dot(robust_center, up))
    )
    rotation = np.column_stack([axis_x, up, axis_z])
    room_to_world = np.eye(4, dtype=np.float64)
    room_to_world[:3, :3] = rotation
    room_to_world[:3, 3] = origin
    world_to_room = np.eye(4, dtype=np.float64)
    world_to_room[:3, :3] = rotation.T
    world_to_room[:3, 3] = -rotation.T @ origin

    residuals: list[tuple[float, float]] = []
    axes = [axis_x, up, axis_z]
    for record in records:
        residual = min(
            _axis_angle_degrees(record["normal"], axis)
            for axis in axes
        )
        residuals.append((residual, record["weight"]))

    if residuals:
        residual_values = np.array([value for value, _ in residuals])
        residual_weights = np.array([weight for _, weight in residuals])
        residual_mean = float(np.average(residual_values, weights=residual_weights))
        ordering = np.argsort(residual_values)
        sorted_values = residual_values[ordering]
        sorted_weights = residual_weights[ordering]
        cumulative = np.cumsum(sorted_weights) / max(float(sorted_weights.sum()), 1e-9)
        residual_p90 = float(sorted_values[min(np.searchsorted(cumulative, 0.90), len(sorted_values) - 1)])
    else:
        residual_mean = 90.0
        residual_p90 = 90.0

    up_alignment = (
        float(up_alignment_sum / up_support_weight)
        if up_support_weight > 0.0
        else 0.0
    )
    up_support_ratio = float(up_support_weight / total_weight)
    wall_support_ratio = float(wall_weight / total_weight)
    structural_support = min(1.0, up_support_ratio + wall_support_ratio)
    confidence = float(np.clip(
        0.35 * up_alignment
        + 0.45 * wall_alignment
        + 0.20 * structural_support,
        0.0,
        1.0,
    ))
    mode = "manhattan" if len(horizontal_records) >= 2 and wall_alignment >= 0.35 else "up_only"
    snap_enabled = bool(mode == "manhattan" and confidence >= 0.55)

    def vector_dict(vector: np.ndarray) -> dict[str, float]:
        return {
            "x": float(vector[0]),
            "y": float(vector[1]),
            "z": float(vector[2]),
        }

    return {
        "mode": mode,
        "snap_enabled": snap_enabled,
        "confidence": confidence,
        "origin": vector_dict(origin),
        "axis_x": vector_dict(axis_x),
        "axis_y_up": vector_dict(up),
        "axis_z": vector_dict(axis_z),
        "room_to_world": room_to_world.tolist(),
        "world_to_room": world_to_room.tolist(),
        "floor_height_world_up": floor_height,
        "floor_source_component_id": floor_source,
        "metrics": {
            "plane_count": len(records),
            "up_candidate_count": len(up_candidates),
            "wall_candidate_count": len(horizontal_records),
            "up_support_ratio": up_support_ratio,
            "wall_support_ratio": wall_support_ratio,
            "up_alignment": up_alignment,
            "wall_alignment": wall_alignment,
            "weighted_residual_mean_degrees": residual_mean,
            "weighted_residual_p90_degrees": residual_p90,
            "yaw_from_world_x_degrees": float(math.degrees(theta)),
        },
    }


def _vector_from_dict(value: dict[str, Any]) -> np.ndarray:
    return np.array([
        float(value.get("x", 0.0)),
        float(value.get("y", 0.0)),
        float(value.get("z", 0.0)),
    ], dtype=np.float64)


def _transform_xyz(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([values, np.ones(len(values), dtype=np.float64)])
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T[:, :3]


def build_structural_layout(
    plane_components: list[dict[str, Any]],
    room_frame: dict[str, Any],
    scene_points: np.ndarray,
    scene_diagonal: float,
) -> dict[str, Any]:
    """Create clean floor/wall/ceiling components in the derived room frame."""
    if not room_frame.get("snap_enabled"):
        return {
            "enabled": False,
            "reason": "room_frame_confidence_too_low",
            "components": [],
            "component_count": 0,
            "wall_count": 0,
            "footprint": {
                "closed": False,
                "polygon_room_xz": [],
                "wall_segments": [],
            },
        }

    world_to_room = np.asarray(room_frame["world_to_room"], dtype=np.float64)
    room_to_world = np.asarray(room_frame["room_to_world"], dtype=np.float64)
    room_points = _transform_xyz(np.asarray(scene_points, dtype=np.float64), world_to_room)
    room_lower, room_upper = _robust_bounds(room_points)
    room_span = np.maximum(room_upper - room_lower, 1e-6)
    room_mid = (room_lower + room_upper) * 0.5

    axis_x = _vector_from_dict(room_frame["axis_x"])
    up = _vector_from_dict(room_frame["axis_y_up"])
    axis_z = _vector_from_dict(room_frame["axis_z"])
    thickness = max(0.015, scene_diagonal * 0.007)

    records: list[dict[str, Any]] = []
    for component in plane_components:
        if component.get("source") != "plane":
            continue
        metadata = component.get("metadata") or {}
        normal_data = metadata.get("normal")
        vertices = component.get("vertices") or []
        if normal_data is None or len(vertices) != 8:
            continue
        normal = _unit(np.asarray(normal_data, dtype=np.float64))
        center_world = _vector_from_dict(component.get("center") or {})
        center_room = _transform_xyz(center_world[None, :], world_to_room)[0]
        vertices_room = _transform_xyz(np.asarray(vertices, dtype=np.float64), world_to_room)
        extents_room = vertices_room.max(axis=0) - vertices_room.min(axis=0)
        records.append({
            "component": component,
            "normal": normal,
            "center_room": center_room,
            "vertices_room": vertices_room,
            "extents_room": extents_room,
            "area": float(metadata.get("area", 0.0)),
            "point_count": int(component.get("point_count", 0)),
            "angle_up": _axis_angle_degrees(normal, up),
            "angle_x": _axis_angle_degrees(normal, axis_x),
            "angle_z": _axis_angle_degrees(normal, axis_z),
        })

    structural: list[dict[str, Any]] = []
    source_floor_id = room_frame.get("floor_source_component_id")
    floor_record = next(
        (
            record
            for record in records
            if record["component"].get("id") == source_floor_id
        ),
        None,
    )

    def room_to_world_point(point: np.ndarray) -> np.ndarray:
        return _transform_xyz(np.asarray(point, dtype=np.float64)[None, :], room_to_world)[0]

    if floor_record is not None:
        vertices_room = floor_record["vertices_room"]
        x0, x1 = np.quantile(vertices_room[:, 0], [0.01, 0.99])
        z0, z1 = np.quantile(vertices_room[:, 2], [0.01, 0.99])
        center_room = np.array([
            (x0 + x1) * 0.5,
            0.0,
            (z0 + z1) * 0.5,
        ])
        structural.append(
            _component(
                component_id="struct-floor",
                kind="floor",
                center=room_to_world_point(center_room),
                axes=np.column_stack([axis_x, axis_z, up]),
                extent=np.array([
                    max(float(x1 - x0), thickness),
                    max(float(z1 - z0), thickness),
                    thickness,
                ]),
                point_count=floor_record["point_count"],
                source="structural_regularized",
                confidence=float(room_frame["confidence"]),
                metadata={
                    "raw_component_ids": [source_floor_id],
                    "target_axis": "y",
                    "side": "min",
                    "room_plane_coordinate": 0.0,
                    "angular_residual_degrees": floor_record["angle_up"],
                    "regularized": True,
                },
            )
        )

    horizontal_records = [
        record
        for record in records
        if record["angle_up"] <= 20.0
        and record["component"].get("id") != source_floor_id
    ]
    ceiling_record = None
    if horizontal_records:
        floor_area = floor_record["area"] if floor_record is not None else max(
            record["area"] for record in horizontal_records
        )
        candidates = [
            record
            for record in horizontal_records
            if record["center_room"][1] >= room_lower[1] + room_span[1] * 0.65
            and record["area"] >= max(floor_area * 0.18, 0.05)
        ]
        if candidates:
            ceiling_record = max(candidates, key=lambda record: record["center_room"][1])

    if ceiling_record is not None:
        vertices_room = ceiling_record["vertices_room"]
        x0, x1 = np.quantile(vertices_room[:, 0], [0.01, 0.99])
        z0, z1 = np.quantile(vertices_room[:, 2], [0.01, 0.99])
        plane_y = float(np.median(vertices_room[:, 1]))
        center_room = np.array([
            (x0 + x1) * 0.5,
            plane_y,
            (z0 + z1) * 0.5,
        ])
        structural.append(
            _component(
                component_id="struct-ceiling",
                kind="ceiling",
                center=room_to_world_point(center_room),
                axes=np.column_stack([axis_x, axis_z, up]),
                extent=np.array([
                    max(float(x1 - x0), thickness),
                    max(float(z1 - z0), thickness),
                    thickness,
                ]),
                point_count=ceiling_record["point_count"],
                source="structural_regularized",
                confidence=float(room_frame["confidence"]),
                metadata={
                    "raw_component_ids": [ceiling_record["component"].get("id")],
                    "target_axis": "y",
                    "side": "max",
                    "room_plane_coordinate": plane_y,
                    "angular_residual_degrees": ceiling_record["angle_up"],
                    "regularized": True,
                },
            )
        )

    wall_candidates: list[dict[str, Any]] = []
    for record in records:
        target_axis = "x" if record["angle_x"] <= record["angle_z"] else "z"
        residual = record["angle_x"] if target_axis == "x" else record["angle_z"]
        if residual > 20.0:
            continue

        axis_index = 0 if target_axis == "x" else 2
        along_index = 2 if target_axis == "x" else 0
        coordinate = float(record["center_room"][axis_index])
        side = "min" if coordinate <= room_mid[axis_index] else "max"
        boundary = room_lower[axis_index] if side == "min" else room_upper[axis_index]
        boundary_distance = abs(coordinate - float(boundary))
        if boundary_distance > room_span[axis_index] * 0.22:
            continue

        vertical_span = float(record["extents_room"][1])
        along_span = float(record["extents_room"][along_index])
        if vertical_span < room_span[1] * 0.28:
            continue
        if along_span < room_span[along_index] * 0.16:
            continue

        item = dict(record)
        item.update({
            "target_axis": target_axis,
            "axis_index": axis_index,
            "along_index": along_index,
            "coordinate": coordinate,
            "side": side,
            "residual": residual,
        })
        wall_candidates.append(item)

    grouped: list[list[dict[str, Any]]] = []
    for key in [("x", "min"), ("x", "max"), ("z", "min"), ("z", "max")]:
        candidates = sorted(
            [
                item
                for item in wall_candidates
                if (item["target_axis"], item["side"]) == key
            ],
            key=lambda item: item["coordinate"],
        )
        if not candidates:
            continue
        axis_index = 0 if key[0] == "x" else 2
        merge_tolerance = max(0.04, float(room_span[axis_index]) * 0.04)
        current = [candidates[0]]
        for candidate in candidates[1:]:
            current_coordinate = float(np.average(
                [item["coordinate"] for item in current],
                weights=[max(item["area"], 1e-6) for item in current],
            ))
            if abs(candidate["coordinate"] - current_coordinate) <= merge_tolerance:
                current.append(candidate)
            else:
                grouped.append(current)
                current = [candidate]
        grouped.append(current)

    wall_segments: list[dict[str, Any]] = []
    for wall_index, group in enumerate(grouped):
        target_axis = group[0]["target_axis"]
        side = group[0]["side"]
        weights = np.array([max(item["area"], 1e-6) for item in group], dtype=np.float64)
        coordinate = float(np.average(
            np.array([item["coordinate"] for item in group], dtype=np.float64),
            weights=weights,
        ))
        all_vertices = np.concatenate([item["vertices_room"] for item in group], axis=0)
        raw_ids = [item["component"].get("id") for item in group]
        point_count = sum(item["point_count"] for item in group)
        residual = float(np.average(
            np.array([item["residual"] for item in group], dtype=np.float64),
            weights=weights,
        ))

        y0, y1 = np.quantile(all_vertices[:, 1], [0.01, 0.99])
        if target_axis == "x":
            z0, z1 = np.quantile(all_vertices[:, 2], [0.01, 0.99])
            center_room = np.array([coordinate, (y0 + y1) * 0.5, (z0 + z1) * 0.5])
            axes = np.column_stack([axis_z, up, axis_x])
            extent = np.array([
                max(float(z1 - z0), thickness),
                max(float(y1 - y0), thickness),
                thickness,
            ])
            segment = {
                "axis": "x",
                "side": side,
                "coordinate": coordinate,
                "start": {"x": coordinate, "z": float(z0)},
                "end": {"x": coordinate, "z": float(z1)},
            }
        else:
            x0, x1 = np.quantile(all_vertices[:, 0], [0.01, 0.99])
            center_room = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, coordinate])
            axes = np.column_stack([axis_x, up, axis_z])
            extent = np.array([
                max(float(x1 - x0), thickness),
                max(float(y1 - y0), thickness),
                thickness,
            ])
            segment = {
                "axis": "z",
                "side": side,
                "coordinate": coordinate,
                "start": {"x": float(x0), "z": coordinate},
                "end": {"x": float(x1), "z": coordinate},
            }

        structural.append(
            _component(
                component_id=f"struct-wall-{wall_index:03d}",
                kind="wall",
                center=room_to_world_point(center_room),
                axes=axes,
                extent=extent,
                point_count=point_count,
                source="structural_regularized",
                confidence=float(room_frame["confidence"]),
                metadata={
                    "raw_component_ids": raw_ids,
                    "target_axis": target_axis,
                    "side": side,
                    "room_plane_coordinate": coordinate,
                    "angular_residual_degrees": residual,
                    "regularized": True,
                },
            )
        )
        segment["component_id"] = f"struct-wall-{wall_index:03d}"
        wall_segments.append(segment)

    sides = {(segment["axis"], segment["side"]) for segment in wall_segments}
    closed = all(
        key in sides
        for key in [("x", "min"), ("x", "max"), ("z", "min"), ("z", "max")]
    )
    polygon: list[dict[str, float]] = []
    if closed:
        x_min = min(
            segment["coordinate"]
            for segment in wall_segments
            if segment["axis"] == "x" and segment["side"] == "min"
        )
        x_max = max(
            segment["coordinate"]
            for segment in wall_segments
            if segment["axis"] == "x" and segment["side"] == "max"
        )
        z_min = min(
            segment["coordinate"]
            for segment in wall_segments
            if segment["axis"] == "z" and segment["side"] == "min"
        )
        z_max = max(
            segment["coordinate"]
            for segment in wall_segments
            if segment["axis"] == "z" and segment["side"] == "max"
        )
        polygon = [
            {"x": float(x_min), "z": float(z_min)},
            {"x": float(x_max), "z": float(z_min)},
            {"x": float(x_max), "z": float(z_max)},
            {"x": float(x_min), "z": float(z_max)},
        ]

    return {
        "enabled": True,
        "confidence": float(room_frame["confidence"]),
        "components": structural,
        "component_count": len(structural),
        "floor_component_id": "struct-floor" if floor_record is not None else None,
        "ceiling_component_id": "struct-ceiling" if ceiling_record is not None else None,
        "wall_count": len(wall_segments),
        "room_bounds": {
            "min": {
                "x": float(room_lower[0]),
                "y": float(room_lower[1]),
                "z": float(room_lower[2]),
            },
            "max": {
                "x": float(room_upper[0]),
                "y": float(room_upper[1]),
                "z": float(room_upper[2]),
            },
        },
        "footprint": {
            "closed": closed,
            "polygon_room_xz": polygon,
            "wall_segments": wall_segments,
        },
    }


def build_object_layout(
    components: list[dict[str, Any]],
    room_frame: dict[str, Any],
    structural_layout: dict[str, Any],
    scene_points: np.ndarray,
    scene_diagonal: float,
    max_yaw_residual_degrees: float = 18.0,
    max_volume_expansion: float = 1.50,
) -> dict[str, Any]:
    """Regularize object-region OBBs while preserving intentional diagonals."""
    if not room_frame.get("snap_enabled"):
        return {
            "enabled": False,
            "reason": "room_frame_confidence_too_low",
            "components": [],
            "component_count": 0,
            "snapped_count": 0,
            "grounded_count": 0,
        }

    world_to_room = np.asarray(room_frame["world_to_room"], dtype=np.float64)
    room_to_world = np.asarray(room_frame["room_to_world"], dtype=np.float64)
    room_rotation = world_to_room[:3, :3]
    axis_x = _vector_from_dict(room_frame["axis_x"])
    up = _vector_from_dict(room_frame["axis_y_up"])
    axis_z = _vector_from_dict(room_frame["axis_z"])
    thickness = max(0.015, scene_diagonal * 0.006)
    ground_tolerance = max(0.04, scene_diagonal * 0.04)
    wall_tolerance = max(0.05, scene_diagonal * 0.05)

    wall_segments = (
        structural_layout.get("footprint", {}).get("wall_segments", [])
        if structural_layout.get("enabled")
        else []
    )

    def room_to_world_point(point: np.ndarray) -> np.ndarray:
        return _transform_xyz(np.asarray(point, dtype=np.float64)[None, :], room_to_world)[0]

    output: list[dict[str, Any]] = []
    snapped_count = 0
    grounded_count = 0

    for raw_index, component in enumerate(
        item for item in components if item.get("source") == "cluster"
    ):
        vertices = np.asarray(component.get("vertices") or [], dtype=np.float64)
        axes_world = np.asarray(component.get("axes") or [], dtype=np.float64)
        if vertices.shape != (8, 3) or axes_world.shape != (3, 3):
            continue

        size_data = component.get("size") or {}
        raw_extent = np.array([
            float(size_data.get("x", 0.0)),
            float(size_data.get("y", 0.0)),
            float(size_data.get("z", 0.0)),
        ], dtype=np.float64)
        raw_extent = np.maximum(raw_extent, thickness)
        raw_volume = float(np.prod(raw_extent))

        vertices_room = _transform_xyz(vertices, world_to_room)
        component_metadata = component.get("metadata") or {}
        room_fit = component_metadata.get("room_aligned_fit")
        if room_fit:
            room_min = np.asarray(room_fit["min"], dtype=np.float64)
            room_max = np.asarray(room_fit["max"], dtype=np.float64)
            snapped_extent = np.maximum(
                np.asarray(room_fit["extent"], dtype=np.float64),
                thickness,
            )
            volume_expansion = float(room_fit["volume_expansion"])
        else:
            room_min = vertices_room.min(axis=0)
            room_max = vertices_room.max(axis=0)
            snapped_extent = np.maximum(room_max - room_min, thickness)
            snapped_volume = float(np.prod(snapped_extent))
            volume_expansion = snapped_volume / max(raw_volume, 1e-9)

        axes_room = room_rotation @ axes_world
        best_horizontal_index = None
        best_horizontal_score = -1.0
        for axis_index in range(3):
            direction = axes_room[:, axis_index]
            horizontal_length = float(np.hypot(direction[0], direction[2]))
            score = horizontal_length * float(raw_extent[axis_index])
            if horizontal_length >= 0.55 and score > best_horizontal_score:
                best_horizontal_index = axis_index
                best_horizontal_score = score

        if best_horizontal_index is None:
            yaw_residual = 45.0
        else:
            direction = axes_room[:, best_horizontal_index]
            angle = math.degrees(math.atan2(abs(float(direction[2])), abs(float(direction[0]))))
            yaw_residual = min(angle, 90.0 - angle)

        snap_applied = bool(
            yaw_residual <= max_yaw_residual_degrees
            and volume_expansion <= max_volume_expansion
        )

        center_world_raw = _vector_from_dict(component.get("center") or {})
        raw_center_room = _transform_xyz(center_world_raw[None, :], world_to_room)[0]
        shift_room_y = 0.0
        bottom_room_y = float(room_min[1])
        grounded = bool(
            -ground_tolerance <= bottom_room_y <= ground_tolerance * 1.5
        )
        if grounded:
            shift_room_y = -bottom_room_y
            grounded_count += 1

        nearest_wall_id = None
        nearest_wall_distance = None
        for segment in wall_segments:
            axis_index = 0 if segment.get("axis") == "x" else 2
            coordinate = float(segment.get("coordinate", 0.0))
            if coordinate < room_min[axis_index]:
                distance = float(room_min[axis_index] - coordinate)
            elif coordinate > room_max[axis_index]:
                distance = float(coordinate - room_max[axis_index])
            else:
                distance = 0.0
            if nearest_wall_distance is None or distance < nearest_wall_distance:
                nearest_wall_distance = distance
                nearest_wall_id = segment.get("component_id")

        near_wall = bool(
            nearest_wall_distance is not None
            and nearest_wall_distance <= wall_tolerance
        )

        reasons: list[str] = []
        if snap_applied:
            snapped_count += 1
            reasons.append("room_axis")
            center_room = (room_min + room_max) * 0.5
            center_room[1] += shift_room_y
            output_component = _component(
                component_id=f"object-{raw_index:03d}",
                kind="object_region",
                center=room_to_world_point(center_room),
                axes=np.column_stack([axis_x, up, axis_z]),
                extent=snapped_extent,
                point_count=int(component.get("point_count", 0)),
                source="object_regularized",
                confidence=float(component.get("confidence", 0.0)),
                metadata={},
            )
        else:
            center_world = center_world_raw + up * shift_room_y
            output_component = _component(
                component_id=f"object-{raw_index:03d}",
                kind="object_region",
                center=center_world,
                axes=axes_world,
                extent=raw_extent,
                point_count=int(component.get("point_count", 0)),
                source="object_regularized",
                confidence=float(component.get("confidence", 0.0)),
                metadata={},
            )

        if grounded:
            reasons.append("floor_support")
        if near_wall:
            reasons.append("near_wall")

        output_component["metadata"] = {
            "raw_component_id": component.get("id"),
            "raw_center": component.get("center"),
            "raw_axes": component.get("axes"),
            "raw_size": component.get("size"),
            "snap_applied": snap_applied,
            "snap_reasons": reasons,
            "yaw_residual_degrees": float(yaw_residual),
            "volume_expansion": float(volume_expansion),
            "max_yaw_residual_degrees": float(max_yaw_residual_degrees),
            "max_volume_expansion": float(max_volume_expansion),
            "grounded": grounded,
            "floor_shift_room_y": float(shift_room_y),
            "raw_bottom_room_y": bottom_room_y,
            "nearest_wall_component_id": nearest_wall_id if near_wall else None,
            "nearest_wall_distance": (
                float(nearest_wall_distance)
                if nearest_wall_distance is not None
                else None
            ),
            "near_wall": near_wall,
            "raw_center_room": {
                "x": float(raw_center_room[0]),
                "y": float(raw_center_room[1]),
                "z": float(raw_center_room[2]),
            },
            "regularized": True,
        }
        output.append(output_component)

    return {
        "enabled": True,
        "components": output,
        "component_count": len(output),
        "snapped_count": snapped_count,
        "grounded_count": grounded_count,
        "max_yaw_residual_degrees": float(max_yaw_residual_degrees),
        "max_volume_expansion": float(max_volume_expansion),
        "ground_tolerance": float(ground_tolerance),
        "wall_tolerance": float(wall_tolerance),
    }


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    helper = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(normal, helper))) > 0.92:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(normal, u)
    v /= max(np.linalg.norm(v), 1e-9)
    return u, v


def _plane_component(
    component_id: str,
    points: np.ndarray,
    model: np.ndarray,
    scene_lower: np.ndarray,
    scene_upper: np.ndarray,
    scene_diagonal: float,
) -> dict[str, Any]:
    normal = np.asarray(model[:3], dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1e-9)
    u, v = _plane_basis(normal)
    origin = np.median(points, axis=0)
    rel = points - origin
    pu = rel @ u
    pv = rel @ v
    low_u, high_u = np.quantile(pu, [0.01, 0.99])
    low_v, high_v = np.quantile(pv, [0.01, 0.99])
    center = (
        origin
        + u * ((low_u + high_u) * 0.5)
        + v * ((low_v + high_v) * 0.5)
    )
    thickness = max(0.015, scene_diagonal * 0.007)
    extent = np.array([
        max(float(high_u - low_u), thickness),
        max(float(high_v - low_v), thickness),
        thickness,
    ])
    axes = np.column_stack([u, v, normal])

    abs_y = abs(float(normal[1]))
    y_span = max(float(scene_upper[1] - scene_lower[1]), 1e-6)
    y_ratio = float((center[1] - scene_lower[1]) / y_span)
    if abs_y >= 0.75:
        if y_ratio <= 0.18:
            kind = "floor"
        elif y_ratio >= 0.85:
            kind = "ceiling"
        else:
            kind = "horizontal_surface"
    elif abs_y <= 0.40:
        kind = "wall"
    else:
        kind = "sloped_surface"

    area = float(extent[0] * extent[1])
    return _component(
        component_id=component_id,
        kind=kind,
        center=center,
        axes=axes,
        extent=extent,
        point_count=len(points),
        source="plane",
        confidence=min(1.0, 0.55 + math.log10(max(len(points), 10)) / 10.0),
        metadata={
            "plane": [float(value) for value in model.tolist()],
            "area": area,
            "normal": [float(value) for value in normal.tolist()],
        },
    )


def _cluster_components(
    cloud: o3d.geometry.PointCloud,
    start_index: int,
    scene_diagonal: float,
    max_components: int = 24,
    room_frame: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if len(cloud.points) < 50:
        return []
    voxel = float(np.clip(scene_diagonal * 0.0075, 0.012, 0.025))
    reduced = cloud.voxel_down_sample(voxel)
    if len(reduced.points) < 30:
        return []

    eps = float(np.clip(scene_diagonal * 0.018, 0.03, 0.05))
    labels = np.asarray(
        reduced.cluster_dbscan(
            eps=eps,
            min_points=8,
            print_progress=False,
        )
    )
    if labels.size == 0 or labels.max(initial=-1) < 0:
        return []

    candidates: list[tuple[int, int, np.ndarray]] = []
    for label in range(int(labels.max()) + 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 20:
            continue
        candidates.append((len(indices), label, indices))
    candidates.sort(reverse=True)

    components: list[dict[str, Any]] = []
    points = np.asarray(reduced.points)
    world_to_room = (
        np.asarray(room_frame["world_to_room"], dtype=np.float64)
        if room_frame is not None
        else None
    )
    for _, label, indices in candidates[:max_components]:
        cluster_points = points[indices]
        if len(cluster_points) < 4:
            continue
        cluster_cloud = o3d.geometry.PointCloud()
        cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
        try:
            obb = cluster_cloud.get_oriented_bounding_box(robust=True)
        except RuntimeError:
            aabb = cluster_cloud.get_axis_aligned_bounding_box()
            center = np.asarray(aabb.get_center(), dtype=np.float64)
            extent = np.asarray(aabb.get_extent(), dtype=np.float64)
            axes = np.eye(3, dtype=np.float64)
        else:
            center = np.asarray(obb.center, dtype=np.float64)
            extent = np.asarray(obb.extent, dtype=np.float64)
            axes = np.asarray(obb.R, dtype=np.float64)

        min_extent = max(0.02, scene_diagonal * 0.006)
        extent = np.maximum(extent, min_extent)
        if float(np.linalg.norm(extent)) < scene_diagonal * 0.035:
            continue
        raw_volume = float(np.prod(np.maximum(extent, 1e-6)))
        metadata: dict[str, Any] = {
            "dbscan_label": int(label),
            "eps": eps,
            "voxel_size": voxel,
        }
        if world_to_room is not None:
            room_points = _transform_xyz(cluster_points, world_to_room)
            room_min = room_points.min(axis=0)
            room_max = room_points.max(axis=0)
            room_extent = np.maximum(room_max - room_min, 1e-6)
            room_volume = float(np.prod(room_extent))
            metadata["room_aligned_fit"] = {
                "min": room_min.tolist(),
                "max": room_max.tolist(),
                "extent": room_extent.tolist(),
                "volume": room_volume,
                "raw_obb_volume": raw_volume,
                "volume_expansion": room_volume / max(raw_volume, 1e-9),
            }

        components.append(
            _component(
                component_id=f"cluster-{start_index + len(components):03d}",
                kind="object_region",
                center=center,
                axes=axes,
                extent=extent,
                point_count=len(indices),
                source="cluster",
                confidence=min(0.9, 0.45 + len(indices) / max(len(points), 1)),
                metadata=metadata,
            )
        )
    return components


def _build_shell(
    cloud: o3d.geometry.PointCloud,
    output_path: Path,
    scene_diagonal: float,
    target_triangles: int,
) -> dict[str, Any]:
    voxel = float(np.clip(scene_diagonal * 0.0075, 0.012, 0.025))
    working = cloud.voxel_down_sample(voxel)
    if len(working.points) < 100:
        raise ValueError("Not enough points for low-poly shell")

    normal_radius = float(np.clip(scene_diagonal * 0.03, 0.04, 0.10))
    working.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=40,
        )
    )

    started = time.perf_counter()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        working,
        depth=8,
        scale=1.05,
        linear_fit=False,
        n_threads=4,
    )
    poisson_seconds = time.perf_counter() - started

    density = np.asarray(densities)
    if density.size:
        cutoff = float(np.quantile(density, 0.015))
        mesh.remove_vertices_by_mask(density < cutoff)

    points = np.asarray(working.points)
    lower, upper = _robust_bounds(points)
    pad = (upper - lower) * 0.05
    mesh = mesh.crop(
        o3d.geometry.AxisAlignedBoundingBox(
            lower - pad,
            upper + pad,
        )
    )
    raw_triangles = len(mesh.triangles)
    raw_vertices = len(mesh.vertices)

    simplify_started = time.perf_counter()
    if raw_triangles > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    simplify_seconds = time.perf_counter() - simplify_started
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )
    if not ok:
        raise RuntimeError("Failed to write low-poly shell")

    return {
        "path": str(output_path),
        "voxel_size": voxel,
        "normal_radius": normal_radius,
        "input_points": len(cloud.points),
        "working_points": len(working.points),
        "raw_vertices": raw_vertices,
        "raw_triangles": raw_triangles,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "has_vertex_colors": bool(mesh.has_vertex_colors()),
        "poisson_seconds": poisson_seconds,
        "simplify_seconds": simplify_seconds,
    }


def build_spatial_model(
    point_cloud_path: str | Path,
    output_dir: str | Path,
    scene_id: str | None = None,
    target_triangles: int = 20_000,
    max_planes: int = 8,
    max_cluster_components: int = 24,
    random_seed: int = 7,
) -> dict[str, Any]:
    source = Path(point_cloud_path)
    if not source.exists():
        raise ValueError(f"Point cloud not found: {source}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    try:
        o3d.utility.random.seed(int(random_seed))
    except AttributeError:
        pass
    cloud = o3d.io.read_point_cloud(str(source))
    if len(cloud.points) < 100:
        raise ValueError("Point cloud is too small for a spatial model")

    points = np.asarray(cloud.points)
    scene_lower, scene_upper = _robust_bounds(points)
    scene_diagonal = _scene_diagonal(points)

    component_voxel = float(np.clip(scene_diagonal * 0.0075, 0.012, 0.025))
    component_cloud = cloud.voxel_down_sample(component_voxel)
    remaining = component_cloud
    components: list[dict[str, Any]] = []
    plane_threshold = float(np.clip(scene_diagonal * 0.009, 0.012, 0.025))
    min_plane_points = max(250, int(len(component_cloud.points) * 0.02))

    plane_started = time.perf_counter()
    for plane_index in range(max_planes):
        if len(remaining.points) < min_plane_points * 2:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=plane_threshold,
            ransac_n=3,
            num_iterations=800,
        )
        if len(inliers) < min_plane_points:
            break
        inlier_cloud = remaining.select_by_index(inliers)
        component = _plane_component(
            component_id=f"plane-{plane_index:03d}",
            points=np.asarray(inlier_cloud.points),
            model=np.asarray(model, dtype=np.float64),
            scene_lower=scene_lower,
            scene_upper=scene_upper,
            scene_diagonal=scene_diagonal,
        )
        components.append(component)
        remaining = remaining.select_by_index(inliers, invert=True)
    plane_seconds = time.perf_counter() - plane_started

    room_frame = estimate_room_frame(
        plane_components=components,
        scene_points=points,
    )
    structural_layout = build_structural_layout(
        plane_components=components,
        room_frame=room_frame,
        scene_points=points,
        scene_diagonal=scene_diagonal,
    )

    cluster_started = time.perf_counter()
    components.extend(
        _cluster_components(
            remaining,
            start_index=0,
            scene_diagonal=scene_diagonal,
            max_components=max_cluster_components,
            room_frame=room_frame,
        )
    )
    cluster_seconds = time.perf_counter() - cluster_started

    object_layout = build_object_layout(
        components=components,
        room_frame=room_frame,
        structural_layout=structural_layout,
        scene_points=points,
        scene_diagonal=scene_diagonal,
    )

    shell_path = output / "shell.ply"
    shell = _build_shell(
        cloud=cloud,
        output_path=shell_path,
        scene_diagonal=scene_diagonal,
        target_triangles=target_triangles,
    )

    manifest = {
        "version": 1,
        "scene_id": scene_id,
        "source_point_cloud": str(source),
        "shell": {
            **shell,
            "url": "shell",
        },
        "components": components,
        "component_count": len(components),
        "room_frame": room_frame,
        "structural_layout": structural_layout,
        "object_layout": object_layout,
        "bounds": {
            "min": {
                "x": float(scene_lower[0]),
                "y": float(scene_lower[1]),
                "z": float(scene_lower[2]),
            },
            "max": {
                "x": float(scene_upper[0]),
                "y": float(scene_upper[1]),
                "z": float(scene_upper[2]),
            },
        },
        "metrics": {
            "scene_diagonal": scene_diagonal,
            "random_seed": int(random_seed),
            "component_voxel_size": component_voxel,
            "plane_distance_threshold": plane_threshold,
            "plane_count": sum(1 for c in components if c["source"] == "plane"),
            "cluster_count": sum(1 for c in components if c["source"] == "cluster"),
            "room_frame_confidence": room_frame["confidence"],
            "room_frame_snap_enabled": room_frame["snap_enabled"],
            "room_frame_residual_mean_degrees": room_frame["metrics"]["weighted_residual_mean_degrees"],
            "structural_component_count": structural_layout["component_count"],
            "structural_wall_count": structural_layout["wall_count"],
            "structural_footprint_closed": structural_layout["footprint"]["closed"],
            "object_component_count": object_layout["component_count"],
            "object_snapped_count": object_layout["snapped_count"],
            "object_grounded_count": object_layout["grounded_count"],
            "plane_seconds": plane_seconds,
            "cluster_seconds": cluster_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    manifest_path = output / "manifest.json"
    from .hybrid import build_hybrid
    try:
        manifest["hybrid"] = build_hybrid(
            source, shell_path, components, structural_layout, scene_diagonal, output)
    except (ValueError, RuntimeError) as exc:
        manifest["hybrid_error"] = str(exc)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
