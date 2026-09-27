from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageStat

from .models import BoundingBox3D, Observation, Vec3
from .projection import load_reconstruction, project_bbox_geometry
from .service import SpatialMemoryService
from .storage import SpatialMemoryDB
from .vision.grounding_dino import GroundingDinoDetector
from .vision.prompts import DEFAULT_INDOOR_LABELS


_KIND_BY_LABEL = {
    "desk": "furniture",
    "chair": "furniture",
    "shelf": "furniture",
    "monitor": "furniture",
    "box": "container",
}


@dataclass
class _ProjectedDetection:
    label: str
    score: float
    position: Vec3
    bbox3d: BoundingBox3D
    appearance: tuple[float, ...]
    frame_index: int
    image_path: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class _InstanceCluster:
    label: str
    members: list[_ProjectedDetection] = field(default_factory=list)

    @property
    def centroid(self) -> Vec3:
        if not self.members:
            return Vec3(x=0.0, y=0.0, z=0.0)
        weights = [max(member.score, 0.05) for member in self.members]
        total = sum(weights)
        return Vec3(
            x=sum(m.position.x * w for m, w in zip(self.members, weights)) / total,
            y=sum(m.position.y * w for m, w in zip(self.members, weights)) / total,
            z=sum(m.position.z * w for m, w in zip(self.members, weights)) / total,
        )

    @property
    def mean_size(self) -> Vec3:
        if not self.members:
            return Vec3(x=0.0, y=0.0, z=0.0)
        weights = [max(member.score, 0.05) for member in self.members]
        total = sum(weights)
        return Vec3(
            x=sum(m.bbox3d.size.x * w for m, w in zip(self.members, weights)) / total,
            y=sum(m.bbox3d.size.y * w for m, w in zip(self.members, weights)) / total,
            z=sum(m.bbox3d.size.z * w for m, w in zip(self.members, weights)) / total,
        )

    @property
    def bounding_box(self) -> BoundingBox3D:
        return BoundingBox3D(center=self.centroid, size=self.mean_size)

    @property
    def appearance(self) -> tuple[float, ...]:
        if not self.members:
            return ()
        length = len(self.members[0].appearance)
        if length == 0:
            return ()
        weights = [max(member.score, 0.05) for member in self.members]
        total = sum(weights)
        return tuple(
            sum(member.appearance[index] * weight
                for member, weight in zip(self.members, weights)) / total
            for index in range(length)
        )

    @property
    def frame_indices(self) -> set[int]:
        return {member.frame_index for member in self.members}

    def distance_to(self, point: Vec3) -> float:
        center = self.centroid
        return math.sqrt(
            (center.x - point.x) ** 2
            + (center.y - point.y) ** 2
            + (center.z - point.z) ** 2
        )

    def shape_distance(self, bbox3d: BoundingBox3D) -> float:
        current = self.mean_size
        a = (current.x, current.y, current.z)
        b = (bbox3d.size.x, bbox3d.size.y, bbox3d.size.z)
        values = []
        for left, right in zip(a, b):
            left = max(float(left), 0.03)
            right = max(float(right), 0.03)
            values.append(abs(math.log(left / right)))
        return sum(values) / len(values)

    def appearance_distance(self, descriptor: tuple[float, ...]) -> float:
        current = self.appearance
        if not current or not descriptor or len(current) != len(descriptor):
            return 0.0
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(current, descriptor))
            / len(current)
        )

    def match_cost(
        self,
        observation: _ProjectedDetection,
        radius: float,
        max_shape_distance: float,
        max_appearance_distance: float,
    ) -> float | None:
        if observation.frame_index in self.frame_indices:
            return None
        distance = self.distance_to(observation.position)
        if distance > radius:
            return None
        shape_distance = self.shape_distance(observation.bbox3d)
        if shape_distance > max_shape_distance:
            return None
        appearance_distance = self.appearance_distance(observation.appearance)
        if appearance_distance > max_appearance_distance:
            return None
        return (
            distance / max(radius, 1e-6)
            + 0.35 * shape_distance
            + 0.45 * appearance_distance
        )

    @property
    def position_spread(self) -> float:
        center = self.centroid
        if not self.members:
            return 0.0
        return max(
            math.sqrt(
                (member.position.x - center.x) ** 2
                + (member.position.y - center.y) ** 2
                + (member.position.z - center.z) ** 2
            )
            for member in self.members
        )


def _appearance_descriptor(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, ...]:
    x1, y1, x2, y2 = bbox_xyxy
    width, height = image.size
    left = max(0, min(width - 1, int(math.floor(min(x1, x2)))))
    top = max(0, min(height - 1, int(math.floor(min(y1, y2)))))
    right = max(left + 1, min(width, int(math.ceil(max(x1, x2)))))
    bottom = max(top + 1, min(height, int(math.ceil(max(y1, y2)))))
    crop = image.crop((left, top, right, bottom)).resize((16, 16))
    stats = ImageStat.Stat(crop)
    means = [float(value) / 255.0 for value in stats.mean[:3]]
    stddev = [float(value) / 255.0 for value in stats.stddev[:3]]
    return tuple(means + stddev)


class SceneAutoIndexer:
    def __init__(
        self,
        db: SpatialMemoryDB,
        memory: SpatialMemoryService,
        detector: GroundingDinoDetector,
    ):
        self.db = db
        self.memory = memory
        self.detector = detector

    @staticmethod
    def _suppress_nested_same_label_detections(
        detections,
        containment_threshold: float = 0.85,
    ):
        def area(bbox) -> float:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        def containment(inner, outer) -> float:
            ix1, iy1, ix2, iy2 = [float(value) for value in inner]
            ox1, oy1, ox2, oy2 = [float(value) for value in outer]
            inter_x1 = max(ix1, ox1)
            inter_y1 = max(iy1, oy1)
            inter_x2 = min(ix2, ox2)
            inter_y2 = min(iy2, oy2)
            intersection = (
                max(0.0, inter_x2 - inter_x1)
                * max(0.0, inter_y2 - inter_y1)
            )
            return intersection / max(area(inner), 1e-9)

        kept = []
        suppressed = 0
        for detection in sorted(
            detections,
            key=lambda item: item.score,
            reverse=True,
        ):
            duplicate = False
            candidate_area = area(detection.bbox_xyxy)
            for existing in kept:
                if existing.label.lower() != detection.label.lower():
                    continue
                existing_area = area(existing.bbox_xyxy)
                if candidate_area > existing_area * 1.05:
                    continue
                if (
                    containment(detection.bbox_xyxy, existing.bbox_xyxy)
                    >= containment_threshold
                ):
                    duplicate = True
                    break
            if duplicate:
                suppressed += 1
            else:
                kept.append(detection)
        return kept, suppressed

    @staticmethod
    def _cluster_observations(
        observations: list[_ProjectedDetection],
        radius: float,
        max_shape_distance: float = 0.95,
        max_appearance_distance: float = 0.32,
    ) -> list[_InstanceCluster]:
        clusters: list[_InstanceCluster] = []
        ordered = sorted(
            observations,
            key=lambda item: (
                item.label.lower(),
                item.frame_index,
                -item.score,
            ),
        )
        for observation in ordered:
            candidates: list[tuple[float, _InstanceCluster]] = []
            for cluster in clusters:
                if cluster.label.lower() != observation.label.lower():
                    continue
                cost = cluster.match_cost(
                    observation,
                    radius=radius,
                    max_shape_distance=max_shape_distance,
                    max_appearance_distance=max_appearance_distance,
                )
                if cost is not None:
                    candidates.append((cost, cluster))

            if candidates:
                cluster = min(candidates, key=lambda item: item[0])[1]
            else:
                cluster = _InstanceCluster(label=observation.label)
                clusters.append(cluster)
            cluster.members.append(observation)

        clusters.sort(key=lambda cluster: (
            cluster.label.lower(),
            round(cluster.centroid.x, 4),
            round(cluster.centroid.y, 4),
            round(cluster.centroid.z, 4),
            tuple(round(value, 3) for value in cluster.appearance),
        ))
        return clusters

    async def run(
        self,
        scene_id: str,
        labels: list[str] | None = None,
        frame_step: int = 5,
        max_frames: int = 12,
        max_detections_per_frame: int = 20,
        threshold: float = 0.35,
        text_threshold: float = 0.25,
        cluster_radius: float = 0.45,
        max_shape_distance: float = 0.95,
        max_appearance_distance: float = 0.32,
        singleton_min_confidence: float = 0.55,
    ) -> dict:
        scene = self.db.get_scene(scene_id)
        if scene is None:
            raise ValueError("Scene not found")
        if not scene.frames_dir or not scene.reconstruction_npz:
            raise ValueError("Scene requires frames and reconstruction data")

        frames_dir = Path(scene.frames_dir)
        frames = sorted(
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not frames:
            raise ValueError("Scene has no frames")

        labels = labels or list(DEFAULT_INDOOR_LABELS)
        frame_step = max(frame_step, 1)
        cluster_radius = max(float(cluster_radius), 0.05)
        singleton_min_confidence = max(
            float(singleton_min_confidence),
            float(threshold),
        )
        selected = list(range(0, len(frames), frame_step))[:max_frames]
        reconstruction = load_reconstruction(scene.reconstruction_npz)

        projected: list[_ProjectedDetection] = []
        detections_seen = 0
        detections_suppressed = 0
        projection_failures = 0

        for frame_index in selected:
            image = Image.open(frames[frame_index]).convert("RGB")
            detections = self.detector.detect(
                image,
                labels=labels,
                threshold=threshold,
                text_threshold=text_threshold,
            )
            detections_seen += len(detections)
            detections, suppressed = self._suppress_nested_same_label_detections(
                detections
            )
            detections_suppressed += suppressed
            detections = sorted(
                detections,
                key=lambda item: item.score,
                reverse=True,
            )[:max_detections_per_frame]
            for detection in detections:
                bbox = tuple(float(v) for v in detection.bbox_xyxy)
                try:
                    position, bbox3d = project_bbox_geometry(
                        reconstruction,
                        frame_index,
                        tuple(int(round(v)) for v in bbox),
                        source_size=(image.width, image.height),
                    )
                except (IndexError, KeyError, ValueError):
                    projection_failures += 1
                    continue

                projected.append(
                    _ProjectedDetection(
                        label=detection.label,
                        score=float(detection.score),
                        position=position,
                        bbox3d=bbox3d,
                        appearance=_appearance_descriptor(image, bbox),
                        frame_index=frame_index,
                        image_path=str(frames[frame_index]),
                        bbox_xyxy=bbox,
                    )
                )

        raw_clusters = self._cluster_observations(
            projected,
            radius=cluster_radius,
            max_shape_distance=max_shape_distance,
            max_appearance_distance=max_appearance_distance,
        )
        weak_clusters = [
            cluster
            for cluster in raw_clusters
            if len(cluster.members) == 1
            and cluster.members[0].score < singleton_min_confidence
        ]
        clusters = [
            cluster
            for cluster in raw_clusters
            if cluster not in weak_clusters
        ]

        label_counts: dict[str, int] = {}
        for cluster in clusters:
            key = cluster.label.lower()
            label_counts[key] = label_counts.get(key, 0) + 1

        entity_ids: set[str] = set()
        multi_observation_clusters = 0
        label_seen_index: dict[str, int] = {}

        for cluster in clusters:
            if not cluster.members:
                continue

            key = cluster.label.lower()
            label_seen_index[key] = label_seen_index.get(key, 0) + 1
            instance_index = label_seen_index[key]
            members = cluster.members
            if len(members) > 1:
                multi_observation_clusters += 1

            best = max(members, key=lambda item: item.score)
            scores = [float(member.score) for member in members]
            centroid = cluster.centroid
            appearance = cluster.appearance
            mean_size = cluster.mean_size
            signature_cell = 0.10
            instance_signature = (
                f"{key}#{instance_index}@"
                f"{round(centroid.x / signature_cell)}:"
                f"{round(centroid.y / signature_cell)}:"
                f"{round(centroid.z / signature_cell)}"
            )
            metadata = {
                "observation_count": len(members),
                "frame_indices": sorted({m.frame_index for m in members}),
                "vision_scores": scores,
                "vision_score_mean": sum(scores) / len(scores),
                "vision_score_max": max(scores),
                "detected_labels": sorted({m.label for m in members}),
                "image_paths": sorted({m.image_path for m in members}),
                "instance_index": instance_index,
                "instance_count_for_label": label_counts[key],
                "instance_signature": instance_signature,
                "position_spread_m": cluster.position_spread,
                "bbox3d_size_mean": {
                    "x": mean_size.x,
                    "y": mean_size.y,
                    "z": mean_size.z,
                },
                "appearance_descriptor_mean": [
                    round(value, 6) for value in appearance
                ],
                "fusion_thresholds": {
                    "cluster_radius": cluster_radius,
                    "max_shape_distance": max_shape_distance,
                    "max_appearance_distance": max_appearance_distance,
                    "singleton_min_confidence": singleton_min_confidence,
                },
                "auto_index": True,
            }

            observation = Observation(
                scene_id=scene_id,
                position=centroid,
                bbox=cluster.bounding_box,
                kind=_KIND_BY_LABEL.get(key, "object"),
                detected_label=cluster.label,
                detection_confidence=max(scores),
                image_path=best.image_path,
                frame_index=best.frame_index,
                detection_bbox_xyxy=best.bbox_xyxy,
                metadata=metadata,
            )
            entity = await self.memory.ingest_observation(
                observation,
                merge_radius=max(cluster_radius * 0.60, 0.10),
            )

            if (
                label_counts[key] > 1
                and entity.source == "vision"
                and not entity.metadata.get("human_explicit_name")
            ):
                base_name = str(
                    entity.metadata.get("instance_base_name")
                    or entity.name
                )
                indexed_name = f"{base_name} {instance_index}"
                aliases = set(entity.aliases)
                aliases.update({
                    base_name,
                    cluster.label,
                    f"{cluster.label} {instance_index}",
                })
                entity_metadata = dict(entity.metadata)
                entity_metadata["instance_base_name"] = base_name
                entity_metadata["instance_display_name"] = indexed_name
                entity = self.db.upsert_entity(entity.model_copy(update={
                    "name": indexed_name,
                    "aliases": sorted(aliases),
                    "metadata": entity_metadata,
                }))

            entity_ids.add(entity.id)

        return {
            "scene_id": scene_id,
            "frames_processed": len(selected),
            "detections_seen": detections_seen,
            "detections_suppressed": detections_suppressed,
            "projection_failures": projection_failures,
            "projected_observations": len(projected),
            "clusters_built_raw": len(raw_clusters),
            "weak_singleton_clusters_skipped": len(weak_clusters),
            "clusters_built": len(clusters),
            "multi_observation_clusters": multi_observation_clusters,
            "entity_count": len(entity_ids),
            "entity_ids": sorted(entity_ids),
            "cluster_radius": cluster_radius,
            "max_shape_distance": max_shape_distance,
            "max_appearance_distance": max_appearance_distance,
            "singleton_min_confidence": singleton_min_confidence,
            "labels": labels,
            "vision_metrics": self.detector.last_metrics,
        }
