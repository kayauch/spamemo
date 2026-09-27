import asyncio
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_memory.auto_index import SceneAutoIndexer
from spatial_memory.llm.rules import RuleBasedInterpreter
from spatial_memory.models import Scene
from spatial_memory.service import SpatialMemoryService
from spatial_memory.storage import SpatialMemoryDB
from spatial_memory.vision.models import Detection2D


class FakeDetector:
    def __init__(self) -> None:
        self.last_metrics = {"model": "fake"}

    def detect(self, image, labels, threshold=0.35, text_threshold=0.25):
        return [
            Detection2D(
                label="key",
                score=0.91,
                bbox_xyxy=(20, 20, 44, 44),
                image_width=image.width,
                image_height=image.height,
            )
        ]


def test_auto_index_creates_localized_entity(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (64, 64), "white").save(frames / "000000.jpg")

    depth = np.full((1, 64, 64), 2.0, dtype=np.float32)
    intrinsics = np.array([[
        [32.0, 0.0, 32.0],
        [0.0, 32.0, 32.0],
        [0.0, 0.0, 1.0],
    ]], dtype=np.float32)
    extrinsics = np.array([[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]], dtype=np.float32)
    reconstruction = tmp_path / "reconstruction.npz"
    np.savez(reconstruction, depth=depth, intrinsics=intrinsics, extrinsics=extrinsics)

    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(
        id="room",
        name="Room",
        frames_dir=str(frames),
        reconstruction_npz=str(reconstruction),
        frame_count=1,
    ))
    memory = SpatialMemoryService(db, RuleBasedInterpreter())
    indexer = SceneAutoIndexer(db, memory, FakeDetector())

    result = asyncio.run(indexer.run(
        "room",
        labels=["key"],
        frame_step=1,
        max_frames=1,
    ))

    entities = db.list_entities("room")
    assert result["entity_count"] == 1
    assert len(entities) == 1
    assert entities[0].name == "鍵"
    assert "key" in entities[0].aliases
    assert entities[0].source == "vision"



class SameObjectDetector:
    def __init__(self) -> None:
        self.last_metrics = {"model": "same-object"}

    def detect(self, image, labels, threshold=0.35, text_threshold=0.25):
        return [
            Detection2D(
                label="chair",
                score=0.90,
                bbox_xyxy=(20, 20, 44, 44),
                image_width=image.width,
                image_height=image.height,
            )
        ]


class TwoObjectDetector:
    def __init__(self) -> None:
        self.last_metrics = {"model": "two-object"}

    def detect(self, image, labels, threshold=0.35, text_threshold=0.25):
        return [
            Detection2D(
                label="chair",
                score=0.92,
                bbox_xyxy=(5, 20, 21, 44),
                image_width=image.width,
                image_height=image.height,
            ),
            Detection2D(
                label="chair",
                score=0.89,
                bbox_xyxy=(43, 20, 59, 44),
                image_width=image.width,
                image_height=image.height,
            ),
        ]


def make_static_scene(tmp_path: Path, frame_count: int = 3) -> tuple[SpatialMemoryDB, Path]:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(frame_count):
        Image.new("RGB", (64, 64), "white").save(frames / f"{index:06d}.jpg")

    depth = np.full((frame_count, 64, 64), 2.0, dtype=np.float32)
    intrinsics = np.repeat(
        np.array([[
            [32.0, 0.0, 32.0],
            [0.0, 32.0, 32.0],
            [0.0, 0.0, 1.0],
        ]], dtype=np.float32),
        frame_count,
        axis=0,
    )
    extrinsics = np.repeat(
        np.array([[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]], dtype=np.float32),
        frame_count,
        axis=0,
    )
    reconstruction = tmp_path / "reconstruction.npz"
    np.savez(
        reconstruction,
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )

    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(
        id="cluster-room",
        name="Cluster Room",
        frames_dir=str(frames),
        reconstruction_npz=str(reconstruction),
        frame_count=frame_count,
    ))
    return db, frames


def test_auto_index_fuses_same_instance_across_frames(tmp_path: Path) -> None:
    db, _ = make_static_scene(tmp_path, frame_count=3)
    memory = SpatialMemoryService(db, RuleBasedInterpreter())
    indexer = SceneAutoIndexer(db, memory, SameObjectDetector())

    result = asyncio.run(indexer.run(
        "cluster-room",
        labels=["chair"],
        frame_step=1,
        max_frames=3,
        cluster_radius=0.45,
    ))

    entities = db.list_entities("cluster-room")
    assert result["detections_seen"] == 3
    assert result["clusters_built"] == 1
    assert result["multi_observation_clusters"] == 1
    assert len(entities) == 1
    assert entities[0].name == "椅子"
    assert entities[0].metadata["observation_count"] == 3
    assert entities[0].metadata["frame_indices"] == [0, 1, 2]


def test_auto_index_keeps_spatially_separate_same_label_instances(tmp_path: Path) -> None:
    db, _ = make_static_scene(tmp_path, frame_count=1)
    memory = SpatialMemoryService(db, RuleBasedInterpreter())
    indexer = SceneAutoIndexer(db, memory, TwoObjectDetector())

    result = asyncio.run(indexer.run(
        "cluster-room",
        labels=["chair"],
        frame_step=1,
        max_frames=1,
        cluster_radius=0.45,
    ))

    entities = db.list_entities("cluster-room")
    assert result["detections_seen"] == 2
    assert result["clusters_built"] == 2
    assert len(entities) == 2
    assert sorted(entity.name for entity in entities) == ["椅子 1", "椅子 2"]
    assert all("椅子" in entity.aliases for entity in entities)
    assert all(entity.bbox is not None for entity in entities)
    xs = sorted(entity.position.x for entity in entities)
    assert xs[1] - xs[0] > 1.0
    assert sorted(entity.metadata["instance_index"] for entity in entities) == [1, 2]
    assert all(entity.metadata["instance_count_for_label"] == 2 for entity in entities)
    assert len({entity.metadata["instance_signature"] for entity in entities}) == 2


class CloseTwoObjectDetector:
    def __init__(self) -> None:
        self.last_metrics = {"model": "close-two-object"}

    def detect(self, image, labels, threshold=0.35, text_threshold=0.25):
        return [
            Detection2D(
                label="chair",
                score=0.93,
                bbox_xyxy=(20, 20, 36, 44),
                image_width=image.width,
                image_height=image.height,
            ),
            Detection2D(
                label="chair",
                score=0.90,
                bbox_xyxy=(28, 20, 44, 44),
                image_width=image.width,
                image_height=image.height,
            ),
        ]


def test_auto_index_tracks_close_same_label_instances_across_frames(tmp_path: Path) -> None:
    db, _ = make_static_scene(tmp_path, frame_count=3)
    memory = SpatialMemoryService(db, RuleBasedInterpreter())
    indexer = SceneAutoIndexer(db, memory, CloseTwoObjectDetector())

    first = asyncio.run(indexer.run(
        "cluster-room",
        labels=["chair"],
        frame_step=1,
        max_frames=3,
        cluster_radius=0.90,
    ))

    entities = sorted(db.list_entities("cluster-room"), key=lambda entity: entity.name)
    assert first["detections_seen"] == 6
    assert first["clusters_built"] == 2
    assert first["multi_observation_clusters"] == 2
    assert len(entities) == 2
    assert [entity.name for entity in entities] == ["椅子 1", "椅子 2"]
    assert [entity.metadata["observation_count"] for entity in entities] == [3, 3]
    assert all(entity.metadata["frame_indices"] == [0, 1, 2] for entity in entities)
    assert all(entity.bbox is not None for entity in entities)
    assert all("bbox3d_size_mean" in entity.metadata for entity in entities)
    assert all("appearance_descriptor_mean" in entity.metadata for entity in entities)
    assert len({entity.metadata["instance_signature"] for entity in entities}) == 2

    second = asyncio.run(indexer.run(
        "cluster-room",
        labels=["chair"],
        frame_step=1,
        max_frames=3,
        cluster_radius=0.90,
    ))

    rerun = sorted(db.list_entities("cluster-room"), key=lambda entity: entity.name)
    assert second["clusters_built"] == 2
    assert len(rerun) == 2
    assert [entity.name for entity in rerun] == ["椅子 1", "椅子 2"]
    assert [entity.metadata["observation_count"] for entity in rerun] == [6, 6]


class OneBoxDetector:
    def __init__(self) -> None:
        self.last_metrics = {"model": "appearance-box"}

    def detect(self, image, labels, threshold=0.35, text_threshold=0.25):
        return [
            Detection2D(
                label="box",
                score=0.90,
                bbox_xyxy=(16, 16, 48, 48),
                image_width=image.width,
                image_height=image.height,
            )
        ]


def test_auto_index_appearance_can_separate_same_position_instances(tmp_path: Path) -> None:
    frames = tmp_path / "appearance-frames"
    frames.mkdir()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(frames / "000000.jpg")
    Image.new("RGB", (64, 64), (0, 0, 255)).save(frames / "000001.jpg")

    depth = np.full((2, 64, 64), 2.0, dtype=np.float32)
    intrinsics = np.repeat(
        np.array([[
            [32.0, 0.0, 32.0],
            [0.0, 32.0, 32.0],
            [0.0, 0.0, 1.0],
        ]], dtype=np.float32),
        2,
        axis=0,
    )
    extrinsics = np.repeat(
        np.array([[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]], dtype=np.float32),
        2,
        axis=0,
    )
    reconstruction = tmp_path / "appearance.npz"
    np.savez(
        reconstruction,
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )

    db = SpatialMemoryDB(tmp_path / "appearance.db")
    db.upsert_scene(Scene(
        id="appearance-room",
        name="Appearance Room",
        frames_dir=str(frames),
        reconstruction_npz=str(reconstruction),
        frame_count=2,
    ))
    memory = SpatialMemoryService(db, RuleBasedInterpreter())
    indexer = SceneAutoIndexer(db, memory, OneBoxDetector())

    result = asyncio.run(indexer.run(
        "appearance-room",
        labels=["box"],
        frame_step=1,
        max_frames=2,
        cluster_radius=0.50,
        max_appearance_distance=0.20,
    ))

    entities = db.list_entities("appearance-room")
    assert result["clusters_built"] == 2
    assert len(entities) == 2
    assert sorted(entity.name for entity in entities) == ["箱 1", "箱 2"]
    descriptors = [
        tuple(entity.metadata["appearance_descriptor_mean"])
        for entity in entities
    ]
    assert descriptors[0] != descriptors[1]


class NestedDuplicateDetector:
    def __init__(self) -> None:
        self.last_metrics = {"model": "nested-duplicate"}

    def detect(self, image, labels, threshold=0.35, text_threshold=0.25):
        return [
            Detection2D(
                label="robot",
                score=0.92,
                bbox_xyxy=(8, 8, 56, 56),
                image_width=image.width,
                image_height=image.height,
            ),
            Detection2D(
                label="robot",
                score=0.31,
                bbox_xyxy=(20, 16, 30, 24),
                image_width=image.width,
                image_height=image.height,
            ),
            Detection2D(
                label="robot",
                score=0.27,
                bbox_xyxy=(35, 17, 45, 25),
                image_width=image.width,
                image_height=image.height,
            ),
        ]


def test_auto_index_suppresses_nested_same_label_duplicates(tmp_path: Path) -> None:
    db, _ = make_static_scene(tmp_path, frame_count=1)
    memory = SpatialMemoryService(db, RuleBasedInterpreter())
    indexer = SceneAutoIndexer(db, memory, NestedDuplicateDetector())

    result = asyncio.run(indexer.run(
        "cluster-room",
        labels=["robot"],
        frame_step=1,
        max_frames=1,
        threshold=0.20,
    ))

    entities = db.list_entities("cluster-room")
    assert result["detections_seen"] == 3
    assert result["detections_suppressed"] == 2
    assert result["projected_observations"] == 1
    assert result["clusters_built"] == 1
    assert len(entities) == 1
    assert entities[0].metadata["observation_count"] == 1
    assert len(entities[0].metadata["vision_scores"]) == 1


class WeakSingletonDetector:
    def __init__(self) -> None:
        self.last_metrics = {"model": "weak-singleton"}

    def detect(self, image, labels, threshold=0.35, text_threshold=0.25):
        return [
            Detection2D(
                label="robot",
                score=0.30,
                bbox_xyxy=(10, 10, 30, 30),
                image_width=image.width,
                image_height=image.height,
            )
        ]


def test_auto_index_skips_weak_singleton_cluster(tmp_path: Path) -> None:
    db, _ = make_static_scene(tmp_path, frame_count=1)
    memory = SpatialMemoryService(db, RuleBasedInterpreter())
    indexer = SceneAutoIndexer(db, memory, WeakSingletonDetector())

    result = asyncio.run(indexer.run(
        "cluster-room",
        labels=["robot"],
        frame_step=1,
        max_frames=1,
        threshold=0.25,
        singleton_min_confidence=0.55,
    ))

    assert result["detections_seen"] == 1
    assert result["clusters_built_raw"] == 1
    assert result["weak_singleton_clusters_skipped"] == 1
    assert result["clusters_built"] == 0
    assert result["entity_count"] == 0
    assert db.list_entities("cluster-room") == []
