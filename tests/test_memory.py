import asyncio
from pathlib import Path

from spatial_memory.llm.rules import RuleBasedInterpreter
from spatial_memory.models import Observation, Scene, SpatialEntity, Vec3
from spatial_memory.service import SpatialMemoryService
from spatial_memory.storage import SpatialMemoryDB


def test_query_finds_expected_entity(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(id="room", name="Room"))
    db.upsert_entity(SpatialEntity(
        id="g1",
        scene_id="room",
        name="父の眼鏡",
        kind="object",
        position=Vec3(x=1, y=2, z=3),
        aliases=["お父さんのメガネ"],
        tags=["父", "眼鏡"],
    ))
    service = SpatialMemoryService(db, RuleBasedInterpreter())
    result = asyncio.run(service.query("room", "父の眼鏡どこ？"))
    assert result.target is not None
    assert result.target.id == "g1"
    assert result.target.position.x == 1


def test_observation_becomes_entity(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(id="room", name="Room"))
    service = SpatialMemoryService(db, RuleBasedInterpreter())
    observation = Observation(
        scene_id="room",
        position=Vec3(x=0.5, y=1.0, z=-2.0),
        transcript="これは父の眼鏡です",
        detected_label="glasses",
        detection_confidence=0.91,
    )
    entity = asyncio.run(service.ingest_observation(observation))
    assert entity.name == "父の眼鏡"
    assert entity.source == "fused"
    assert entity.metadata["owner"] == "父"
    assert entity.position.z == -2.0


def test_nearby_same_named_observation_merges(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(id="room", name="Room"))
    service = SpatialMemoryService(db, RuleBasedInterpreter(), merge_radius=0.75)

    first = Observation(
        scene_id="room",
        position=Vec3(x=1.0, y=0.8, z=2.0),
        transcript="これは予備の鍵です",
        detected_label="key",
        detection_confidence=0.80,
    )
    second = Observation(
        scene_id="room",
        position=Vec3(x=1.2, y=0.8, z=2.1),
        transcript="これは予備の鍵です",
        detected_label="key",
        detection_confidence=0.95,
    )
    a = asyncio.run(service.ingest_observation(first))
    b = asyncio.run(service.ingest_observation(second))

    assert a.id == b.id
    assert len(db.list_entities("room")) == 1
    assert b.position.x == 1.2
    assert b.confidence == 0.95


def test_container_phrase_extracts_hidden_item(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(id="room", name="Room"))
    parent = db.upsert_entity(SpatialEntity(
        id="box",
        scene_id="room",
        name="青い箱",
        kind="container",
        position=Vec3(x=2.0, y=1.0, z=-3.0),
    ))
    service = SpatialMemoryService(db, RuleBasedInterpreter())
    observation = Observation(
        scene_id="room",
        position=parent.position,
        parent_id=parent.id,
        relation_to_parent="inside",
        transcript="この箱に冬服を入れた",
    )
    entity = asyncio.run(service.ingest_observation(observation))

    assert entity.name == "冬服"
    assert entity.parent_id == "box"
    assert entity.relation_to_parent == "inside"
    assert entity.position == parent.position

    result = asyncio.run(service.query("room", "冬服どこ？"))
    assert result.target is not None
    assert result.target.id == entity.id



def test_vision_update_does_not_downgrade_fused_source(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(id="room", name="Room"))
    service = SpatialMemoryService(db, RuleBasedInterpreter(), merge_radius=0.75)

    fused = Observation(
        scene_id="room",
        position=Vec3(x=0.0, y=0.0, z=2.0),
        transcript="これはロボットです",
        detected_label="robot",
        detection_confidence=0.85,
    )
    first = asyncio.run(service.ingest_observation(fused))
    assert first.source == "fused"

    vision = Observation(
        scene_id="room",
        position=Vec3(x=0.1, y=0.0, z=2.0),
        detected_label="robot",
        detection_confidence=0.95,
        metadata={
            "auto_index": True,
            "observation_count": 3,
            "frame_indices": [0, 1, 2],
        },
    )
    updated = asyncio.run(
        service.ingest_observation(vision, merge_radius=0.5)
    )
    assert updated.id == first.id
    assert updated.source == "fused"
    assert updated.metadata["observation_count"] == 4



def test_human_explicit_name_renames_existing_visual_entity(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(id="room", name="Room"))
    service = SpatialMemoryService(db, RuleBasedInterpreter(), merge_radius=0.75)

    visual = Observation(
        scene_id="room",
        position=Vec3(x=0.0, y=0.0, z=2.0),
        detected_label="robot",
        detection_confidence=0.9,
    )
    first = asyncio.run(service.ingest_observation(visual))
    assert first.name == "robot"
    assert first.source == "vision"

    named = Observation(
        scene_id="room",
        position=Vec3(x=0.05, y=0.0, z=2.0),
        transcript="これはロボットです",
        detected_label="robot",
        detection_confidence=0.9,
    )
    updated = asyncio.run(service.ingest_observation(named))

    assert updated.id == first.id
    assert updated.name == "ロボット"
    assert "robot" in updated.aliases
    assert updated.source == "fused"
    assert updated.metadata["human_explicit_name"] == "ロボット"
