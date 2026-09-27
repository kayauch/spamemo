import asyncio
from pathlib import Path

from spatial_memory.llm.rules import RuleBasedInterpreter
from spatial_memory.models import Scene, SearchHit, SpatialEntity, Vec3
from spatial_memory.service import SpatialMemoryService
from spatial_memory.storage import SpatialMemoryDB


class FakeSemantic:
    def search(self, query, entities, limit=5):
        by_name = {entity.name: entity for entity in entities}
        return [
            SearchHit(
                entity=by_name["父の眼鏡"],
                score=0.82,
                reasons=["semantic"],
            ),
            SearchHit(
                entity=by_name["共有HDMIケーブル"],
                score=0.25,
                reasons=["semantic"],
            ),
        ][:limit]


class BrokenSemantic:
    def search(self, query, entities, limit=5):
        raise RuntimeError("embedding server unavailable")


def seed(db: SpatialMemoryDB) -> None:
    db.upsert_scene(Scene(id="room", name="Room"))
    db.upsert_entity(SpatialEntity(
        id="glasses",
        scene_id="room",
        name="父の眼鏡",
        position=Vec3(x=1, y=1, z=1),
        tags=["父", "眼鏡"],
    ))
    db.upsert_entity(SpatialEntity(
        id="hdmi",
        scene_id="room",
        name="共有HDMIケーブル",
        position=Vec3(x=2, y=1, z=3),
        tags=["HDMI", "ケーブル"],
    ))


def test_semantic_retrieval_handles_paraphrase(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    seed(db)
    service = SpatialMemoryService(
        db,
        RuleBasedInterpreter(),
        semantic=FakeSemantic(),
    )
    result = asyncio.run(
        service.query("room", "お父さんがいつも使っているメガネってどこだっけ？")
    )
    assert result.target is not None
    assert result.target.id == "glasses"
    assert "semantic" in result.hits[0].reasons


def test_semantic_failure_falls_back_to_lexical(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    seed(db)
    service = SpatialMemoryService(
        db,
        RuleBasedInterpreter(),
        semantic=BrokenSemantic(),
    )
    result = asyncio.run(service.query("room", "共有HDMIケーブルどこ？"))
    assert result.target is not None
    assert result.target.id == "hdmi"


class AmbiguousSemantic:
    def search(self, query, entities, limit=5):
        return [
            SearchHit(entity=entities[0], score=0.76, reasons=["semantic"]),
            SearchHit(entity=entities[1], score=0.755, reasons=["semantic"]),
        ][:limit]


def test_ambiguous_semantic_result_is_rejected(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    seed(db)
    service = SpatialMemoryService(
        db,
        RuleBasedInterpreter(),
        semantic=AmbiguousSemantic(),
    )
    result = asyncio.run(service.query("room", "ヘリコプターどこ？"))
    assert result.target is None
    assert result.hits == []
