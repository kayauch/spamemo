from __future__ import annotations

import argparse
import json
import os

from spatial_memory.models import NavEdge, NavNode, Scene, SpatialEntity, Vec3
from spatial_memory.storage import SpatialMemoryDB


DEMO_SCENE_ID = "00000000-0000-4000-8000-000000000001"
SOURCE_SCENE_ID = "relocalization-smoke"


def clear_demo_scene(db: SpatialMemoryDB, scene_id: str) -> None:
    with db.connect() as conn:
        for table in ("entities", "observations", "nav_nodes", "nav_edges"):
            conn.execute(f"DELETE FROM {table} WHERE scene_id=?", (scene_id,))
        conn.execute("DELETE FROM scenes WHERE id=?", (scene_id,))


def build_demo_scene(db: SpatialMemoryDB) -> Scene:
    source = db.get_scene(SOURCE_SCENE_ID)
    lowpoly = db.get_scene("lowpoly-smoke")
    if source is None:
        raise RuntimeError(
            "relocalization-smoke is missing; run the fixture preparation first"
        )

    scene = Scene(
        id=DEMO_SCENE_ID,
        name="Hackathon Demo Room",
        description="Deterministic fallback scene for the 3-5 minute demo path",
        point_cloud_path=source.point_cloud_path,
        spatial_model_path=(
            lowpoly.spatial_model_path
            if lowpoly and lowpoly.spatial_model_path
            else source.spatial_model_path
        ),
        reconstruction_npz=source.reconstruction_npz,
        frames_dir=source.frames_dir,
        source_video_path=source.source_video_path,
        reconstruction_model=source.reconstruction_model or "fixture",
        frame_count=source.frame_count,
    )
    return db.upsert_scene(scene)


def seed_entities(db: SpatialMemoryDB, scene: Scene) -> list[SpatialEntity]:
    box = SpatialEntity(
        id="demo-box",
        scene_id=scene.id,
        name="デモ用の箱",
        kind="container",
        position=Vec3(x=0.10, y=0.00, z=1.10),
        tags=["箱", "box", "デモ"],
        aliases=["箱"],
        source="human",
        confidence=1.0,
    )
    entities = [
        SpatialEntity(
            id="demo-robot",
            scene_id=scene.id,
            name="ロボット",
            kind="object",
            position=Vec3(x=-0.0516, y=-0.0548, z=1.0295),
            tags=["robot", "ロボット"],
            aliases=["robot"],
            description="デモで探すロボット",
            source="human",
            confidence=1.0,
        ),
        box,
        SpatialEntity(
            id="demo-hidden-cable",
            scene_id=scene.id,
            name="充電ケーブル",
            kind="object",
            position=box.position,
            parent_id=box.id,
            relation_to_parent="inside",
            tags=["ケーブル", "充電", "cable"],
            description="デモ用の箱に入れた充電ケーブル",
            source="human",
            confidence=1.0,
        ),
        SpatialEntity(
            id="demo-glasses",
            scene_id=scene.id,
            name="父の眼鏡",
            kind="object",
            position=Vec3(x=0.00, y=-0.08, z=0.90),
            tags=["眼鏡", "父", "eyeglasses"],
            aliases=["お父さんのメガネ"],
            metadata={"owner": "父"},
            source="human",
            confidence=1.0,
        ),
    ]
    for entity in entities:
        db.upsert_entity(entity)
    return entities


def seed_navigation(db: SpatialMemoryDB, scene: Scene) -> None:
    nodes = [
        NavNode(
            id="demo-entrance",
            scene_id=scene.id,
            name="入口",
            position=Vec3(x=0.0, y=0.0, z=1.45),
        ),
        NavNode(
            id="demo-center",
            scene_id=scene.id,
            name="中央",
            position=Vec3(x=0.0, y=0.0, z=1.0),
        ),
        NavNode(
            id="demo-target",
            scene_id=scene.id,
            name="ロボット付近",
            position=Vec3(x=-0.05, y=0.0, z=1.03),
        ),
    ]
    for node in nodes:
        db.upsert_nav_node(node)
    for source, target in (
        ("demo-entrance", "demo-center"),
        ("demo-center", "demo-target"),
    ):
        db.upsert_nav_edge(
            NavEdge(
                scene_id=scene.id,
                source_id=source,
                target_id=target,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete and recreate only the dedicated hackathon demo scene",
    )
    args = parser.parse_args()

    db_path = os.getenv("SPATIAL_DB", "/results/spatial-memory-v2.db")
    db = SpatialMemoryDB(db_path)
    if args.reset:
        clear_demo_scene(db, DEMO_SCENE_ID)

    scene = build_demo_scene(db)
    entities = seed_entities(db, scene)
    seed_navigation(db, scene)

    print(
        json.dumps(
            {
                "db": db_path,
                "scene_id": scene.id,
                "scene_name": scene.name,
                "entities": [entity.name for entity in entities],
                "entity_count": len(entities),
                "nav_nodes": len(db.list_nav_nodes(scene.id)),
                "nav_edges": len(db.list_nav_edges(scene.id)),
                "reset": args.reset,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
