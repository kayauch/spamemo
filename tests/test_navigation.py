from pathlib import Path

from spatial_memory.models import NavEdge, NavNode, Scene, SpatialEntity, Vec3
from spatial_memory.navigation import route_to_entity
from spatial_memory.storage import SpatialMemoryDB


def test_route_to_entity(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(id="room", name="Room"))
    db.upsert_entity(SpatialEntity(
        id="target", scene_id="room", name="Target",
        position=Vec3(x=2.2, y=0, z=0.0),
    ))
    for node in [
        NavNode(id="a", scene_id="room", name="A", position=Vec3(x=0, y=0, z=0)),
        NavNode(id="b", scene_id="room", name="B", position=Vec3(x=1, y=0, z=0)),
        NavNode(id="c", scene_id="room", name="C", position=Vec3(x=2, y=0, z=0)),
    ]:
        db.upsert_nav_node(node)
    db.upsert_nav_edge(NavEdge(scene_id="room", source_id="a", target_id="b"))
    db.upsert_nav_edge(NavEdge(scene_id="room", source_id="b", target_id="c"))

    route = route_to_entity(db, "room", Vec3(x=-0.2, y=0, z=0), "target")
    assert route.node_ids == ["a", "b", "c"]
    assert route.points[-1].x == 2.2
    assert route.total_cost > 2.0


def test_route_falls_back_to_direct_line_without_graph(tmp_path: Path) -> None:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    db.upsert_scene(Scene(id="room", name="Room"))
    db.upsert_entity(SpatialEntity(
        id="target-direct",
        scene_id="room",
        name="Direct Target",
        position=Vec3(x=3.0, y=0.0, z=4.0),
    ))

    route = route_to_entity(
        db,
        "room",
        Vec3(x=0.0, y=0.0, z=0.0),
        "target-direct",
    )
    assert route.node_ids == []
    assert len(route.points) == 2
    assert abs(route.total_cost - 5.0) < 1e-6
