from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from spatial_memory.models import NavEdge, NavNode, Scene, SpatialEntity, Vec3
from spatial_memory.spatial_model import build_spatial_model
from spatial_memory.storage import SpatialMemoryDB

DEMO_SCENE_ID = "00000000-0000-4000-8000-000000000001"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "spatial-memory-v2.db"
DEFAULT_SCENE_ROOT = DEFAULT_DATA_DIR / "scenes-v2"


def _synthetic_room_cloud(path: Path) -> None:
    # Build a room: floor (y=-1.0), ceiling (y=1.0), walls (x=±1.5, z=±1.5)
    # Plus a table in the center
    values_x = np.linspace(-1.5, 1.5, 30)
    values_z = np.linspace(-1.5, 1.5, 30)
    values_y = np.linspace(-1.0, 1.0, 20)

    # Floor & Ceiling
    grid_xz_x, grid_xz_z = np.meshgrid(values_x, values_z, indexing="ij")
    floor = np.column_stack([grid_xz_x.ravel(), np.full(grid_xz_x.size, -1.0), grid_xz_z.ravel()])
    ceiling = np.column_stack([grid_xz_x.ravel(), np.full(grid_xz_x.size, 1.0), grid_xz_z.ravel()])

    # Walls
    grid_xy_x, grid_xy_y = np.meshgrid(values_x, values_y, indexing="ij")
    wall_north = np.column_stack([grid_xy_x.ravel(), grid_xy_y.ravel(), np.full(grid_xy_x.size, 1.5)])
    wall_south = np.column_stack([grid_xy_x.ravel(), grid_xy_y.ravel(), np.full(grid_xy_x.size, -1.5)])

    grid_zy_z, grid_zy_y = np.meshgrid(values_z, values_y, indexing="ij")
    wall_east = np.column_stack([np.full(grid_zy_z.size, 1.5), grid_zy_y.ravel(), grid_zy_z.ravel()])
    wall_west = np.column_stack([np.full(grid_zy_z.size, -1.5), grid_zy_y.ravel(), grid_zy_z.ravel()])

    # Table in the center (y = -0.3, x in [-0.5, 0.5], z in [-0.4, 0.4])
    tx = np.linspace(-0.5, 0.5, 15)
    tz = np.linspace(-0.4, 0.4, 15)
    tg_x, tg_z = np.meshgrid(tx, tz, indexing="ij")
    table_top = np.column_stack([tg_x.ravel(), np.full(tg_x.size, -0.3), tg_z.ravel()])

    points = np.concatenate([floor, ceiling, wall_north, wall_south, wall_east, wall_west, table_top], axis=0)
    rng = np.random.default_rng(42)
    points += rng.normal(scale=0.003, size=points.shape)

    colors = np.empty_like(points)
    # Realistic indoor palette: warm floor, off-white walls, soft wood table
    for i, p in enumerate(points):
        if p[1] < -0.95:  # floor
            colors[i] = [0.65, 0.52, 0.40]
        elif p[1] > 0.95:  # ceiling
            colors[i] = [0.92, 0.92, 0.94]
        elif abs(p[1] - (-0.3)) < 0.05 and abs(p[0]) <= 0.55 and abs(p[2]) <= 0.45:  # table
            colors[i] = [0.45, 0.32, 0.22]
        else:  # walls
            colors[i] = [0.82, 0.84, 0.86]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def _generate_demo_frames(frames_dir: Path, count: int = 5) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = Image.new("RGB", (640, 480), color=(30 + i * 5, 34 + i * 4, 42 + i * 3))
        draw = ImageDraw.Draw(img)
        # Draw mock room outline
        draw.rectangle([50, 40, 590, 440], outline=(100, 110, 130), width=3)
        draw.rectangle([180, 200, 460, 360], fill=(80, 60, 45), outline=(140, 110, 80), width=2)
        draw.text((220, 250), f"Room Frame #{i}", fill=(240, 240, 240))
        draw.text((220, 280), "Spatial Memory V2 Demo", fill=(180, 190, 210))
        img.save(frames_dir / f"{i}.jpg", quality=90)


def _generate_reconstruction_npz(path: Path, frame_count: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.ones((frame_count, 48, 64), dtype=np.float32) * 2.0
    intrinsics = np.repeat(np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)[None], frame_count, axis=0)
    extrinsics = np.repeat(np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1)[None], frame_count, axis=0)
    np.savez(path, depth=depth, intrinsics=intrinsics, extrinsics=extrinsics)


def setup_local_demo(reset: bool = True) -> Scene:
    db_path = Path(os.getenv("SPATIAL_DB", str(DEFAULT_DB_PATH)))
    scene_root = Path(os.getenv("SCENE_ROOT", str(DEFAULT_SCENE_ROOT)))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    scene_root.mkdir(parents=True, exist_ok=True)

    db = SpatialMemoryDB(db_path)
    scene_dir = scene_root / DEMO_SCENE_ID
    scene_dir.mkdir(parents=True, exist_ok=True)

    cloud_path = scene_dir / "scene.ply"
    recon_path = scene_dir / "reconstruction.npz"
    frames_dir = scene_dir / "frames"
    model_dir = scene_dir / "spatial_model"

    print(f"Generating synthetic 3D room cloud at {cloud_path}...")
    _synthetic_room_cloud(cloud_path)

    print(f"Building 3D low-poly spatial model at {model_dir}...")
    manifest = build_spatial_model(
        point_cloud_path=cloud_path,
        output_dir=model_dir,
        scene_id=DEMO_SCENE_ID,
        target_triangles=4000,
        max_planes=6,
        max_cluster_components=8,
    )

    print(f"Generating preview frames at {frames_dir}...")
    _generate_demo_frames(frames_dir, count=5)

    print(f"Generating reconstruction NPZ at {recon_path}...")
    _generate_reconstruction_npz(recon_path, frame_count=5)

    if reset:
        with db.connect() as conn:
            for table in ("entities", "observations", "nav_nodes", "nav_edges"):
                conn.execute(f"DELETE FROM {table} WHERE scene_id=?", (DEMO_SCENE_ID,))
            conn.execute("DELETE FROM scenes WHERE id=?", (DEMO_SCENE_ID,))

    scene = Scene(
        id=DEMO_SCENE_ID,
        name="Hackathon Demo Room",
        description="ローカル動作用のデモルーム（合成点群＋低ポリゴン3Dモデル＋探索対象物体）",
        point_cloud_path=str(cloud_path),
        spatial_model_path=manifest["manifest_path"],
        reconstruction_npz=str(recon_path),
        frames_dir=str(frames_dir),
        reconstruction_model="fixture",
        frame_count=5,
    )
    db.upsert_scene(scene)

    # Entities
    box = SpatialEntity(
        id="demo-box",
        scene_id=DEMO_SCENE_ID,
        name="デモ用の箱",
        kind="container",
        position=Vec3(x=0.35, y=-0.25, z=0.0),
        tags=["箱", "box", "収納", "ケース"],
        aliases=["段ボール箱", "ボックス"],
        description="テーブルの上に置いてある小物入れの箱",
        source="human",
        confidence=1.0,
    )
    cable = SpatialEntity(
        id="demo-cable",
        scene_id=DEMO_SCENE_ID,
        name="充電ケーブル",
        kind="object",
        position=Vec3(x=0.35, y=-0.25, z=0.0),
        parent_id=box.id,
        relation_to_parent="inside",
        tags=["ケーブル", "充電", "cable", "USB"],
        aliases=["スマホの充電器", "Type-Cケーブル"],
        description="デモ用の箱の中に入っている充電ケーブル",
        source="human",
        confidence=1.0,
    )
    robot = SpatialEntity(
        id="demo-robot",
        scene_id=DEMO_SCENE_ID,
        name="ロボット",
        kind="object",
        position=Vec3(x=-0.35, y=-0.25, z=0.1),
        tags=["robot", "ロボット", "おもちゃ", "フィギュア"],
        aliases=["ロボットのおもちゃ"],
        description="テーブルの左側に置かれているロボット",
        source="human",
        confidence=1.0,
    )
    glasses = SpatialEntity(
        id="demo-glasses",
        scene_id=DEMO_SCENE_ID,
        name="父の眼鏡",
        kind="object",
        position=Vec3(x=0.0, y=-0.28, z=-0.15),
        tags=["眼鏡", "メガネ", "父", "リーディンググラス"],
        aliases=["お父さんのメガネ", "お父さんの眼鏡"],
        metadata={"owner": "父"},
        description="テーブルの中央手前に置いてある黒縁の眼鏡",
        source="human",
        confidence=1.0,
    )
    laptop = SpatialEntity(
        id="demo-laptop",
        scene_id=DEMO_SCENE_ID,
        name="ノートパソコン",
        kind="object",
        position=Vec3(x=-0.05, y=-0.27, z=0.15),
        tags=["パソコン", "PC", "ノートPC", "laptop"],
        aliases=["仕事用PC", "ラップトップ"],
        description="テーブルの上に開いて置いてあるノートパソコン",
        source="human",
        confidence=1.0,
    )
    plant = SpatialEntity(
        id="demo-plant",
        scene_id=DEMO_SCENE_ID,
        name="観葉植物",
        kind="object",
        position=Vec3(x=-1.2, y=-0.8, z=-1.1),
        tags=["植物", "観葉植物", "グリーン", "plant"],
        aliases=["植木鉢", "ポトス"],
        description="部屋の隅の床に置かれている観葉植物",
        source="human",
        confidence=1.0,
    )

    entities = [box, cable, robot, glasses, laptop, plant]
    for entity in entities:
        db.upsert_entity(entity)

    # Navigation
    nav_nodes = [
        NavNode(id="nav-entrance", scene_id=DEMO_SCENE_ID, name="部屋の入口", position=Vec3(x=0.0, y=-0.9, z=1.2)),
        NavNode(id="nav-center", scene_id=DEMO_SCENE_ID, name="部屋の中央", position=Vec3(x=0.0, y=-0.9, z=0.0)),
        NavNode(id="nav-table", scene_id=DEMO_SCENE_ID, name="テーブル前", position=Vec3(x=0.0, y=-0.9, z=-0.4)),
        NavNode(id="nav-corner", scene_id=DEMO_SCENE_ID, name="植物の近く", position=Vec3(x=-1.0, y=-0.9, z=-0.9)),
    ]
    for node in nav_nodes:
        db.upsert_nav_node(node)

    edges = [
        ("nav-entrance", "nav-center"),
        ("nav-center", "nav-table"),
        ("nav-center", "nav-corner"),
    ]
    for src, dst in edges:
        db.upsert_nav_edge(NavEdge(scene_id=DEMO_SCENE_ID, source_id=src, target_id=dst))

    print(f"Successfully seeded demo scene '{scene.name}' ({scene.id}) with {len(entities)} entities!")
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Spatial Memory local demo scene")
    parser.add_argument("--no-reset", action="store_true", help="Do not clear existing entities")
    args = parser.parse_args()

    scene = setup_local_demo(reset=not args.no_reset)
    print(json.dumps({
        "status": "ok",
        "scene_id": scene.id,
        "scene_name": scene.name,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
