from __future__ import annotations

import json
import shutil
from pathlib import Path

from spatial_memory.models import Scene
from spatial_memory.spatial_model import build_spatial_model
from spatial_memory.storage import SpatialMemoryDB


SCENE_ID = "lowpoly-smoke"
RESULTS = Path("/results")
SOURCE_CLOUD = RESULTS / "da3-video-smoke" / "scene.ply"
SOURCE_RECON = RESULTS / "da3-video-smoke" / "reconstruction.npz"
SOURCE_FRAMES = RESULTS / "da3-video-smoke" / "frames"
ROOT = RESULTS / SCENE_ID
MODEL_DIR = ROOT / "spatial_model"
DB_PATH = RESULTS / "spatial-memory-v2.db"


def main() -> None:
    required = [SOURCE_CLOUD, SOURCE_RECON, SOURCE_FRAMES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing low-poly smoke fixture(s): " + ", ".join(missing))

    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)

    manifest = build_spatial_model(
        point_cloud_path=SOURCE_CLOUD,
        output_dir=MODEL_DIR,
        scene_id=SCENE_ID,
        target_triangles=20_000,
    )

    db = SpatialMemoryDB(DB_PATH)
    with db.connect() as conn:
        conn.execute("DELETE FROM entities WHERE scene_id=?", (SCENE_ID,))
        conn.execute("DELETE FROM observations WHERE scene_id=?", (SCENE_ID,))

    scene = Scene(
        id=SCENE_ID,
        name="Low Poly Smoke",
        point_cloud_path=str(SOURCE_CLOUD),
        spatial_model_path=manifest["manifest_path"],
        reconstruction_npz=str(SOURCE_RECON),
        frames_dir=str(SOURCE_FRAMES),
        reconstruction_model="fixture",
        frame_count=35,
    )
    db.upsert_scene(scene)

    print(json.dumps({
        "scene_id": SCENE_ID,
        "component_count": manifest["component_count"],
        "shell_vertices": manifest["shell"]["vertices"],
        "shell_triangles": manifest["shell"]["triangles"],
        "total_seconds": manifest["metrics"]["total_seconds"],
        "component_kinds": [
            component["kind"] for component in manifest["components"]
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
