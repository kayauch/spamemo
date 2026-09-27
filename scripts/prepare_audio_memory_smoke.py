from __future__ import annotations

import json
import shutil
from pathlib import Path

from spatial_memory.models import Scene
from spatial_memory.storage import SpatialMemoryDB


SCENE_ID = "audio-memory-smoke"
RESULTS = Path("/results")
SOURCE_FRAMES = RESULTS / "da3-video-smoke" / "frames"
SOURCE_RECON = RESULTS / "da3-video-smoke" / "reconstruction.npz"
SOURCE_PLY = RESULTS / "da3-video-smoke" / "scene.ply"
FIXTURE_AUDIO = Path("/workspace/test-assets/voice-memory-ja.wav")
FIXTURE_MANIFEST = Path("/workspace/test-assets/audio-memory-smoke-manifest.json")
SCENE_ROOT = RESULTS / SCENE_ID
FRAMES_DIR = SCENE_ROOT / "frames"
DB_PATH = RESULTS / "spatial-memory-v2.db"


def main() -> None:
    required = [
        SOURCE_FRAMES,
        SOURCE_RECON,
        FIXTURE_AUDIO,
        FIXTURE_MANIFEST,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing audio-memory smoke fixture(s): " + ", ".join(missing))

    if SCENE_ROOT.exists():
        shutil.rmtree(SCENE_ROOT)
    shutil.copytree(SOURCE_FRAMES, FRAMES_DIR)
    shutil.copyfile(FIXTURE_MANIFEST, FRAMES_DIR / "manifest.json")

    db = SpatialMemoryDB(DB_PATH)
    with db.connect() as conn:
        conn.execute("DELETE FROM entities WHERE scene_id=?", (SCENE_ID,))
        conn.execute("DELETE FROM observations WHERE scene_id=?", (SCENE_ID,))

    scene = Scene(
        id=SCENE_ID,
        name="Audio Memory Smoke",
        point_cloud_path=str(SOURCE_PLY) if SOURCE_PLY.exists() else None,
        reconstruction_npz=str(SOURCE_RECON),
        frames_dir=str(FRAMES_DIR),
        source_video_path=str(FIXTURE_AUDIO),
        reconstruction_model="fixture",
        frame_count=35,
    )
    db.upsert_scene(scene)

    manifest = json.loads((FRAMES_DIR / "manifest.json").read_text(encoding="utf-8"))
    target = next(
        item for item in manifest["frames"]
        if int(item["index"]) == 10
    )
    print(json.dumps({
        "scene_id": SCENE_ID,
        "frame_count": scene.frame_count,
        "target_frame": target,
        "audio_bytes": FIXTURE_AUDIO.stat().st_size,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
