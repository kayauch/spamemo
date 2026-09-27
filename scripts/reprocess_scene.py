from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "vggt"))

from spatial_recon.adapters.vggt import VGGTAdapter
from spatial_recon.geometry import save_point_cloud
from spatial_recon.io import save_result
from spatial_memory.models import Scene
from spatial_memory.spatial_model import build_spatial_model
from spatial_memory.storage import SpatialMemoryDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reprocess_scene")


def reprocess_scene(
    scene_id: str = "90839eb4-82ed-44fd-9265-c377f0624dbc",
    max_frames: int = 60,
    point_stride: int = 4,
):
    db_path = PROJECT_ROOT / "data" / "spatial-memory-v2.db"
    scenes_dir = PROJECT_ROOT / "data" / "scenes-v2"
    scene_dir = scenes_dir / scene_id

    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    db = SpatialMemoryDB(db_path)
    scene = db.get_scene(scene_id)
    scene_name = scene.name if scene else f"Scene {scene_id[:8]}"

    frames_dir = scene_dir / "frames"
    frames = sorted(list(frames_dir.glob("*.jpg")))
    if not frames:
        frames = sorted(list(frames_dir.glob("*.png")))

    if len(frames) < 2:
        raise ValueError(f"Not enough frames found in {frames_dir}: {len(frames)}")

    if len(frames) > max_frames:
        step = len(frames) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        frames = [frames[i] for i in indices]

    logger.info(f"Using {len(frames)} frames for genuine VGGT-1B 3D reconstruction...")

    checkpoint_path = PROJECT_ROOT / "checkpoints" / "vggt" / "model.pt"
    repo_path = PROJECT_ROOT / "third_party" / "vggt"

    adapter = VGGTAdapter(
        checkpoint=checkpoint_path,
        repo_path=repo_path,
        device="cuda",
    )

    logger.info("Loading VGGT-1B model onto RTX 5060 Ti GPU...")
    adapter.load()
    logger.info(f"Model loaded: {adapter.load_metrics}")

    logger.info(f"Running 3D inference across {len(frames)} frames...")
    result = adapter.infer(frames)
    logger.info(f"Inference complete in {result.metrics.get('inference_seconds', 0):.2f}s!")

    reconstruction_dir = scene_dir / "reconstruction"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    npz_path = save_result(result, reconstruction_dir)
    ply_path = reconstruction_dir / "scene.ply"
    point_count = save_point_cloud(result, ply_path, stride=point_stride)
    result.metrics["point_count"] = point_count
    logger.info(f"Point cloud saved to {ply_path} ({point_count:,} points)")

    spatial_model_dir = scene_dir / "spatial_model"
    spatial_model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building spatial model (walls, floor, ceiling, shell mesh)...")
    spatial_model = build_spatial_model(
        point_cloud_path=ply_path,
        output_dir=spatial_model_dir,
        scene_id=scene_id,
        target_triangles=20_000,
    )
    spatial_model_path = spatial_model["manifest_path"]
    logger.info(
        f"Spatial model built! Components: {spatial_model['component_count']}, "
        f"Shell triangles: {spatial_model['shell']['triangles']}"
    )

    result.metrics["spatial_model"] = {
        "manifest_path": spatial_model_path,
        "component_count": spatial_model["component_count"],
        "shell_vertices": spatial_model["shell"]["vertices"],
        "shell_triangles": spatial_model["shell"]["triangles"],
        "total_seconds": spatial_model["metrics"]["total_seconds"],
    }
    save_result(result, reconstruction_dir)

    video_path = scene_dir / "input" / "scene.mp4"
    updated_scene = Scene(
        id=scene_id,
        name=scene_name,
        description=scene.description if scene else None,
        point_cloud_path=str(ply_path),
        spatial_model_path=spatial_model_path,
        reconstruction_npz=str(npz_path),
        frames_dir=str(frames_dir),
        source_video_path=str(video_path) if video_path.exists() else None,
        reconstruction_model="VGGT-1B",
        frame_count=len(frames),
    )
    db.upsert_scene(updated_scene)
    logger.info(f"Successfully reprocessed scene '{scene_id}' with VGGT-1B!")
    return updated_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default="90839eb4-82ed-44fd-9265-c377f0624dbc")
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    reprocess_scene(
        scene_id=args.scene_id,
        max_frames=args.max_frames,
        point_stride=args.stride,
    )
