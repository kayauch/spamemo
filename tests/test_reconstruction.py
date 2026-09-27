import io
from pathlib import Path

import cv2
import numpy as np
from starlette.datastructures import UploadFile

from spatial_memory.reconstruction import ReconstructionService
from spatial_memory.storage import SpatialMemoryDB
from spatial_recon.types import ReconstructionResult


class FakeAdapter:
    def infer(self, image_paths: list[Path]) -> ReconstructionResult:
        n = len(image_paths)
        h, w = 8, 8
        depth = np.full((n, h, w), 2.0, dtype=np.float32)
        intrinsics = np.repeat(
            np.array([[[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]], dtype=np.float32),
            n,
            axis=0,
        )
        extrinsics = np.repeat(
            np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], dtype=np.float32),
            n,
            axis=0,
        )
        images = np.zeros((n, h, w, 3), dtype=np.uint8)
        return ReconstructionResult(
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            images=images,
            metrics={"inference_seconds": 0.01, "inference_peak_allocated_gib": 0.1},
        )


def make_video(path: Path) -> bytes:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for i in range(12):
        frame = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def test_reconstruction_service_registers_scene(tmp_path: Path) -> None:
    video_path = tmp_path / "source.avi"
    upload = UploadFile(
        filename="source.avi",
        file=io.BytesIO(make_video(video_path)),
    )
    db = SpatialMemoryDB(tmp_path / "memory.db")
    service = ReconstructionService(db, root_dir=tmp_path / "scenes")
    service._adapter = FakeAdapter()

    scene, metrics = service.reconstruct_upload(
        upload,
        name="Test Room",
        sample_fps=5.0,
        max_frames=5,
        point_stride=2,
    )

    assert scene.name == "Test Room"
    assert scene.frame_count >= 2
    assert Path(scene.reconstruction_npz).exists()
    assert Path(scene.point_cloud_path).exists()
    assert Path(scene.frames_dir).exists()
    assert db.get_scene(scene.id) is not None
    assert metrics["scene_id"] == scene.id



def test_reconstruction_backend_factory(tmp_path: Path) -> None:
    from spatial_recon.adapters import DA3Adapter, VGGTAdapter

    db = SpatialMemoryDB(tmp_path / "memory.db")

    vggt_service = ReconstructionService(
        db,
        root_dir=tmp_path / "scenes-vggt",
        backend="vggt",
        vggt_checkpoint=tmp_path / "model.pt",
        vggt_repo=tmp_path / "vggt-repo",
    )
    assert vggt_service.model_label == "VGGT-1B"
    assert isinstance(vggt_service.adapter, VGGTAdapter)

    da3_service = ReconstructionService(
        db,
        root_dir=tmp_path / "scenes-da3",
        backend="da3",
        model_id="depth-anything/DA3-LARGE-1.1",
    )
    assert da3_service.model_label == "depth-anything/DA3-LARGE-1.1"
    assert isinstance(da3_service.adapter, DA3Adapter)


def test_unknown_reconstruction_backend_is_rejected(tmp_path: Path) -> None:
    import pytest

    db = SpatialMemoryDB(tmp_path / "memory.db")
    service = ReconstructionService(
        db,
        root_dir=tmp_path / "scenes",
        backend="unknown",
    )
    with pytest.raises(ValueError, match="Unsupported reconstruction backend"):
        _ = service.adapter
