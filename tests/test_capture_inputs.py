import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from spatial_memory.capture_inputs import CaptureInput, allocate_frame_budget, constrain_panorama_cameras, mask_vggt_padding, panorama_faces, prepare_capture
from spatial_memory.reconstruction import ReconstructionService
from spatial_memory.sensor_import import import_sensor_bundle
from spatial_memory.storage import SpatialMemoryDB
from spatial_recon.types import ReconstructionResult


def make_photo(path: Path, color=(120, 150, 180), size=(64, 48)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_budget_balances_videos_without_exceeding_total(tmp_path):
    inputs = [CaptureInput(tmp_path / "a.mp4", "video"), CaptureInput(tmp_path / "b.mp4", "video"), CaptureInput(tmp_path / "photo.jpg"), CaptureInput(tmp_path / "pano.jpg", "panorama")]
    assert allocate_frame_budget(inputs, 60) == [27, 26, 1, 6]
    with pytest.raises(ValueError, match="budget"):
        allocate_frame_budget(inputs, 8)


def test_panorama_projection_direction_and_poles(tmp_path):
    # Longitude gradient lets us verify front/right/left without relying on
    # implementation matrices. Latitude gradient independently checks poles.
    width, height = 1024, 512
    panorama = np.zeros((height, width, 3), dtype=np.uint8)
    panorama[..., 0] = np.arange(width)[None, :] * 255 // (width - 1)
    panorama[..., 1] = np.arange(height)[:, None] * 255 // (height - 1)
    source = tmp_path / "pano.png"
    Image.fromarray(panorama).save(source)
    faces = {name: np.asarray(Image.open(path)) for path, name, _ in panorama_faces(source, tmp_path / "faces", 64)}
    assert abs(int(faces["front"][32, 32, 0]) - 128) < 5
    assert abs(int(faces["right"][32, 32, 0]) - 191) < 5
    assert abs(int(faces["left"][32, 32, 0]) - 64) < 5
    assert faces["up"][32, 32, 1] < 10
    assert faces["down"][32, 32, 1] > 245


def test_panorama_requires_full_sphere_and_more_than_one_physical_view(tmp_path):
    wide = make_photo(tmp_path / "wide.jpg", size=(900, 300))
    with pytest.raises(ValueError, match="2:1"):
        panorama_faces(wide, tmp_path / "faces")
    valid = make_photo(tmp_path / "valid.jpg", size=(1024, 512))
    with pytest.raises(ValueError, match="physical viewpoints"):
        prepare_capture([CaptureInput(valid, "panorama")], tmp_path / "scene")


def test_capture_provenance_and_panorama_shared_camera_constraint(tmp_path):
    pano = make_photo(tmp_path / "pano.jpg", size=(1024, 512))
    photo = make_photo(tmp_path / "photo.jpg")
    frames, records, manifest = prepare_capture([CaptureInput(pano, "panorama"), CaptureInput(photo)], tmp_path / "scene", max_frames=8, parent_scene_id="old")
    assert len(frames) == 7
    assert manifest["parent_scene_id"] == "old"
    assert len({record["panorama_group"] for record in records[:6]}) == 1
    assert all(record["independent_baseline"] is False for record in records[:6])
    ext = np.repeat(np.eye(4)[None, :3], 7, axis=0).astype(np.float32)
    for i, record in enumerate(records[:6]):
        ext[i, :3, :3] = np.asarray(record["face_camera_to_panorama"]).T
        ext[i, :3, 3] = np.array([i * .1, 0, 0])
    result = ReconstructionResult(np.ones((7, 8, 8)), ext, np.repeat(np.eye(3)[None], 7, axis=0))
    constrain_panorama_cameras(result, records)
    centers = [-pose[:, :3].T @ pose[:, 3] for pose in result.extrinsics[:6]]
    assert np.allclose(centers, centers[0])
    assert result.metrics["panorama_camera_constraints"][0]["physical_viewpoints"] == 1
    assert np.allclose(result.intrinsics[0], [[4, 0, 3.5], [0, 4, 3.5], [0, 0, 1]])


def test_mixed_aspect_batch_padding_is_not_room_geometry(tmp_path):
    landscape = make_photo(tmp_path / "landscape.jpg", size=(1280, 720))
    square = make_photo(tmp_path / "square.jpg", size=(768, 768))
    result = ReconstructionResult(
        np.ones((2, 518, 518), np.float32), np.repeat(np.eye(4)[None, :3], 2, axis=0),
        np.repeat(np.eye(3)[None], 2, axis=0), confidence=np.full((2, 518, 518), 3.0, np.float32),
    )
    records = [{}, {}]
    mask_vggt_padding(result, [landscape, square], records)
    assert records[0]["valid_image_rect"] == [0, 112, 518, 406]
    assert np.all(result.depth[0, :112] == 0)
    assert np.all(result.confidence[0, 406:] == 0)
    assert np.all(result.depth[0, 112:406] == 1)
    assert np.all(result.depth[1] == 1)


class RecordingAdapter:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def infer(self, frames):
        self.calls.append(list(frames))
        if self.fail:
            raise RuntimeError("GPU unavailable")
        n = len(frames)
        return ReconstructionResult(np.ones((n, 8, 8), np.float32), np.repeat(np.eye(4)[None, :3], n, axis=0).astype(np.float32), np.repeat(np.array([[[4, 0, 4], [0, 4, 4], [0, 0, 1]]], np.float32), n, axis=0), images=np.zeros((n, 8, 8, 3), np.uint8))


def fake_spatial_model(**kwargs):
    raise ValueError("Too few synthetic test points for a model")


def test_revision_uses_one_joint_inference_and_preserves_parent(tmp_path, monkeypatch):
    monkeypatch.setattr("spatial_memory.reconstruction.build_spatial_model", fake_spatial_model)
    db = SpatialMemoryDB(tmp_path / "db.sqlite")
    service = ReconstructionService(db, tmp_path / "scenes")
    adapter = RecordingAdapter()
    service._adapter = adapter
    photos = [make_photo(tmp_path / f"p{i}.jpg", color=(40*i, 150, 180)) for i in range(3)]
    parent, _ = service.reconstruct_capture(photos[:2], "parent")
    before = {path: path.read_bytes() for path in Path(parent.frames_dir).parent.rglob("*") if path.is_file()}
    child, metrics = service.reconstruct_capture([photos[2]], "child", parent_scene_id=parent.id)
    assert len(adapter.calls) == 2
    assert len(adapter.calls[1]) == 3
    assert metrics["joint_inference"] is True
    assert metrics["parent_scene_id"] == parent.id
    assert child.id != parent.id
    assert child.parent_scene_id == parent.id
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert db.get_scene(parent.id).name == "parent"


def test_inference_failure_does_not_register_fake_scene(tmp_path):
    db = SpatialMemoryDB(tmp_path / "db.sqlite")
    service = ReconstructionService(db, tmp_path / "scenes")
    service._adapter = RecordingAdapter(fail=True)
    photos = [make_photo(tmp_path / f"p{i}.jpg", color=(40*i, 150, 180)) for i in range(2)]
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        service.reconstruct_capture(photos, "test")
    assert db.list_scenes() == []


def make_sensor_bundle(path: Path, *, unit="meters", bad_pose=False, extra=None):
    records = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for i in range(2):
            rgb = io.BytesIO()
            Image.new("RGB", (16, 12), (40, 80, 120)).save(rgb, format="PNG")
            archive.writestr(f"rgb/{i}.png", rgb.getvalue())
            depth = io.BytesIO()
            np.save(depth, np.full((12, 16), 2.0, np.float32), allow_pickle=False)
            archive.writestr(f"depth/{i}.npy", depth.getvalue())
            pose = np.eye(4)
            pose[0, 3] = -i
            if bad_pose:
                pose[0, 0] = 2
            records.append({"rgb": f"rgb/{i}.png", "depth": f"depth/{i}.npy", "intrinsics": [[8, 0, 8], [0, 8, 6], [0, 0, 1]], "extrinsics": pose.tolist()})
        archive.writestr("manifest.json", json.dumps({"schema_version": 1, "depth_unit": unit, "pose_convention": "world_to_camera_opencv", "frames": records}))
        if extra:
            member = zipfile.ZipInfo("placeholder")
            member.filename = extra
            archive.writestr(member, "malicious")
    return path


def test_sensor_import_preserves_metric_calibration(tmp_path):
    bundle = make_sensor_bundle(tmp_path / "capture.zip")
    result, frames, capture = import_sensor_bundle(bundle, tmp_path / "scene")
    assert result.metrics["model"] == "calibrated-rgbd"
    assert np.all(result.depth == 2)
    assert result.extrinsics[1, 0, 3] == -1
    assert len(frames) == 2
    assert capture["coordinate_units"] == "meters"


@pytest.mark.parametrize("extra", ["../escape.txt", "C:/escape.txt", "dir\\escape.txt"])
def test_sensor_zip_rejects_unsafe_members(tmp_path, extra):
    bundle = make_sensor_bundle(tmp_path / "capture.zip", extra=extra)
    with pytest.raises(ValueError, match="Unsafe path"):
        import_sensor_bundle(bundle, tmp_path / "scene")
    assert not (tmp_path / "escape.txt").exists()


def test_sensor_import_rejects_wrong_units_and_nonrigid_pose(tmp_path):
    with pytest.raises(ValueError, match="depth_unit"):
        import_sensor_bundle(make_sensor_bundle(tmp_path / "units.zip", unit="millimeters"), tmp_path / "one")
    with pytest.raises(ValueError, match="orthonormal"):
        import_sensor_bundle(make_sensor_bundle(tmp_path / "pose.zip", bad_pose=True), tmp_path / "two")


def test_duplicate_video_is_removed_before_frame_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    a, b = tmp_path / "original.mp4", tmp_path / "renamed.mp4"
    a.write_bytes(b"same video content")
    b.write_bytes(a.read_bytes())
    calls = []
    def extract(path, output, fps, quota, quality):
        calls.append(quota)
        output.mkdir(parents=True)
        return [SimpleNamespace(path=make_photo(output / f"{i}.jpg", color=(i, 50, 100)),
                source_frame_index=i, timestamp_seconds=float(i), sharpness=20., motion_score=1.) for i in range(quota)]
    monkeypatch.setattr("spatial_memory.capture_inputs.extract_video_frames_with_metadata", extract)
    frames, records, manifest = prepare_capture([CaptureInput(a, "video"), CaptureInput(b, "video")], tmp_path / "scene", max_frames=12)
    assert calls == [12]
    assert len(frames) == 12 and len(manifest["sources"]) == 1
    assert len(manifest["duplicates_removed"]) == 1
    assert a.read_bytes() == b.read_bytes() == b"same video content"


def test_duplicate_photos_cannot_claim_multiple_viewpoints(tmp_path):
    a = make_photo(tmp_path / "a.jpg")
    b = tmp_path / "b.jpg"
    b.write_bytes(a.read_bytes())
    with pytest.raises(ValueError, match="physical viewpoints"):
        prepare_capture([a, b], tmp_path / "scene")
