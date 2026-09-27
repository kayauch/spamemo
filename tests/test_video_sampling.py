from pathlib import Path

import cv2
import numpy as np
import pytest

from spatial_recon.video import extract_video_frames, extract_video_frames_with_metadata


def make_motion_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (96, 64),
    )
    assert writer.isOpened()
    for i in range(48):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        if i % 9 not in (0, 1):
            x = (i * 3) % 70
            cv2.rectangle(frame, (x, 18), (x + 22, 42), (255, 255, 255), -1)
            cv2.line(frame, (0, i % 64), (95, (i * 2) % 64), (120, 120, 120), 2)
        writer.write(frame)
    writer.release()


def test_quality_filtered_frame_extraction_keeps_usable_sequence(tmp_path: Path) -> None:
    video = tmp_path / "motion.avi"
    make_motion_video(video)
    output = tmp_path / "frames"

    frames = extract_video_frames(
        video,
        output,
        sample_fps=3.0,
        max_frames=10,
        quality_filter=True,
    )

    assert 2 <= len(frames) <= 10
    assert all(path.exists() for path in frames)
    assert [path.name for path in frames] == [
        f"{i:06d}.jpg" for i in range(len(frames))
    ]


def test_frame_metadata_preserves_source_timestamps(tmp_path: Path) -> None:
    video = tmp_path / "motion.avi"
    make_motion_video(video)
    output = tmp_path / "metadata-frames"

    frames = extract_video_frames_with_metadata(
        video,
        output,
        sample_fps=3.0,
        max_frames=6,
        quality_filter=False,
    )

    assert len(frames) == 6
    assert frames[0].source_frame_index == 0
    assert abs(frames[0].timestamp_seconds - 0.0) < 1e-6
    assert frames[1].source_frame_index == 8
    assert abs(frames[1].timestamp_seconds - (8 / 12.0)) < 1e-6
    assert frames[-1].source_frame_index == 44
    assert all(item.path.exists() for item in frames)


def test_quality_sampling_covers_end_of_long_video(tmp_path: Path) -> None:
    video = tmp_path / "motion.avi"
    make_motion_video(video)
    frames = extract_video_frames_with_metadata(video, tmp_path / "all", sample_fps=12, max_frames=4, quality_filter=True)
    assert len(frames) == 4
    assert frames[0].source_frame_index < 12
    assert frames[-1].source_frame_index >= 36
    assert all(a.source_frame_index < b.source_frame_index for a, b in zip(frames, frames[1:]))


def test_invalid_video_sampling_parameters_are_rejected(tmp_path: Path) -> None:
    import pytest
    for value in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="sample_fps"):
            extract_video_frames_with_metadata(tmp_path / "missing", tmp_path / "out", sample_fps=value)
    with pytest.raises(ValueError, match="max_frames"):
        extract_video_frames_with_metadata(tmp_path / "missing", tmp_path / "out", max_frames=0)


def test_quality_sampling_does_not_delete_unrelated_images(tmp_path: Path) -> None:
    video = tmp_path / "motion.avi"
    make_motion_video(video)
    output = tmp_path / "frames"
    output.mkdir()
    unrelated = output / "existing-reference.jpg"
    unrelated.write_bytes(b"preserve")
    extract_video_frames_with_metadata(video, output, max_frames=4, quality_filter=True)
    assert unrelated.read_bytes() == b"preserve"


@pytest.mark.parametrize("reported_count", [
    pytest.param(-9.223372036854776e18, id="negative-count"),
    pytest.param(0.0, id="zero-count"),
    pytest.param(float("nan"), id="nan-count"),
    pytest.param(float("inf"), id="infinite-count"),
])
@pytest.mark.parametrize("reported_fps", [
    pytest.param(12.0, id="accurate-fps"),
    pytest.param(1000.0, id="webm-timebase-fps"),
])
@pytest.mark.parametrize("quality_filter", [False, True], ids=["uniform", "quality"])
def test_unknown_video_length_and_timebase_metadata_cover_whole_capture(
    tmp_path: Path, monkeypatch, reported_count: float,
    reported_fps: float, quality_filter: bool,
) -> None:
    """Use real decoded video; only container metadata is unreliable.

    Browser MediaRecorder WebM can report a negative frame count and a
    1000 Hz timebase instead of its recording rate. A valid long-enough
    capture must still supply distinct frames across its full sequence.
    """
    video = tmp_path / "motion.avi"
    make_motion_video(video)
    real_video_capture = cv2.VideoCapture

    class MetadataOverrideCapture:
        def __init__(self, *args, **kwargs):
            self.capture = real_video_capture(*args, **kwargs)

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return reported_count
            if property_id == cv2.CAP_PROP_FPS:
                return reported_fps
            return self.capture.get(property_id)

        def __getattr__(self, name):
            # Decode, timestamp, seek, grab and release are OpenCV's real
            # operations; no replacement images or fake frame reads.
            return getattr(self.capture, name)

    monkeypatch.setattr(cv2, "VideoCapture", MetadataOverrideCapture)
    frames = extract_video_frames_with_metadata(
        video, tmp_path / "metadata-recovery", sample_fps=4,
        max_frames=4, quality_filter=quality_filter,
    )

    assert len(frames) == 4
    indexes = [frame.source_frame_index for frame in frames]
    assert indexes == sorted(set(indexes))
    assert 0 <= indexes[0] < 12
    assert 36 <= indexes[-1] < 48
    assert all(frame.path.is_file() for frame in frames)
    assert all(cv2.imread(str(frame.path)).shape == (64, 96, 3) for frame in frames)
