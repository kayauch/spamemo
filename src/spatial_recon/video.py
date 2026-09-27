from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ExtractedVideoFrame:
    path: Path
    source_frame_index: int
    timestamp_seconds: float
    sharpness: float | None = None
    motion_score: float | None = None


def _frame_signature(frame: np.ndarray) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = min(1.0, 192.0 / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return sharpness, gray


def _motion_score(previous: np.ndarray | None, current: np.ndarray) -> float | None:
    if previous is None:
        return None
    if previous.shape != current.shape:
        current = cv2.resize(current, (previous.shape[1], previous.shape[0]))
    diff = cv2.absdiff(previous, current)
    return float(diff.mean() / 255.0)


def extract_video_frames_with_metadata(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 2.0,
    max_frames: int = 120,
    quality_filter: bool = False,
) -> list[ExtractedVideoFrame]:
    """Select temporal bins across the *whole* video, using bounded memory.

    ``max_frames`` limits inference input, never how far we read the video.
    Each bin retains its sharpest useful view on disk. Only one decoded image
    and a small previous-frame signature are retained in memory.
    """
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and greater than zero")
    if not 1 <= max_frames <= 10000:
        raise ValueError("max_frames must be between 1 and 10000")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Video could not be opened")
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    reliable_fps = np.isfinite(reported_fps) and 0 < reported_fps <= 240
    reliable_count = np.isfinite(reported_count) and reported_count > 0
    source_fps = reported_fps if reliable_fps else 30.0
    total_frames = int(reported_count) if reliable_count else 0
    if not reliable_count or not reliable_fps:
        # MediaRecorder WebM may report a negative sentinel frame count and
        # a 1000 Hz timebase as FPS. Count from zero and measure the real rate
        # from decoded presentation times, retaining no image array.
        total_frames = 0
        first_time = last_time = None
        try:
            while cap.grab():
                seconds = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if np.isfinite(seconds) and seconds >= 0:
                    if first_time is None:
                        first_time = seconds
                    last_time = seconds
                total_frames += 1
        finally:
            cap.release()
        if total_frames > 1 and first_time is not None and last_time is not None and last_time > first_time:
            source_fps = (total_frames - 1) / (last_time - first_time)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise ValueError("Video could not be reopened after counting frames")
    interval = max(int(round(source_fps / sample_fps)), 1)
    candidate_count = max(1, (total_frames + interval - 1) // interval)
    slots = min(max_frames, candidate_count)
    target_samples = np.linspace(0, candidate_count - 1, slots)
    selected: dict[int, ExtractedVideoFrame] = {}
    scores: dict[int, float] = {}
    previous_gray: np.ndarray | None = None
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_index = frame_index
            frame_index += 1
            if source_index % interval:
                continue
            sample_index = source_index // interval
            if quality_filter:
                slot = min(slots - 1, sample_index * slots // candidate_count)
                sharpness, gray = _frame_signature(frame)
                motion = _motion_score(previous_gray, gray)
                previous_gray = gray
                # Prefer detail and some change of view, while suppressing
                # cuts. A static/blank interval remains represented, allowing
                # the capture-quality report to ask for a replacement.
                novelty = 1.0 if motion is None else 0.65 + min(motion / 0.04, 1.0) * 0.35
                cut_penalty = 0.15 if motion is not None and motion > 0.55 else 1.0
                score = float(np.log1p(sharpness) * novelty * cut_penalty)
            else:
                slot = int(np.argmin(np.abs(target_samples - sample_index)))
                score = -abs(float(target_samples[slot]) - sample_index)
                sharpness, motion = None, None
            if slot in scores and score <= scores[slot]:
                continue
            path = output / f"{slot:06d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"Could not write extracted frame: {path}")
            scores[slot] = score
            presentation_seconds = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            timestamp_seconds = presentation_seconds if np.isfinite(presentation_seconds) and (presentation_seconds > 0 or source_index == 0) else source_index / source_fps
            selected[slot] = ExtractedVideoFrame(
                path=path,
                source_frame_index=source_index,
                timestamp_seconds=timestamp_seconds,
                sharpness=sharpness,
                motion_score=motion,
            )
    finally:
        cap.release()
    return [selected[slot] for slot in sorted(selected)]


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 2.0,
    max_frames: int = 120,
    quality_filter: bool = False,
) -> list[Path]:
    return [
        item.path
        for item in extract_video_frames_with_metadata(
            video_path,
            output_dir,
            sample_fps=sample_fps,
            max_frames=max_frames,
            quality_filter=quality_filter,
        )
    ]
