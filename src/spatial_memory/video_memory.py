from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .models import Observation
from .projection import project_bbox_from_npz
from .service import SpatialMemoryService
from .speech import SpeechTranscriber
from .storage import SpatialMemoryDB
from .vision.grounding_dino import GroundingDinoDetector, select_target
from .vision.prompts import prompt_from_transcript


_KIND_BY_LABEL = {
    "desk": "furniture",
    "chair": "furniture",
    "shelf": "furniture",
    "monitor": "furniture",
    "box": "container",
}


def _split_phrases(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])\s*", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _load_frame_manifest(frames_dir: Path) -> list[dict]:
    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = data.get("frames") or []
        if frames:
            return frames

    image_paths = sorted(
        path
        for path in frames_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    return [
        {
            "index": index,
            "name": path.name,
            "source_frame_index": index,
            "timestamp_seconds": float(index),
        }
        for index, path in enumerate(image_paths)
    ]


def _nearest_frame(
    frames: list[dict],
    start_seconds: float,
    end_seconds: float,
) -> dict:
    if not frames:
        raise ValueError("Scene has no frame manifest")
    midpoint = (start_seconds + end_seconds) * 0.5
    inside = [
        frame
        for frame in frames
        if start_seconds <= float(frame.get("timestamp_seconds", 0.0)) <= end_seconds
    ]
    pool = inside or frames
    return min(
        pool,
        key=lambda frame: abs(float(frame.get("timestamp_seconds", 0.0)) - midpoint),
    )


class SceneVideoMemoryIndexer:
    def __init__(
        self,
        db: SpatialMemoryDB,
        memory: SpatialMemoryService,
        detector: GroundingDinoDetector,
        transcriber: SpeechTranscriber,
    ):
        self.db = db
        self.memory = memory
        self.detector = detector
        self.transcriber = transcriber

    async def run(
        self,
        scene_id: str,
        language: str | None = "ja",
        detection_threshold: float = 0.25,
        text_threshold: float = 0.25,
        auto_confirm_threshold: float = 0.60,
    ) -> dict:
        scene = self.db.get_scene(scene_id)
        if scene is None:
            raise ValueError("Scene not found")
        if not scene.source_video_path:
            raise ValueError("Scene has no source video")
        if not scene.frames_dir or not scene.reconstruction_npz:
            raise ValueError("Scene requires frames and reconstruction data")

        video_path = Path(scene.source_video_path)
        frames_dir = Path(scene.frames_dir)
        if not video_path.exists():
            raise ValueError("Source video file not found")
        if not frames_dir.exists():
            raise ValueError("Frames directory not found")

        frames = _load_frame_manifest(frames_dir)
        raw_video = video_path.read_bytes()
        transcription = self.transcriber.transcribe_segments(
            raw_video,
            language=language,
        )

        confirmed: list[dict] = []
        pending: list[dict] = []
        utterances: list[dict] = []

        for segment_index, segment in enumerate(transcription.segments):
            phrases = _split_phrases(segment.text) or [segment.text.strip()]
            for phrase_index, phrase in enumerate(phrases):
                if not phrase:
                    continue
                frame = _nearest_frame(
                    frames,
                    segment.start_seconds,
                    segment.end_seconds,
                )
                frame_index = int(frame["index"])
                frame_path = frames_dir / str(frame["name"])
                item = {
                    "segment_index": segment_index,
                    "phrase_index": phrase_index,
                    "text": phrase,
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "frame_index": frame_index,
                    "frame_timestamp_seconds": float(frame.get("timestamp_seconds", 0.0)),
                    "frame_name": frame_path.name,
                }
                detection_prompt = prompt_from_transcript(phrase)
                item["prompt"] = detection_prompt
                utterances.append(dict(item))

                if not detection_prompt:
                    pending.append({
                        **item,
                        "reason": "no_object_prompt",
                    })
                    continue

                image = Image.open(frame_path).convert("RGB")
                detections = self.detector.detect(
                    image,
                    labels=[detection_prompt],
                    threshold=detection_threshold,
                    text_threshold=text_threshold,
                )
                target = select_target(detections)
                if target is None:
                    pending.append({
                        **item,
                        "reason": "not_detected",
                    })
                    continue

                bbox = tuple(int(round(value)) for value in target.bbox_xyxy)
                candidate = {
                    **item,
                    "detected_label": target.label,
                    "detection_confidence": float(target.score),
                    "bbox_xyxy": list(target.bbox_xyxy),
                    "image_width": target.image_width,
                    "image_height": target.image_height,
                }

                if float(target.score) < auto_confirm_threshold:
                    pending.append({
                        **candidate,
                        "reason": "low_confidence",
                    })
                    continue

                try:
                    position = project_bbox_from_npz(
                        scene.reconstruction_npz,
                        frame_index,
                        bbox,
                        source_size=(image.width, image.height),
                    )
                except (IndexError, KeyError, ValueError) as exc:
                    pending.append({
                        **candidate,
                        "reason": "projection_failed",
                        "detail": str(exc),
                    })
                    continue

                observation = Observation(
                    scene_id=scene_id,
                    position=position,
                    kind=_KIND_BY_LABEL.get(target.label.lower(), "object"),
                    transcript=phrase,
                    detected_label=target.label,
                    detection_confidence=float(target.score),
                    image_path=str(frame_path),
                    frame_index=frame_index,
                    detection_bbox_xyxy=tuple(float(value) for value in target.bbox_xyxy),
                    metadata={
                        "video_audio_memory": True,
                        "audio_segment_start_seconds": segment.start_seconds,
                        "audio_segment_end_seconds": segment.end_seconds,
                        "frame_timestamp_seconds": float(frame.get("timestamp_seconds", 0.0)),
                    },
                )
                entity = await self.memory.ingest_observation(observation)
                confirmed.append({
                    **candidate,
                    "position": position.model_dump(),
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                })

        report = {
            "scene_id": scene_id,
            "transcript": transcription.text,
            "language": transcription.language,
            "transcription_metrics": transcription.metrics,
            "segments": [asdict(segment) for segment in transcription.segments],
            "utterances": utterances,
            "confirmed": confirmed,
            "pending": pending,
            "confirmed_count": len(confirmed),
            "pending_count": len(pending),
            "auto_confirm_threshold": auto_confirm_threshold,
            "detection_threshold": detection_threshold,
            "vision_metrics": self.detector.last_metrics,
        }
        report_path = frames_dir.parent / "audio_memory.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report["report_path"] = str(report_path)
        return report
