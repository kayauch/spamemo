import asyncio
import json
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_memory.llm.rules import RuleBasedInterpreter
from spatial_memory.models import Scene
from spatial_memory.service import SpatialMemoryService
from spatial_memory.speech.whisper import (
    TranscriptionResult,
    TranscriptionSegment,
)
from spatial_memory.storage import SpatialMemoryDB
from spatial_memory.video_memory import SceneVideoMemoryIndexer
from spatial_memory.vision.models import Detection2D


class FakeTranscriber:
    def transcribe_segments(self, audio: bytes, language: str | None = "ja"):
        return TranscriptionResult(
            text="これは父の眼鏡です。これは予備の鍵です。",
            language=language,
            metrics={"fake": "yes"},
            segments=[
                TranscriptionSegment(
                    text="これは父の眼鏡です。",
                    start_seconds=0.5,
                    end_seconds=1.5,
                ),
                TranscriptionSegment(
                    text="これは予備の鍵です。",
                    start_seconds=4.5,
                    end_seconds=5.5,
                ),
            ],
        )


class FakeDetector:
    def __init__(self, score: float = 0.9):
        self.score = score
        self.last_metrics = {"fake": True}

    def detect(self, image, labels, threshold=0.25, text_threshold=0.25):
        label = labels[0]
        return [
            Detection2D(
                label=label,
                score=self.score,
                bbox_xyxy=(2.0, 2.0, 6.0, 6.0),
                image_width=image.width,
                image_height=image.height,
            )
        ]


def build_scene(tmp_path: Path) -> tuple[SpatialMemoryDB, Scene]:
    db = SpatialMemoryDB(tmp_path / "memory.db")
    scene_dir = tmp_path / "scene"
    frames_dir = scene_dir / "frames"
    input_dir = scene_dir / "input"
    reconstruction_dir = scene_dir / "reconstruction"
    frames_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    reconstruction_dir.mkdir(parents=True)

    for index in range(2):
        Image.new("RGB", (10, 10), color=(20 + index * 20, 30, 40)).save(
            frames_dir / f"{index:06d}.jpg"
        )

    (frames_dir / "manifest.json").write_text(
        json.dumps({
            "frames": [
                {
                    "index": 0,
                    "name": "000000.jpg",
                    "source_frame_index": 12,
                    "timestamp_seconds": 1.0,
                },
                {
                    "index": 1,
                    "name": "000001.jpg",
                    "source_frame_index": 60,
                    "timestamp_seconds": 5.0,
                },
            ]
        }),
        encoding="utf-8",
    )
    video_path = input_dir / "scene.mp4"
    video_path.write_bytes(b"fake-video-with-audio")

    depth = np.full((2, 5, 5), 2.0, dtype=np.float32)
    intrinsics = np.array([
        [
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [0.0, 0.0, 1.0],
        ],
        [
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [0.0, 0.0, 1.0],
        ],
    ], dtype=np.float32)
    extrinsics = np.array([
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    ], dtype=np.float32)
    reconstruction_path = reconstruction_dir / "reconstruction.npz"
    np.savez(
        reconstruction_path,
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )

    scene = Scene(
        id="room",
        name="Room",
        frames_dir=str(frames_dir),
        source_video_path=str(video_path),
        reconstruction_npz=str(reconstruction_path),
        frame_count=2,
    )
    db.upsert_scene(scene)
    return db, scene


def test_two_timestamped_utterances_create_two_3d_entities(tmp_path: Path) -> None:
    db, scene = build_scene(tmp_path)
    memory = SpatialMemoryService(db, RuleBasedInterpreter())
    indexer = SceneVideoMemoryIndexer(
        db=db,
        memory=memory,
        detector=FakeDetector(score=0.9),
        transcriber=FakeTranscriber(),
    )

    report = asyncio.run(indexer.run(scene.id, auto_confirm_threshold=0.60))

    assert report["confirmed_count"] == 2
    assert report["pending_count"] == 0
    assert [item["frame_index"] for item in report["confirmed"]] == [0, 1]
    entities = db.list_entities(scene.id)
    assert {entity.name for entity in entities} == {"父の眼鏡", "予備の鍵"}
    positions = {entity.name: entity.position for entity in entities}
    assert abs(positions["父の眼鏡"].x) < 1e-6
    assert abs(positions["予備の鍵"].x - 2.0) < 1e-6
    assert (Path(scene.frames_dir).parent / "audio_memory.json").exists()


def test_low_confidence_audio_memory_waits_for_confirmation(tmp_path: Path) -> None:
    db, scene = build_scene(tmp_path)
    memory = SpatialMemoryService(db, RuleBasedInterpreter())
    indexer = SceneVideoMemoryIndexer(
        db=db,
        memory=memory,
        detector=FakeDetector(score=0.4),
        transcriber=FakeTranscriber(),
    )

    report = asyncio.run(indexer.run(scene.id, auto_confirm_threshold=0.60))

    assert report["confirmed_count"] == 0
    assert report["pending_count"] == 2
    assert {item["reason"] for item in report["pending"]} == {"low_confidence"}
    assert db.list_entities(scene.id) == []
