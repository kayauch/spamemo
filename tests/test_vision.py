from spatial_memory.vision.grounding_dino import select_target
from spatial_memory.vision.models import Detection2D
from spatial_memory.vision.prompts import prompt_from_transcript


def test_prompt_from_japanese_transcript() -> None:
    assert prompt_from_transcript("これは父の眼鏡です") == "eyeglasses"
    assert prompt_from_transcript("ここに予備の鍵があります") == "key"
    assert prompt_from_transcript("これはロボットです") == "robot"


def test_select_target_prefers_good_center_detection() -> None:
    detections = [
        Detection2D(
            label="key",
            score=0.80,
            bbox_xyxy=(0, 0, 100, 100),
            image_width=640,
            image_height=480,
        ),
        Detection2D(
            label="key",
            score=0.72,
            bbox_xyxy=(280, 200, 360, 280),
            image_width=640,
            image_height=480,
        ),
    ]
    selected = select_target(detections, prefer_center=0.50)
    assert selected is not None
    assert selected.bbox_xyxy == (280.0, 200.0, 360.0, 280.0)
