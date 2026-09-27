from __future__ import annotations

import math
import time

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .models import Detection2D


class GroundingDinoDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.processor = None
        self.model = None
        self.last_metrics: dict[str, float | str] = {}

    def load(self) -> None:
        if self.model is not None:
            return
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device).eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
        else:
            peak = 0.0
        self.last_metrics = {
            "model": self.model_id,
            "load_seconds": time.perf_counter() - started,
            "load_peak_allocated_gib": peak,
        }

    def detect(
        self,
        image: Image.Image,
        labels: list[str],
        threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[Detection2D]:
        if not labels:
            return []
        self.load()
        assert self.processor is not None
        assert self.model is not None

        inputs = self.processor(
            images=image.convert("RGB"),
            text=[labels],
            return_tensors="pt",
        ).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        text_labels = results.get("text_labels", results.get("labels", []))
        detections: list[Detection2D] = []
        for box, score, label in zip(
            results["boxes"], results["scores"], text_labels
        ):
            xyxy = tuple(float(x) for x in box.tolist())
            detections.append(
                Detection2D(
                    label=str(label),
                    score=float(score.item()),
                    bbox_xyxy=xyxy,
                    image_width=image.width,
                    image_height=image.height,
                )
            )
        peak = (
            torch.cuda.max_memory_allocated() / 2**30
            if self.device.type == "cuda"
            else 0.0
        )
        self.last_metrics.update({
            "inference_seconds": time.perf_counter() - started,
            "inference_peak_allocated_gib": peak,
            "detections": float(len(detections)),
        })
        return detections


def select_target(
    detections: list[Detection2D],
    prefer_center: float = 0.35,
) -> Detection2D | None:
    if not detections:
        return None

    def rank(item: Detection2D) -> float:
        cx, cy = item.center
        nx = (cx - item.image_width / 2) / max(item.image_width / 2, 1)
        ny = (cy - item.image_height / 2) / max(item.image_height / 2, 1)
        center_distance = min(math.sqrt(nx * nx + ny * ny), 1.0)
        center_score = 1.0 - center_distance
        return item.score * (1.0 - prefer_center) + center_score * prefer_center

    return max(detections, key=rank)


def select_near_point(
    detections: list[Detection2D],
    x: float,
    y: float,
) -> Detection2D | None:
    if not detections:
        return None

    def cost(item: Detection2D) -> float:
        cx, cy = item.center
        diagonal = math.hypot(item.image_width, item.image_height)
        distance = math.hypot(cx - x, cy - y) / max(diagonal, 1.0)
        return distance - 0.15 * item.score

    return min(detections, key=cost)
