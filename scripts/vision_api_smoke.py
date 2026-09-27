from pathlib import Path

import httpx
import numpy as np

BASE = "http://api:8000"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def prepare_reconstruction() -> str:
    output = Path("/results/vision-smoke")
    output.mkdir(parents=True, exist_ok=True)
    depth = np.full((1, 120, 160), 2.0, dtype=np.float32)
    intrinsics = np.array([[
        [100.0, 0.0, 80.0],
        [0.0, 100.0, 60.0],
        [0.0, 0.0, 1.0],
    ]], dtype=np.float32)
    extrinsics = np.array([[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]], dtype=np.float32)
    path = output / "reconstruction.npz"
    np.savez(path, depth=depth, intrinsics=intrinsics, extrinsics=extrinsics)
    return str(path)


def main() -> None:
    reconstruction = prepare_reconstruction()
    image_bytes = httpx.get(IMAGE_URL, timeout=20).content

    with httpx.Client(timeout=60) as client:
        scene = {
            "id": "vision-smoke",
            "name": "Vision Smoke",
            "reconstruction_npz": reconstruction,
        }
        response = client.post(f"{BASE}/scenes", json=scene)
        response.raise_for_status()

        detect = client.post(
            f"{BASE}/vision/detect",
            files={"image": ("cats.jpg", image_bytes, "image/jpeg")},
            data={"prompt": "cat,remote control", "threshold": "0.30"},
        )
        detect.raise_for_status()
        detection_data = detect.json()
        print("detect target:", detection_data["target"])
        print("detect metrics:", detection_data["metrics"])
        assert detection_data["target"] is not None


        remember = client.post(
            f"{BASE}/scenes/vision-smoke/vision/remember",
            files={"image": ("cats.jpg", image_bytes, "image/jpeg")},
            data={
                "frame_index": "0",
                "transcript": "これは猫です",
                "threshold": "0.30",
            },
        )
        remember.raise_for_status()
        memory = remember.json()
        print("remember:", memory)
        assert memory["prompt"] == "cat"
        assert memory["entity"]["name"] == "猫"

        query = client.post(
            f"{BASE}/query",
            json={"scene_id": "vision-smoke", "text": "猫どこ？"},
        )
        query.raise_for_status()
        result = query.json()
        print("query target:", result["target"])
        assert result["target"]["name"] == "猫"


if __name__ == "__main__":
    main()
