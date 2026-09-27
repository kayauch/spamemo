"""Exercise the running local API using the bundled video fixture."""
import json
from pathlib import Path
import httpx
import torch

root = Path(__file__).resolve().parents[1]
expected_device = "cuda" if torch.cuda.is_available() else "cpu"
with httpx.Client(base_url="http://127.0.0.1:8001", timeout=300) as client:
    with (root / "test-assets/fake-camera.y4m").open("rb") as video:
        response = client.post("/scenes/reconstruct", files={"video": ("fixture.y4m", video, "video/x-yuv4mpeg")}, data={"name": f"{expected_device.upper()} VGGT fixture smoke", "sample_fps": "1", "max_frames": "2", "point_stride": "8", "quality_filter": "false"})
    response.raise_for_status()
    payload = response.json()
    (root / "data/local-reconstruction-smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    metrics = payload["metrics"]
    assert metrics["model"] == "VGGT-1B" and not metrics.get("fallback")
    assert metrics["device"] == expected_device and metrics["point_count"] > 0
    assert "spatial_model_error" not in metrics
    scene_id = payload["scene"]["id"]
    for url in ("/", "/static/app.js", f"/viewer/state/{scene_id}", f"/scenes/{scene_id}/point-cloud?max_points=5000"):
        result = client.get(url)
        result.raise_for_status()
        print(url, result.status_code, len(result.content), flush=True)
