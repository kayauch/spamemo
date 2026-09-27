import json
import httpx

BASE = "http://api:8000"


def show(name: str, response: httpx.Response) -> dict:
    response.raise_for_status()
    data = response.json()
    print(f"--- {name} ---")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


with httpx.Client(timeout=10) as client:
    show("health", client.get(f"{BASE}/health"))
    first = show(
        "query-existing",
        client.post(f"{BASE}/query", json={
            "scene_id": "demo-room",
            "text": "父の眼鏡どこ？",
        }),
    )
    assert first["target"]["id"] == "glasses-father"

    created = show(
        "observation",
        client.post(f"{BASE}/observations", json={
            "scene_id": "demo-room",
            "position": {"x": 3.2, "y": 0.8, "z": -0.4},
            "kind": "object",
            "transcript": "これは予備の鍵です",
            "detected_label": "key",
            "detection_confidence": 0.88,
        }),
    )
    assert created["name"] == "予備の鍵"

    second = show(
        "query-new",
        client.post(f"{BASE}/query", json={
            "scene_id": "demo-room",
            "text": "予備の鍵どこ？",
        }),
    )
    assert second["target"]["name"] == "予備の鍵"
    assert second["target"]["position"]["x"] == 3.2


    guide = show(
        "guide",
        client.post(f"{BASE}/guide", json={
            "scene_id": "demo-room",
            "start": {"x": 0.0, "y": 0.0, "z": 4.2},
            "text": "父の眼鏡どこ？",
        }),
    )
    assert guide["query"]["target"]["id"] == "glasses-father"
    assert guide["route"]["target_entity_id"] == "glasses-father"
    assert len(guide["route"]["points"]) >= 3
