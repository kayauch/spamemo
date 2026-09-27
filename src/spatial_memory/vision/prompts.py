from __future__ import annotations

import re


_JA_TO_EN = {
    "眼鏡": "eyeglasses",
    "メガネ": "eyeglasses",
    "猫": "cat",
    "ネコ": "cat",
    "鍵": "key",
    "かぎ": "key",
    "財布": "wallet",
    "スマホ": "smartphone",
    "携帯": "smartphone",
    "リモコン": "remote control",
    "ケーブル": "cable",
    "hdmi": "HDMI cable",
    "箱": "box",
    "段ボール": "cardboard box",
    "本": "book",
    "ペン": "pen",
    "ハサミ": "scissors",
    "はさみ": "scissors",
    "机": "desk",
    "椅子": "chair",
    "イス": "chair",
    "棚": "shelf",
    "ロボット": "robot",
}


def prompt_from_transcript(text: str | None, explicit: str | None = None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()

    normalized = (text or "").strip()
    lowered = normalized.lower()
    for key, value in _JA_TO_EN.items():
        if key.lower() in lowered:
            return value

    latin = re.findall(r"[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*)*", normalized)
    if latin:
        return max(latin, key=len).strip()
    return None


DEFAULT_INDOOR_LABELS = [
    "desk",
    "chair",
    "shelf",
    "box",
    "eyeglasses",
    "key",
    "wallet",
    "smartphone",
    "remote control",
    "laptop",
    "monitor",
    "keyboard",
    "mouse",
    "book",
    "bottle",
    "mug",
    "backpack",
    "cable",
]

_EN_TO_JA = {
    "desk": "机",
    "chair": "椅子",
    "shelf": "棚",
    "box": "箱",
    "eyeglasses": "眼鏡",
    "key": "鍵",
    "wallet": "財布",
    "smartphone": "スマホ",
    "remote control": "リモコン",
    "laptop": "ノートPC",
    "monitor": "モニター",
    "keyboard": "キーボード",
    "mouse": "マウス",
    "book": "本",
    "bottle": "ボトル",
    "mug": "マグカップ",
    "backpack": "リュック",
    "cable": "ケーブル",
    "HDMI cable": "HDMIケーブル",
}


def localized_label(label: str) -> str:
    key = label.strip()
    return _EN_TO_JA.get(key, _EN_TO_JA.get(key.lower(), key))
