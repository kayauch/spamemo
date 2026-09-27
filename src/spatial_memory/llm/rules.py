from __future__ import annotations

import re

from spatial_memory.vision.prompts import localized_label

from .base import MemoryDraft, QueryDraft


class RuleBasedInterpreter:
    async def memory_from_observation(self, transcript: str | None, detected_label: str | None) -> MemoryDraft:
        spoken = (transcript or "").strip()
        label = (detected_label or "").strip()
        display_label = localized_label(label) if label else ""
        explicit_name = self.explicit_name(spoken)
        name = explicit_name or self._extract_name(spoken) or display_label or "unnamed object"
        tags = [x for x in (label, display_label, self._owner_tag(spoken)) if x]
        aliases = [label] if label and display_label and label != display_label else []
        metadata = {}
        if explicit_name:
            metadata["human_explicit_name"] = explicit_name
        owner = self._owner_tag(spoken)
        if owner:
            metadata["owner"] = owner
        return MemoryDraft(name=name, description=spoken or None, tags=tags, aliases=aliases, metadata=metadata)

    async def query(self, text: str) -> QueryDraft:
        return QueryDraft(search_text=text.strip())

    @staticmethod
    def explicit_name(text: str | None) -> str | None:
        if not text:
            return None
        patterns = [
            r"(?:この|その)?(?:箱|ケース|引き出し|棚|バッグ|袋|収納)(?:に|には)(.+?)(?:を)?(?:入れた|しまった|収納した)(?:よ|です|$)",
            r"(?:この|その)?(?:箱|ケース|引き出し|棚|バッグ|袋|収納)(?:に|には)(.+?)(?:が)?(?:入っている|入ってる)(?:よ|です|$)",
            r"(?:この|その)?(?:棚|机|テーブル|台)(?:の上)?に(.+?)(?:を)?(?:置いた|置いてある)(?:よ|です|$)",
            r"これは(.+?)(?:です|だよ|だ|$)",
            r"ここに(.+?)(?:があります|ある|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip(" 。、")
                value = re.sub(r"(?:を|が)$", "", value).strip()
                return value or None
        return None

    @classmethod
    def _extract_name(cls, text: str) -> str | None:
        explicit = cls.explicit_name(text)
        if explicit:
            return explicit
        return text.strip(" 。、") if text and len(text) <= 40 else None

    @staticmethod
    def _owner_tag(text: str) -> str | None:
        for owner in ("父", "母", "お父さん", "お母さん", "部長", "高木さん"):
            if owner in text:
                return owner
        return None
