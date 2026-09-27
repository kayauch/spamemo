from __future__ import annotations

from .base import MemoryDraft, QueryDraft, SpatialInterpreter
from .rules import RuleBasedInterpreter


class FallbackInterpreter:
    def __init__(self, primary: SpatialInterpreter, fallback: SpatialInterpreter):
        self.primary = primary
        self.fallback = fallback

    async def memory_from_observation(
        self,
        transcript: str | None,
        detected_label: str | None,
    ) -> MemoryDraft:
        try:
            primary = await self.primary.memory_from_observation(
                transcript,
                detected_label,
            )
        except Exception:
            return await self.fallback.memory_from_observation(
                transcript,
                detected_label,
            )

        explicit_name = RuleBasedInterpreter.explicit_name(transcript)
        if not explicit_name:
            return primary

        fallback = await self.fallback.memory_from_observation(
            transcript,
            detected_label,
        )
        aliases = list(
            dict.fromkeys(
                [
                    *primary.aliases,
                    *fallback.aliases,
                    *(
                        [primary.name]
                        if primary.name and primary.name != explicit_name
                        else []
                    ),
                ]
            )
        )
        tags = list(dict.fromkeys([*primary.tags, *fallback.tags]))
        metadata = dict(primary.metadata)
        metadata.update(fallback.metadata)
        metadata["human_explicit_name"] = explicit_name

        return primary.model_copy(
            update={
                "name": explicit_name,
                "description": primary.description or fallback.description,
                "tags": tags,
                "aliases": aliases,
                "metadata": metadata,
            }
        )

    async def query(self, text: str) -> QueryDraft:
        try:
            return await self.primary.query(text)
        except Exception:
            return await self.fallback.query(text)
