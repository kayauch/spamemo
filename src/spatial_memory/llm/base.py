from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class MemoryDraft(BaseModel):
    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryDraft(BaseModel):
    search_text: str
    requested_kind: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SpatialInterpreter(Protocol):
    async def memory_from_observation(self, transcript: str | None, detected_label: str | None) -> MemoryDraft:
        ...

    async def query(self, text: str) -> QueryDraft:
        ...
