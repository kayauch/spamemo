from __future__ import annotations

import json

import httpx

from .base import MemoryDraft, QueryDraft


class OpenAICompatibleInterpreter:
    def __init__(self, base_url: str, model: str, api_key: str = "local"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    async def _json(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    async def memory_from_observation(self, transcript: str | None, detected_label: str | None) -> MemoryDraft:
        data = await self._json(
            "Extract spatial-memory metadata. Return JSON with name, description, tags, aliases, metadata. Human naming overrides vision labels.",
            f"transcript={transcript!r}\ndetected_label={detected_label!r}",
        )
        return MemoryDraft.model_validate(data)

    async def query(self, text: str) -> QueryDraft:
        data = await self._json(
            "Convert a natural-language spatial search into JSON with search_text, requested_kind, metadata. Preserve identifying terms such as owner names.",
            text,
        )
        return QueryDraft.model_validate(data)
