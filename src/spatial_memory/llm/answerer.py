from __future__ import annotations

import json

import httpx

from spatial_memory.models import SpatialEntity


class SpatialAnswerer:
    def __init__(self, base_url: str, model: str, api_key: str = "local"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    @staticmethod
    def _entity_payload(entity: SpatialEntity) -> dict:
        return {
            "id": entity.id,
            "name": entity.name,
            "kind": entity.kind,
            "position": entity.position.model_dump(),
            "parent_id": entity.parent_id,
            "relation_to_parent": entity.relation_to_parent,
            "description": entity.description,
            "tags": entity.tags,
            "aliases": entity.aliases,
            "metadata": entity.metadata,
            "confidence": entity.confidence,
            "last_observed_at": entity.last_observed_at.isoformat(),
        }

    async def answer(
        self,
        question: str,
        entities: list[SpatialEntity],
    ) -> str:
        context = [self._entity_payload(entity) for entity in entities]
        system = (
            "You answer questions about a physical space using only the supplied spatial-memory records. "
            "Never invent an object, owner, coordinate, or location. "
            "If the evidence is insufficient, say that it is not confirmed. "
            "When stating a location, prefer natural Japanese and mention that it is the last confirmed location "
            "when appropriate. Be concise."
        )
        user = (
            f"Question:\n{question}\n\n"
            f"Authoritative spatial-memory records:\n"
            f"{json.dumps(context, ensure_ascii=False)}"
        )
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
