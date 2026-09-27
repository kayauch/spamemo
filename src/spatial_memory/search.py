from __future__ import annotations

import re
import unicodedata

from .models import SearchHit, SpatialEntity

_STOP = ("どこだっけ", "どこですか", "どこにある", "どこ", "場所", "教えて", "探して")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"\b(?:please|where|is|are)\b", " ", text)
    for word in _STOP:
        text = text.replace(word, " ")
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def is_searchable(entity: SpatialEntity) -> bool:
    # Rendering assets are not memories of objects or places.
    return not entity.metadata.get("wall_photo")


def _metadata_text(entity: SpatialEntity) -> str:
    values = []
    for key in ("note", "owner", "location", "color", "category", "material"):
        value = entity.metadata.get(key)
        if isinstance(value, str) and not value.startswith(("data:", "http:", "https:")):
            values.append(value[:2000])
    return " ".join(values)


def score_entity(query: str, entity: SpatialEntity) -> SearchHit:
    q = normalize(query)
    if not q:
        return SearchHit(entity=entity, score=0.0)

    fields = [
        (entity.name, 1.00, "name"),
        (" ".join(entity.aliases), 1.00, "alias"),
        (" ".join(entity.tags), 0.80, "tag"),
        (_metadata_text(entity), 0.75, "metadata"),
        (entity.description or "", 0.55, "description"),
    ]
    score = 0.0
    reasons: list[str] = []
    for text, weight, reason in fields:
        n = normalize(text)
        if not n:
            continue
        if q == n:
            score += weight
            reasons.append(f"exact:{reason}")
        elif q in n or n in q:
            score += weight * 0.8
            reasons.append(f"substring:{reason}")
    return SearchHit(entity=entity, score=min(score, 1.0), reasons=reasons)


def search_entities(query: str, entities: list[SpatialEntity], limit: int = 5) -> list[SearchHit]:
    hits = [score_entity(query, e) for e in entities if is_searchable(e)]
    hits = [h for h in hits if h.score > 0]
    hits.sort(key=lambda h: (h.score, h.entity.confidence), reverse=True)
    return hits[:limit]
