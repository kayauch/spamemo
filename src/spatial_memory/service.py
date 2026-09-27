from __future__ import annotations

import math
import asyncio
import logging

from .llm.base import SpatialInterpreter
from .models import Observation, QueryResult, SearchHit, SpatialEntity, Vec3
from .search import normalize, search_entities, is_searchable
from .semantic import SemanticSearchEngine
from .storage import SpatialMemoryDB


class SpatialMemoryService:
    def __init__(
        self,
        db: SpatialMemoryDB,
        interpreter: SpatialInterpreter,
        merge_radius: float = 0.75,
        semantic: SemanticSearchEngine | None = None,
        semantic_min_score: float = 0.80,
        semantic_min_margin: float = 0.035,
    ):
        self.db = db
        self.interpreter = interpreter
        self.merge_radius = merge_radius
        self.semantic = semantic
        self.semantic_min_score = semantic_min_score
        self.semantic_min_margin = semantic_min_margin

    async def ingest_observation(
        self,
        observation: Observation,
        merge_radius: float | None = None,
    ) -> SpatialEntity:
        self.db.add_observation(observation)
        draft = await self.interpreter.memory_from_observation(
            observation.transcript, observation.detected_label
        )
        source = "fused" if observation.transcript and observation.detected_label else (
            "human" if observation.transcript else "vision"
        )
        confidence = (
            observation.detection_confidence
            if observation.detection_confidence is not None
            else 1.0
        )
        observed_metadata = self._observation_metadata(observation)
        radius = self.merge_radius if merge_radius is None else merge_radius
        instance_signature = observed_metadata.get("instance_signature")
        existing = self._find_existing(
            observation.scene_id,
            draft.name,
            observation.position,
            radius=radius,
            instance_signature=instance_signature,
        )
        if (
            existing is None
            and observation.detected_label
            and normalize(observation.detected_label) != normalize(draft.name)
        ):
            existing = self._find_existing(
                observation.scene_id,
                observation.detected_label,
                observation.position,
                radius=radius,
                instance_signature=instance_signature,
            )
        if existing is not None:
            merged_metadata = self._merge_provenance(
                existing.metadata,
                draft.metadata,
                observed_metadata,
            )
            position = self._merged_position(
                existing,
                observation,
                source=source,
                new_count=int(observed_metadata.get("observation_count", 1)),
            )
            explicit_human_name = draft.metadata.get("human_explicit_name")
            renamed = bool(
                explicit_human_name
                and normalize(existing.name) != normalize(draft.name)
            )
            aliases = set(existing.aliases) | set(draft.aliases)
            if renamed:
                aliases.add(existing.name)

            updated = existing.model_copy(update={
                "name": draft.name if renamed else existing.name,
                "position": position,
                "bbox": observation.bbox or existing.bbox,
                "parent_id": observation.parent_id or existing.parent_id,
                "relation_to_parent": observation.relation_to_parent or existing.relation_to_parent,
                "description": draft.description or existing.description,
                "tags": sorted(set(existing.tags) | set(draft.tags)),
                "aliases": sorted(aliases),
                "metadata": merged_metadata,
                "confidence": max(existing.confidence, confidence),
                "source": self._merge_source(existing.source, source),
                "last_observed_at": observation.created_at,
            })
            return self.db.upsert_entity(updated)

        metadata = dict(draft.metadata)
        metadata.update(observed_metadata)
        entity = SpatialEntity(
            scene_id=observation.scene_id,
            name=draft.name,
            kind=observation.kind,
            position=observation.position,
            bbox=observation.bbox,
            parent_id=observation.parent_id,
            relation_to_parent=observation.relation_to_parent,
            description=draft.description,
            tags=draft.tags,
            aliases=draft.aliases,
            metadata=metadata,
            confidence=confidence,
            source=source,
            last_observed_at=observation.created_at,
        )
        return self.db.upsert_entity(entity)

    @staticmethod
    def _merge_source(existing: str, incoming: str) -> str:
        priority = {
            "ai": 0,
            "vision": 1,
            "human": 2,
            "fused": 3,
        }
        return (
            existing
            if priority.get(existing, 0) >= priority.get(incoming, 0)
            else incoming
        )

    @staticmethod
    def _observation_metadata(observation: Observation) -> dict:
        metadata = dict(observation.metadata)
        metadata["observation_count"] = max(
            1,
            int(metadata.get("observation_count", 1)),
        )

        frame_indices = {
            int(value)
            for value in metadata.get("frame_indices", [])
            if value is not None
        }
        if observation.frame_index is not None:
            frame_indices.add(int(observation.frame_index))
        if frame_indices:
            metadata["frame_indices"] = sorted(frame_indices)

        scores = [
            float(value)
            for value in metadata.get("vision_scores", [])
            if value is not None
        ]
        aggregated_count = max(
            1,
            int(metadata.get("observation_count", 1)),
        )
        if observation.detection_confidence is not None and not scores:
            scores.append(float(observation.detection_confidence))
        if scores:
            metadata["vision_scores"] = scores[-100:]

        labels = {
            str(value)
            for value in metadata.get("detected_labels", [])
            if value
        }
        if observation.detected_label:
            labels.add(observation.detected_label)
        if labels:
            metadata["detected_labels"] = sorted(labels)

        paths = {
            str(value)
            for value in metadata.get("image_paths", [])
            if value
        }
        if observation.image_path:
            paths.add(observation.image_path)
        if paths:
            metadata["image_paths"] = sorted(paths)[-100:]

        if observation.detection_bbox_xyxy is not None:
            metadata["last_detection_bbox_xyxy"] = [
                float(value)
                for value in observation.detection_bbox_xyxy
            ]
        return metadata

    @staticmethod
    def _merge_provenance(
        existing: dict,
        draft: dict,
        observed: dict,
    ) -> dict:
        merged = dict(existing)
        merged.update(draft)

        existing_count = max(1, int(existing.get("observation_count", 1)))
        observed_count = max(1, int(observed.get("observation_count", 1)))
        merged["observation_count"] = existing_count + observed_count

        for key in ("frame_indices", "detected_labels", "image_paths"):
            values = {
                value
                for source in (existing.get(key, []), observed.get(key, []))
                for value in source
            }
            if values:
                merged[key] = sorted(values)

        scores = [
            float(value)
            for source in (
                existing.get("vision_scores", []),
                observed.get("vision_scores", []),
            )
            for value in source
        ]
        if scores:
            merged["vision_scores"] = scores[-100:]
            merged["vision_score_mean"] = sum(scores) / len(scores)
            merged["vision_score_max"] = max(scores)

        for key, value in observed.items():
            if key not in {
                "observation_count",
                "frame_indices",
                "detected_labels",
                "image_paths",
                "vision_scores",
            }:
                merged[key] = value
        return merged

    @staticmethod
    def _merged_position(
        existing: SpatialEntity,
        observation: Observation,
        source: str,
        new_count: int,
    ) -> Vec3:
        if existing.source == "human" and source == "vision":
            return existing.position

        is_auto_index = bool(
            observation.metadata.get("auto_index")
            or existing.metadata.get("auto_index")
        )
        if not is_auto_index:
            return observation.position

        existing_count = max(
            1,
            int(existing.metadata.get("observation_count", 1)),
        )
        total = existing_count + max(1, new_count)
        weight_old = existing_count / total
        weight_new = max(1, new_count) / total
        return Vec3(
            x=existing.position.x * weight_old + observation.position.x * weight_new,
            y=existing.position.y * weight_old + observation.position.y * weight_new,
            z=existing.position.z * weight_old + observation.position.z * weight_new,
        )

    async def query(self, scene_id: str, text: str, limit: int = 5) -> QueryResult:
        limit = max(1, min(limit, 10))
        draft = await self.interpreter.query(text)
        entities = [e for e in self.db.list_entities(scene_id) if is_searchable(e)]
        if draft.metadata:
            entities = [e for e in entities if all(
                str(e.metadata.get(k, "")) == str(v) for k, v in draft.metadata.items()
            )]
        if not normalize(draft.search_text):
            return QueryResult(query=text, scene_id=scene_id, hits=[])
        count = max(len(entities), 2)
        lexical = search_entities(draft.search_text, entities, limit=count)
        semantic: list[SearchHit] = []
        mode = "lexical"
        if self.semantic is not None and entities:
            try:
                semantic = await asyncio.to_thread(
                    self.semantic.search, draft.search_text, entities, limit=count,
                )
                mode = "hybrid"
            except Exception:
                logging.getLogger(__name__).warning("Semantic search unavailable; using lexical search")
                mode = "lexical_fallback"

        # Gate raw cosine scores before fusion; a blended score is not confidence.
        semantic = [h for h in semantic if math.isfinite(h.score)]
        raw = {h.entity.id: h.score for h in semantic}
        accepted = [h for h in semantic if h.score >= self.semantic_min_score]
        hits = self._merge_hits(lexical, accepted, limit=count)
        lexical_ids = {h.entity.id for h in lexical}
        semantic_only = bool(hits and hits[0].entity.id not in lexical_ids)
        ambiguous = False
        if semantic_only:
            top = hits[0]
            # Compare BEFORE thresholding/truncation: limit=1 must not bypass abstention.
            rivals = [h for h in semantic if h.entity.id != top.entity.id]
            rivals.sort(key=lambda h: h.score, reverse=True)
            ambiguous = bool(rivals and raw[top.entity.id] - rivals[0].score < self.semantic_min_margin)
            if ambiguous:
                hits = [top] + [h for h in rivals
                               if raw[top.entity.id] - h.score < self.semantic_min_margin]
        elif len(hits) > 1:
            ambiguous = hits[0].score - hits[1].score < self.semantic_min_margin
            if ambiguous:
                hits = [h for h in hits if hits[0].score - h.score < self.semantic_min_margin]
        # Keep candidates separate: legacy clients must not auto-select an uncertain hit.
        status = "ambiguous" if ambiguous else "matched" if hits else "not_found"
        return QueryResult(
            query=text, scene_id=scene_id,
            hits=[] if ambiguous else hits[:limit],
            candidates=hits[:max(2, limit)] if ambiguous else [],
            target=hits[0].entity if status == "matched" else None,
            status=status, search_mode=mode,
        )

    @staticmethod
    def _merge_hits(
        lexical: list[SearchHit],
        semantic: list[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        merged: dict[str, SearchHit] = {}
        for hit in lexical:
            merged[hit.entity.id] = hit.model_copy(deep=True)

        for hit in semantic:
            current = merged.get(hit.entity.id)
            semantic_score = hit.score * 0.88
            if current is None:
                merged[hit.entity.id] = hit.model_copy(
                    update={"score": semantic_score}
                )
                continue

            combined = max(
                current.score,
                min(1.0, 0.60 * current.score + 0.55 * hit.score),
            )
            merged[hit.entity.id] = current.model_copy(update={
                "score": combined,
                "reasons": list(dict.fromkeys(current.reasons + hit.reasons)),
            })

        hits = list(merged.values())
        hits.sort(key=lambda h: (h.score, h.entity.confidence), reverse=True)
        return hits[:limit]

    def _find_existing(
        self,
        scene_id: str,
        name: str,
        position,
        radius: float | None = None,
        instance_signature: str | None = None,
    ) -> SpatialEntity | None:
        radius = self.merge_radius if radius is None else radius
        candidates = []
        wanted = normalize(name)
        for entity in self.db.list_entities(scene_id):
            identity_terms = {
                normalize(entity.name),
                *(normalize(value) for value in entity.aliases),
                *(normalize(value) for value in entity.tags),
            }
            if wanted not in identity_terms:
                continue

            existing_signature = entity.metadata.get("instance_signature")
            if (
                instance_signature
                and existing_signature
                and str(existing_signature) != str(instance_signature)
            ):
                continue

            distance = math.sqrt(
                (entity.position.x - position.x) ** 2
                + (entity.position.y - position.y) ** 2
                + (entity.position.z - position.z) ** 2
            )
            if distance <= radius:
                signature_bonus = (
                    -radius
                    if instance_signature
                    and existing_signature
                    and str(existing_signature) == str(instance_signature)
                    else 0.0
                )
                candidates.append((distance + signature_bonus, distance, entity))

        return (
            min(candidates, key=lambda item: (item[0], item[1]))[2]
            if candidates
            else None
        )

    @staticmethod
    def _filter_metadata(hits: list[SearchHit], wanted: dict[str, str]) -> list[SearchHit]:
        filtered = []
        for hit in hits:
            if all(str(hit.entity.metadata.get(k, "")) == str(v) for k, v in wanted.items()):
                filtered.append(hit)
        return filtered or hits
