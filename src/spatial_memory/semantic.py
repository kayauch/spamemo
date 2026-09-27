from __future__ import annotations

from collections import OrderedDict
from threading import RLock
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .models import SearchHit, SpatialEntity
from .search import _metadata_text, is_searchable
from .search_vocabulary import retrieval_hint


class SemanticSearchEngine:
    def __init__(
        self,
        model_id: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._lock = RLock()
        self._vectors = OrderedDict()
        self.tokenizer = None
        self.model = None
        self.load_metrics: dict[str, float | str] = {}

    def load(self) -> None:
        if self.model is not None:
            return
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
        else:
            peak = 0.0
        self.load_metrics = {
            "model": self.model_id,
            "load_seconds": time.perf_counter() - started,
            "load_peak_allocated_gib": peak,
        }

    @staticmethod
    def entity_text(entity: SpatialEntity) -> str:
        parts = [
            entity.name,
            retrieval_hint(entity),
            " ".join(entity.aliases),
            " ".join(entity.tags),
            entity.description or "",
        ]
        parts.append(_metadata_text(entity))
        return " ".join(part for part in parts if part).strip()

    def _embed(self, texts: list[str], prefix: str) -> torch.Tensor:
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None
        prepared = [f"{prefix}: {text}" for text in texts]
        batch = self.tokenizer(
            prepared,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**batch)
            token_embeddings = outputs.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled

    def search(
        self,
        query: str,
        entities: list[SpatialEntity],
        limit: int = 5,
    ) -> list[SearchHit]:
        entities = [e for e in entities if is_searchable(e)]
        if not query.strip() or not entities:
            return []
        texts = [self.entity_text(e) for e in entities]
        # Key by content: editing or deleting a memory cannot leave stale search results.
        with self._lock:
            query_vec = self._embed([query], "query").cpu()
            missing = list(dict.fromkeys(t for t in texts if t not in self._vectors))
            for offset in range(0, len(missing), 32):
                batch = missing[offset:offset + 32]
                vectors = self._embed(batch, "passage").cpu()
                for text, vector in zip(batch, vectors):
                    self._vectors[text] = vector
            entity_vecs = torch.stack([self._vectors[t] for t in texts])
            for text in texts:
                self._vectors.move_to_end(text)
            while len(self._vectors) > 2048:
                self._vectors.popitem(last=False)
            similarities = (entity_vecs @ query_vec[0]).float().tolist()
        hits = [
            SearchHit(
                entity=entity,
                score=max(0.0, min(1.0, float(score))),
                reasons=["semantic"],
            )
            for entity, score in zip(entities, similarities)
        ]
        hits.sort(key=lambda hit: (hit.score, hit.entity.confidence), reverse=True)
        return hits[:limit]
