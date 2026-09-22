"""In-memory dense retrieval for tests and small chunk sets.

The persisted benchmark path uses Chroma, but tests and small callers may pass chunks
directly to `Retriever`. This index still uses the configured sentence-transformer
embeddings so in-memory retrieval has the same semantic behavior as the persisted path.
"""

from __future__ import annotations

import math
from typing import Any

from ragc.config import load_config
from ragc.retrieval.embeddings import TextEmbedder
from ragc.schemas import Hit


class DenseIndex:
    def __init__(self, cfg: Any | None = None, chunks: list[dict[str, Any]] | None = None):
        self.cfg = cfg or load_config()
        project_cfg = load_config()
        self.chunks = list(chunks or [])
        self.query_prefix = str(self._cfg_get("embedding", "query_prefix", fallback_cfg=project_cfg) or "")
        self._embedder = TextEmbedder(self._cfg_get("embedding", "model", fallback_cfg=project_cfg))
        self._embeddings: list[list[float]] = []
        if self.chunks:
            self.build(self.chunks)

    def _cfg_get(self, *path: str, fallback_cfg):
        missing = object()
        for source in (self.cfg, fallback_cfg):
            current = source
            for key in path:
                if current is None:
                    current = missing
                    break
                if isinstance(current, dict):
                    current = current.get(key, missing)
                else:
                    current = getattr(current, key, missing)
                if current is missing:
                    break
            if current is not missing:
                return current
        raise AttributeError("Missing required config value: " + ".".join(path))

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = list(chunks)
        texts = [f"{chunk.get('header', '')} {chunk.get('text', '')}".strip() for chunk in self.chunks]
        self._embeddings = self._embedder.embed(texts) if texts else []

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        a_norm = math.sqrt(sum(x * x for x in a))
        b_norm = math.sqrt(sum(y * y for y in b))
        if not a_norm or not b_norm:
            return 0.0
        return dot / (a_norm * b_norm)

    def search(self, query: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        if not self.chunks or not self._embeddings:
            return []

        query_text = f"{self.query_prefix}{query}" if self.query_prefix else query
        query_embedding = self._embedder.embed([query_text])[0]
        ranked = []
        for idx, chunk in enumerate(self.chunks):
            if source_types is not None and chunk.get("source_type") not in source_types:
                continue
            score = self._cosine(query_embedding, self._embeddings[idx])
            ranked.append((chunk["chunk_id"], score))

        ranked.sort(key=lambda item: (-item[1], item[0]))
        results: list[Hit] = []
        for rank, (chunk_id, score) in enumerate(ranked[:k], start=1):
            results.append(Hit(chunk_id=chunk_id, score=float(score), rank=rank, via="dense"))
        return results

    def search_passage(self, passage: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        return self.search(passage, k, source_types)
