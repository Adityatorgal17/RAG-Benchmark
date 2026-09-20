"""Facade for retrieving chunks from the corpus.

This file provides the main entry point used by the rest of the project to run dense,
BM25, and hybrid retrieval on the same chunk set. It centralizes chunk loading and makes
it easier to swap retrieval strategies without changing system-level code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragc.retrieval.bm25 import BM25Index
from ragc.retrieval.dense import DenseIndex
from ragc.schemas import Hit


class Retriever:
    def __init__(self, cfg: Any, *, chunks: list[dict[str, Any]] | None = None, chunks_path: str | Path | None = None):
        self.cfg = cfg
        self.chunks_path = Path(chunks_path) if chunks_path is not None else Path(
            getattr(cfg, "paths", type("P", (), {"chunks": "data/processed/chunks.jsonl"})()).chunks
        )
        self._chunks = chunks if chunks is not None else self._load_chunks(self.chunks_path)
        self._chunk_lookup = {chunk["chunk_id"]: chunk for chunk in self._chunks}
        self._dense_index = DenseIndex(cfg, self._chunks)
        self._bm25_index = BM25Index(cfg, self._chunks)

    def _load_chunks(self, path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows

    def dense(self, query: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        return self._dense_index.search(query, k, source_types)

    def bm25(self, query: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        return self._bm25_index.search(query, k, source_types)

    def hybrid(self, query: str, k_dense: int, k_bm25: int, k_out: int, source_types: list[str] | None = None) -> list[Hit]:
        dense_hits = self.dense(query, k_dense, source_types)
        bm25_hits = self.bm25(query, k_bm25, source_types)

        score_map: dict[str, float] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            score_map[hit.chunk_id] = score_map.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)
        for rank, hit in enumerate(bm25_hits, start=1):
            score_map[hit.chunk_id] = score_map.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)

        ordered = sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))[:k_out]
        return [Hit(chunk_id=chunk_id, score=float(score), rank=rank, via="hybrid") for rank, (chunk_id, score) in enumerate(ordered, start=1)]

    def passage(self, passage: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        return self.dense(passage, k, source_types)

    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        return self._chunk_lookup.get(chunk_id, {})
