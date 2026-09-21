"""Facade for retrieving chunks from the corpus.

This file provides the main entry point used by the rest of the project to run dense,
BM25, and hybrid retrieval on the same chunk set. It centralizes chunk loading and makes
it easier to swap retrieval strategies without changing system-level code.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from ragc.retrieval.bm25 import BM25Index
from ragc.retrieval.embeddings import TextEmbedder
from ragc.schemas import Hit

try:
    import chromadb
except ImportError as exc:  # pragma: no cover - dependency should exist in project env
    raise RuntimeError("chromadb is required for dense retrieval. Install requirements.txt.") from exc


class Retriever:
    def __init__(self, cfg: Any, *, chunks_path: str | Path | None = None):
        self.cfg = cfg
        paths = getattr(cfg, "paths", type("P", (), {"chunks": "data/processed/chunks.jsonl", "chroma": "data/index/chroma", "bm25": "data/index/bm25.pkl"})())
        self.chunks_path = Path(chunks_path) if chunks_path is not None else Path(getattr(paths, "chunks", "data/processed/chunks.jsonl"))
        self.chroma_path = Path(getattr(paths, "chroma", "data/index/chroma"))
        self.bm25_path = Path(getattr(paths, "bm25", "data/index/bm25.pkl"))
        embedding_cfg = getattr(cfg, "embedding", None)
        self._embedder = TextEmbedder(getattr(embedding_cfg, "model", "BAAI/bge-small-en-v1.5"))
        self._chunks = self._load_chunks(self.chunks_path)
        if not self._chunks:
            raise FileNotFoundError(f"Missing or empty chunks file: {self.chunks_path}. Run python scripts/build_all.py first.")
        self._chunk_lookup = {chunk["chunk_id"]: chunk for chunk in self._chunks}
        self._bm25_index = self._load_bm25_index(self.bm25_path)
        self._chroma_collection = self._load_chroma_collection(self.chroma_path)

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

    def _load_chroma_collection(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing Chroma index directory: {path}. Run python scripts/build_all.py first.")
        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection("cve_corpus")
        if collection.count() == 0:
            raise RuntimeError(f"Chroma collection 'cve_corpus' is empty in {path}. Rebuild indexes.")
        return collection

    def _load_bm25_index(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing BM25 index: {path}. Run python scripts/build_all.py first.")
        with path.open("rb") as fh:
            index = pickle.load(fh)
        if not isinstance(index, BM25Index):
            raise TypeError(f"Unexpected BM25 index payload in {path}")
        return index

    def _dense_from_chroma(self, query: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        query_embedding = self._embedder.embed([query])[0]
        count = int(self._chroma_collection.count() or 0)
        if count <= 0:
            raise RuntimeError("Chroma collection 'cve_corpus' is empty")
        result = self._chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(k * 5, k), count),
            include=["metadatas", "distances"],
        )

        hits: list[Hit] = []
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for chunk_id, metadata, distance in zip(ids, metadatas, distances):
            meta = metadata or {}
            if source_types is not None and meta.get("source_type") not in source_types:
                continue
            score = 1.0 - float(distance or 0.0)
            hits.append(Hit(chunk_id=str(chunk_id), score=score, rank=len(hits) + 1, via="dense"))
            if len(hits) >= k:
                break
        return hits

    def dense(self, query: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        return self._dense_from_chroma(query, k, source_types)

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
