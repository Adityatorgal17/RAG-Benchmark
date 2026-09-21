"""BM25 sparse retrieval index.

This module builds and searches an inverted sparse index over chunk text so exact
identifiers like CVE numbers and version strings can be retrieved reliably. It is
important for matching precise security terminology that dense embeddings may miss.
"""

from __future__ import annotations

import re
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ragc.schemas import Hit


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._\-]*[a-z0-9]|[a-z0-9]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    def __init__(self, cfg: Any | None = None, chunks: list[dict[str, Any]] | None = None):
        self.cfg = cfg
        self.chunks = list(chunks or [])
        self._tokenized: list[list[str]] = []
        self._index: BM25Okapi | None = None
        if self.chunks:
            self.build(self.chunks)

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = list(chunks)
        self._tokenized = [
            _tokenize(f"{chunk.get('header', '')} {chunk.get('text', '')}")
            for chunk in self.chunks
        ]
        self._index = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, k: int, source_types: list[str] | None = None) -> list[Hit]:
        if not self.chunks or self._index is None:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores = self._index.get_scores(q_tokens)
        ranked = []
        for idx, chunk in enumerate(self.chunks):
            if source_types is not None and chunk.get("source_type") not in source_types:
                continue
            ranked.append((idx, float(scores[idx]), chunk))

        ranked.sort(key=lambda item: (-item[1], item[2]["chunk_id"]))
        hits: list[Hit] = []
        for rank, (idx, score, chunk) in enumerate(ranked[:k], start=1):
            hits.append(Hit(chunk_id=chunk["chunk_id"], score=score, rank=rank, via="bm25"))
        return hits

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        path = Path(path)
        with path.open("rb") as fh:
            return pickle.load(fh)
