"""Sentence-transformer embeddings for semantic retrieval.

This module intentionally has no lexical or hashed fallback. Benchmark and UI retrieval
must use the configured sentence-transformer model so dense results are real semantic
Chroma results instead of a development approximation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable


class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", *, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    @lru_cache(maxsize=1)
    def _load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for dense retrieval. Install requirements.txt.") from exc

        return SentenceTransformer(self.model_name, device=self.device)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        model = self._load_sentence_transformer()
        encoded = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, row)) for row in encoded]
