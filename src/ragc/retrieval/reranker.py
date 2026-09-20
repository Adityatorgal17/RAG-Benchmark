"""Cross-encoder reranker for relevance scoring.

This module scores question/chunk pairs using a lightweight local relevance model.
It is used to reorder retrieved candidates before final generation and can fall back to
an internal lexical heuristic when a real model is unavailable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9._\-]*[a-z0-9]|[a-z0-9]", (text or "").lower())


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", *, batch_size: int = 16, max_length: int = 384, verbose: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose
        self._tokenizer = None
        self._model = None

        if model_name != "dummy":
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self._model.eval()
            except Exception:
                self.model_name = "dummy"

    def _lexical_score(self, query: str, passage: str) -> float:
        q_tokens = set(_tokenize(query))
        p_tokens = set(_tokenize(passage))
        if not q_tokens or not p_tokens:
            return 0.0
        overlap = len(q_tokens & p_tokens)
        union = len(q_tokens | p_tokens)
        lexical = overlap / max(1, union)
        shared_bigram_bonus = 0.15 * min(1.0, overlap / max(1, len(q_tokens)))
        return min(1.0, lexical + shared_bigram_bonus)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        if self.model_name == "dummy":
            return [self._lexical_score(q, p) for q, p in pairs]

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if self._tokenizer is None or self._model is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.eval()

            scores: list[float] = []
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                qs = [q for q, _ in batch]
                ps = [p for _, p in batch]
                with torch.no_grad():
                    enc = self._tokenizer(qs, ps, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                    logits = self._model(**enc).logits
                    probs = torch.sigmoid(logits[:, 0]).tolist()
                scores.extend(float(v) for v in probs)
            return scores
        except Exception:
            return [self._lexical_score(q, p) for q, p in pairs]
