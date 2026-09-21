"""Cross-encoder reranker for relevance scoring.

The benchmark requires the configured transformer reranker. This module does not fall
back to a lexical heuristic because that would change S2/S3/S6c behavior while making
the run look successful.
"""

from __future__ import annotations

from collections.abc import Sequence


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", *, batch_size: int = 16, max_length: int = 384, verbose: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose
        self._tokenizer = None
        self._model = None

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for reranking. Install requirements.txt.") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        import torch

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
