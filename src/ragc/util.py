"""Small helper utilities used across the project.

This module provides text normalization, sentence splitting, RRF score aggregation,
and basic token counting. These are shared low-level utilities that keep the pipeline
logic consistent and reduce duplicate logic across modules.
"""

import re
from collections.abc import Iterable


def token_estimate(text: str) -> int:
    return max(0, len(text or "") // 4)


def split_sentences(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    segments = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [part.strip() for part in segments if part and part.strip()]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def rrf_rank(rank_lists: Iterable[Iterable[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
