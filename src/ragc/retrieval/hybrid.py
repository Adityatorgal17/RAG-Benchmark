"""Reciprocal Rank Fusion for combining dense and sparse retrieval.

This file merges ranked results from multiple retrieval strategies into one final list.
It is the standard way the project combines semantic and keyword retrieval for more
robust candidate selection.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragc.util import rrf_rank


def rrf(rank_lists: Iterable[Iterable[str | int]], k: int = 60) -> list[str | int]:
    return rrf_rank(rank_lists, k=k)


def hybrid_retrieve(rank_lists: Iterable[Iterable[str | int]], k: int = 60) -> list[str | int]:
    return rrf_rank(rank_lists, k=k)
