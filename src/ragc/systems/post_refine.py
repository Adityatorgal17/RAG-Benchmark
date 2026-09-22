"""Shared post-retrieval refiner for S2, S3 and the S2 ablations.

This module performs the plan’s exact post-processing pipeline: rerank candidate chunks,
apply a relevance gate keyed by the calibrated tau threshold, and optionally compress each
retained chunk by selecting the strongest sentence fragments while preserving citations.
"""

from __future__ import annotations

import re

from ragc.generation.evidence import build_evidence
from ragc.schemas import EvidenceItem


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", (text or "").strip())
    return [s.strip() for s in raw if s and s.strip()]


def _score_candidates(reranker, queries: list[str], candidates: list[str], retriever, query_of=None, *, cfg=None) -> list[tuple[str, float]]:
    query_of = query_of or {}
    score_map: dict[str, float] = {}
    for chunk_id in candidates:
        chunk = retriever.get_chunk(chunk_id) if retriever is not None else {}
        text = (chunk.get("header", "") + " " + chunk.get("text", "")).strip()
        if not text:
            continue
        active_queries = query_of.get(chunk_id, queries)
        if not active_queries:
            active_queries = queries
        best = 0.0
        for q in active_queries:
            pair = (q, text)
            try:
                scores = reranker.score([pair])
            except TypeError:
                if hasattr(reranker, "score"):
                    scores = reranker.score([pair])
                else:
                    scores = [0.0]
            best = max(best, float(scores[0]) if scores else 0.0)
        score_map[chunk_id] = best
    return sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))


def post_refine(reranker, retriever, queries: list[str], candidates: list[str], query_of: dict[str, list[str]] | None, cfg, *, do_filter=True, do_compress=True) -> list[EvidenceItem]:
    """Apply reranking, filtering, and compression to candidate chunks."""
    ranked = _score_candidates(reranker, queries, candidates, retriever, query_of, cfg=cfg)

    retrieval_cfg = getattr(cfg, "retrieval", None)
    section_name = "s3" if query_of else "s2"
    section_cfg = getattr(retrieval_cfg, section_name, None)
    keep_k = getattr(section_cfg, "keep_k", 6) if section_cfg is not None else 6
    kept = [chunk_id for chunk_id, _ in ranked[:keep_k]]

    tau = float(getattr(getattr(cfg, "post", None), "tau", 0.15))
    min_keep = int(getattr(getattr(cfg, "post", None), "min_keep", 2))
    if do_filter:
        filtered: list[str] = []
        for chunk_id, score in ranked[:keep_k]:
            if score >= tau or len(filtered) < min_keep:
                filtered.append(chunk_id)
        kept = filtered

    items: list[EvidenceItem] = []
    for chunk_id in kept:
        chunk = retriever.get_chunk(chunk_id) if retriever is not None else {}
        if not chunk:
            continue
        items.append(EvidenceItem(chunk_id=chunk_id, header=chunk.get("header", ""), text=chunk.get("text", ""), source_type=chunk.get("source_type", "")))

    if not do_compress:
        return items

    compacted: list[EvidenceItem] = []
    limit = int(getattr(getattr(cfg, "post", None), "compress_tokens", 1500))
    for item in items:
        sentences = _split_sentences(item.text)
        if not sentences:
            compacted.append(item)
            continue
        scored = []
        for sentence in sentences:
            query_scores = []
            for query in queries:
                try:
                    scores = reranker.score([(query, item.header + ": " + sentence)])
                    query_scores.append(float(scores[0]) if scores else 0.0)
                except Exception:
                    query_scores.append(0.0)
            scored.append((sentence, max(query_scores) if query_scores else 0.0))
        scored = sorted(scored, key=lambda x: (-x[1], x[0]))
        selected: list[str] = []
        if scored:
            selected.append(scored[0][0])
        for sentence, _ in scored[1:]:
            if sum(len(s) for s in selected) + len(sentence) <= limit * 4:
                selected.append(sentence)
        rebuilt = " ".join(selected) if selected else item.text
        compacted.append(EvidenceItem(chunk_id=item.chunk_id, header=item.header, text=rebuilt, source_type=item.source_type))

    return compacted
