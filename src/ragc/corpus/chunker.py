"""Chunking logic for turning documents into retrieval-ready segments.

This file splits raw source text into section-aware, overlap-aware chunks so that the
vector and BM25 indexes receive consistent document windows. It is a core preprocessing
step that directly affects retrieval quality and downstream evaluation.
"""

from __future__ import annotations

import re
from typing import Any


_WORD_RE = re.compile(r"\b\w+\b")


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_name = "Overview"
    current_lines: list[str] = []
    for line in (text or "").splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_name, "\n".join(current_lines).strip()))
            current_name = line[3:].strip() or "Overview"
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_name, "\n".join(current_lines).strip()))
    return sections or [("Overview", (text or "").strip())]


def _split_paragraphs(section_text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", section_text.strip()) if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]


def _extract_cve_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"CVE-\d{4}-\d{4,7}", text or "")))


def _append_candidate(chunks: list[dict[str, Any]], doc: dict[str, Any], section_name: str, words: list[str], ordinal: int, doc_cves: set[str]) -> int:
    text = " ".join(words).strip()
    if not text:
        return ordinal
    final_cves = set(doc_cves)
    final_cves.update(_extract_cve_ids(text))
    final_cves.update(_extract_cve_ids(doc.get("title", "")))
    chunks.append({
        "chunk_id": f"{doc['doc_id']}#{ordinal:03d}",
        "doc_id": doc["doc_id"],
        "source_type": doc.get("source_type", ""),
        "title": doc.get("title", ""),
        "section": section_name,
        "url": doc.get("url", ""),
        "date": doc.get("date"),
        "cve_ids": sorted(final_cves),
        "header": f"{doc.get('title', '')} - {section_name}",
        "text": text,
        "n_words": len(words),
    })
    return ordinal + 1


def chunk_document(doc: dict[str, Any], *, target_words: int = 180, overlap_words: int = 30, min_words: int = 25) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    ordinal = 0
    doc_cves = set(doc.get("cve_ids", []))

    for section_name, section_text in _split_sections(doc.get("text") or ""):
        sentence_tokens: list[str] = []
        for paragraph in _split_paragraphs(section_text):
            sentence_tokens.extend(_split_sentences(paragraph))
        if not sentence_tokens:
            sentence_tokens = [section_text.strip()] if section_text.strip() else []

        words: list[str] = []
        for sentence in sentence_tokens:
            words.extend(_WORD_RE.findall(sentence))

        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + target_words, len(words))
            window = words[start:end]
            if len(window) < min_words and chunks:
                previous = chunks[-1]
                previous["text"] = f"{previous['text']} {' '.join(window)}".strip()
                previous["n_words"] = _count_words(previous["text"])
                previous["cve_ids"] = sorted(set(previous["cve_ids"]) | set(_extract_cve_ids(previous["text"])))
                break
            if start > 0 and len(window) > target_words:
                window = window[:target_words]
            ordinal = _append_candidate(chunks, doc, section_name, window, ordinal, doc_cves)
            if end >= len(words):
                break
            start = max(0, end - overlap_words)

    return chunks
