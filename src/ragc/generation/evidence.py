"""Evidence formatting for final-generation prompts.

This module renders retrieved chunks into a compact evidence block that fits the
context budget while preserving citations and source metadata. It is the adapter between
retrieval results and the generator prompt.
"""

from __future__ import annotations

from ragc.schemas import EvidenceItem


def build_evidence(items: list[EvidenceItem], max_tokens: int = 2000) -> tuple[str, list[EvidenceItem]]:
    if not items:
        return "(none)", []

    block_parts: list[str] = []
    kept: list[EvidenceItem] = []
    current_tokens = 0

    for item in items:
        text = item.text or ""
        note = f"\nNOTE: {item.note}" if item.note else ""
        rendered = (
            f"[{item.chunk_id}] ({item.source_type} | {item.header})\n"
            f"{text}{note}"
        )

        est = max(1, len(rendered) // 4)
        if current_tokens + est > max_tokens and kept:
            break
        block_parts.append(rendered)
        kept.append(item)
        current_tokens += est

    return "\n\n".join(block_parts), kept
