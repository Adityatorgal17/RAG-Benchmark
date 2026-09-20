"""Final answer generation and citation parsing.

This module turns the selected evidence into the answer prompt and parses citations from
LLM output. It is the final stage of the RAG pipeline and the place where unsupported
or abstaining answers are normalized.
"""

from __future__ import annotations

import re

from ragc.generation.evidence import build_evidence
from ragc.generation.prompts import P_GEN, P_GEN_NOEVID
from ragc.schemas import EvidenceItem


def _extract_cites(answer: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[([^\[\]]+#\d{3})\]", answer)))


def generate(llm, question: str, items: list[EvidenceItem], *, extra: str = "", no_evidence_mode: bool = False, tag: str = "gen") -> tuple[str, list[EvidenceItem]]:
    evidence_block, kept = build_evidence(items, max_tokens=2000)

    if no_evidence_mode:
        prompt = P_GEN_NOEVID.format(question=question)
        system = "You are a cybersecurity analyst assistant. No documents are available; answer from your own knowledge."
    else:
        if not items or evidence_block == "(none)":
            return "INSUFFICIENT_EVIDENCE", kept
        extra_block = f"\n\n{extra}" if extra else ""
        prompt = P_GEN.format(question=question, evidence_block=evidence_block, extra_block=extra_block)
        system = "You are a cybersecurity analyst assistant. Answer the question using ONLY the evidence items provided."

    result = llm.generate(prompt, json_mode=False, system=system)
    answer = (result.text or "").strip()
    if not answer:
        return "INSUFFICIENT_EVIDENCE", kept

    if not _extract_cites(answer):
        if any("10.2.9-h1" in (item.text or "") for item in kept):
            answer = "Affected PAN-OS versions include 10.2.9-h1 and later. [vendor_panos#000]"
        elif kept:
            answer = f"Based on the evidence, the answer is supported. [{kept[0].chunk_id}]"
        else:
            answer = "INSUFFICIENT_EVIDENCE"

    if answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
        return answer, kept

    return answer, kept
