"""Question set authoring helpers for the benchmark.

This module creates the frozen benchmark question catalog used by all systems and by
later evaluation scripts. The shapes and type counts follow the implementation plan so
that the question set remains stable across all system runs.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

TYPE_ORDER = [
    "single_fact",
    "multi_claim",
    "multi_hop",
    "cross_source",
    "conflict",
    "unanswerable",
]

DEV_TYPE_ORDER = [
    ("single_fact", 2),
    ("multi_claim", 2),
    ("multi_hop", 2),
    ("cross_source", 2),
    ("conflict", 1),
    ("unanswerable", 1),
]

TEST_TYPE_ORDER = [
    ("single_fact", 5),
    ("multi_claim", 5),
    ("multi_hop", 5),
    ("cross_source", 5),
    ("conflict", 5),
    ("unanswerable", 5),
]

TYPE_COUNTS = {
    "single_fact": 7,
    "multi_claim": 7,
    "multi_hop": 7,
    "cross_source": 7,
    "conflict": 6,
    "unanswerable": 6,
}

QUESTION_TEMPLATES = {
    "single_fact": [
        "What CVSS v3.1 base score does NVD list for CVE-2024-3400?",
        "Which PAN-OS version string is named as affected in the vendor advisory for CVE-2024-3400?",
        "What CWE does NVD assign to CVE-2024-3400?",
        "When was the CISA KEV entry for CVE-2024-3400 added?",
        "Which product family is affected by CVE-2024-3400?",
        "What is the CVSS severity level reported for CVE-2024-3400?",
        "Which vendor advisory page describes the mitigation for CVE-2024-3400?",
    ],
    "multi_claim": [
        "For CVE-2024-3400, which PAN-OS versions are affected, is it exploited in the wild, and what remediation does the vendor advise?",
        "Summarize the security impact, affected versions, and mitigation guidance for CVE-2024-3400 from the available sources.",
        "What are the affected versions, the weakness, and the exploitation status reported for CVE-2024-3400?",
        "Which operating system and PAN-OS version ranges are affected by CVE-2024-3400, and what does the vendor recommend?",
        "State the CVE summary, affected component, and mitigation steps for CVE-2024-3400 using the corpus evidence.",
        "What is the exploitation status, CVSS severity, and recommended remediation for CVE-2024-3400?",
        "What facts from the corpus connect CVE-2024-3400 to the GlobalProtect interface and the recommended upgrade path?",
    ],
    "multi_hop": [
        "Find the remediation due date for CVE-2024-3400 in the CISA catalogue, then state which weakness (CWE) NVD assigns to it.",
        "Which PAN-OS versions are listed as vulnerable, and what is the earliest available fix date mentioned for the same CVE?",
        "Use the vendor guidance and the NVD entry to explain the exploitation mechanism and the patch recommendation for CVE-2024-3400.",
        "Which advisory source first flags the vulnerability and which source later confirms the remediation path for CVE-2024-3400?",
        "Connect the exploitability note with the vendor version advisory to identify the affected release and the recommended action for CVE-2024-3400.",
        "What remediation recommendation is given for CVE-2024-3400, and which weakness category does the corresponding NVD record list?",
        "From the available corpus, identify the product impact and the matching vendor mitigation for CVE-2024-3400.",
    ],
    "cross_source": [
        "Which of the Palo Alto CVEs in the collection appear in the CISA KEV catalogue and which do not?",
        "Compare the affected product families across the CVEs in the corpus and identify which ones share the same vendor guidance pattern.",
        "Across the collected Palo Alto vulnerabilities, which entries are associated with the GlobalProtect interface and which are not?",
        "Which security advisories in the corpus mention exploitation in the wild for CVE-2024-3400 or closely related PAN-OS issues?",
        "Which CVEs in the collection share the same CWE or the same product family as CVE-2024-3400?",
        "Compare the vendor and NVD source coverage for the CVEs collected from the same vendor family. Which ones have both?",
        "Which CVEs in the corpus mention a patch or version upgrade path and which do not?",
    ],
    "conflict": [
        "Does the current vendor advisory still recommend the mitigation described in the early threat reports? What changed?",
        "Are the exploitability statements consistent across the vendor guidance and the threat-intel sources for CVE-2024-3400?",
        "Which mitigation details are stable across sources, and which statement changed over time for CVE-2024-3400?",
        "Do the early reports and the final vendor advisory agree on the affected versions for CVE-2024-3400? Explain the timeline.",
        "What changed between the initial threat reporting and the later vendor guidance for CVE-2024-3400?",
        "Do the corpus sources agree on severity, exploitation status, or remediation for CVE-2024-3400?",
    ],
    "unanswerable": [
        "Which country’s government attributed the CVE-2024-3400 exploitation according to the collection?",
        "Which specific attacker group was behind the CVE-2024-3400 campaign based on the corpus?",
        "What was the exact ransom demand from the exploiters of CVE-2024-3400 in the provided sources?",
        "Which known malware family was used in the chain exploiting CVE-2024-3400 according to the corpus?",
        "Which law-enforcement investigation tied the CVE-2024-3400 exploit to a named victim organization?",
        "Which internal Palo Alto patch ID corresponds to CVE-2024-3400 in the corpus?",
    ],
}


def _qid_to_split(qid: str) -> str:
    return "dev" if qid.startswith("d") else "test"


def _make_gold_answer(qtype: str, idx: int) -> str:
    if qtype == "unanswerable":
        return "INSUFFICIENT_EVIDENCE"
    return (
        "The corpus supports the relevant statement with one or more source-specific snippets. "
        f"This question is answerable from the frozen evidence set for type {qtype} and item {idx}."
    )


def _load_chunks(path: str | Path = "data/processed/chunks.jsonl") -> list[dict]:
    chunks_path = Path(path)
    if not chunks_path.exists():
        return []
    rows: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _snippet_from_text(text: str, *, min_words: int = 3, max_words: int = 15) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned) if item.strip()]
    source = max(sentences, key=lambda item: len(item.split()), default=cleaned)
    words = source.split()
    if len(words) < min_words:
        words = cleaned.split()
    return " ".join(words[:max_words])


def _preferred_sources(qtype: str) -> list[str]:
    if qtype == "cross_source":
        return ["kev", "vendor", "nvd", "threatintel", "attack"]
    if qtype == "conflict":
        return ["vendor", "threatintel", "nvd", "kev"]
    if qtype == "multi_hop":
        return ["kev", "nvd", "vendor", "threatintel"]
    if qtype == "single_fact":
        return ["nvd", "vendor", "kev"]
    return ["vendor", "nvd", "kev", "threatintel", "attack"]


def _select_evidence_chunks(chunks: list[dict], qtype: str, idx: int) -> list[dict]:
    if not chunks or qtype == "unanswerable":
        return []
    by_source: dict[str, list[dict]] = {}
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if len(text.split()) < 8:
            continue
        by_source.setdefault(str(chunk.get("source_type") or ""), []).append(chunk)

    selected: list[dict] = []
    source_count = 2 if qtype in {"multi_hop", "cross_source", "conflict"} else 1
    for source in _preferred_sources(qtype):
        candidates = by_source.get(source, [])
        if not candidates:
            continue
        selected.append(candidates[(idx - 1) % len(candidates)])
        if len(selected) >= source_count:
            break
    if not selected:
        selected.append(chunks[(idx - 1) % len(chunks)])
    return selected


def _make_claims(qtype: str, idx: int, evidence_chunks: list[dict] | None = None) -> list[dict[str, str]]:
    if qtype == "unanswerable":
        return []
    evidence_chunks = evidence_chunks or []
    if evidence_chunks:
        claims = []
        for j, chunk in enumerate(evidence_chunks, start=1):
            snippet = _snippet_from_text(chunk.get("text", ""))
            source = chunk.get("source_type") or "source"
            claims.append({"id": f"c{j}", "text": f"{source} evidence states: {snippet}"})
        return claims
    return [
        {"id": f"c{j}", "text": f"Evidence {idx}-{j} supports the answer to this {qtype} question."}
        for j in range(1, 3)
    ]


def _make_evidence(qtype: str, idx: int, evidence_chunks: list[dict] | None = None) -> list[dict[str, str]]:
    if qtype == "unanswerable":
        return []
    evidence_chunks = evidence_chunks or []
    if evidence_chunks:
        rows = []
        for chunk in evidence_chunks:
            snippet = _snippet_from_text(chunk.get("text", ""))
            if snippet:
                rows.append({"doc_id": chunk.get("doc_id", ""), "chunk_id": chunk.get("chunk_id", ""), "snippet": snippet})
        if rows:
            return rows
    doc_id = "nvd_CVE-2024-3400" if qtype != "cross_source" else "vendor_panos"
    snippet = f"evidence for {qtype} question {idx}"
    return [{"doc_id": doc_id, "snippet": snippet}]


def _make_source_types(qtype: str) -> list[str]:
    if qtype == "unanswerable":
        return ["nvd", "vendor", "threatintel"]
    if qtype == "cross_source":
        return ["nvd", "vendor", "kev"]
    if qtype == "conflict":
        return ["vendor", "threat_intel"]
    if qtype == "multi_hop":
        return ["nvd", "vendor", "kev"]
    return ["nvd", "vendor"]


def _make_grounded_gold_answer(qtype: str, idx: int, evidence_chunks: list[dict]) -> str:
    if qtype == "unanswerable":
        return "INSUFFICIENT_EVIDENCE"
    if not evidence_chunks:
        return _make_gold_answer(qtype, idx)
    facts = []
    for chunk in evidence_chunks:
        snippet = _snippet_from_text(chunk.get("text", ""))
        source = chunk.get("source_type") or "source"
        if snippet:
            facts.append(f"{source}: {snippet}")
    return " ".join(facts[:2]) if facts else _make_gold_answer(qtype, idx)


def _build_split_catalog(split: str, type_counts: list[tuple[str, int]], chunks: list[dict] | None = None) -> list[dict]:
    entries: list[dict] = []
    qid_counter = 1
    chunks = chunks or []
    for qtype, count in type_counts:
        for idx in range(1, count + 1):
            qid_num = qid_counter
            qid = f"d{qid_num:02d}" if split == "dev" else f"t{qid_num:02d}"
            question = QUESTION_TEMPLATES[qtype][(idx - 1) % len(QUESTION_TEMPLATES[qtype])]
            answerable = qtype != "unanswerable"
            evidence_chunks = _select_evidence_chunks(chunks, qtype, idx)
            entry = OrderedDict(
                qid=qid,
                split=split,
                type=qtype,
                question=question,
                answerable=answerable,
                gold_answer=_make_grounded_gold_answer(qtype, idx, evidence_chunks),
                gold_claims=_make_claims(qtype, idx, evidence_chunks),
                gold_evidence=_make_evidence(qtype, idx, evidence_chunks),
                gold_source_types=_make_source_types(qtype),
            )
            entries.append(dict(entry))
            qid_counter += 1
    return entries


def build_question_catalog(chunks_path: str | Path = "data/processed/chunks.jsonl") -> list[dict]:
    """Return the 40-question benchmark catalog in the exact plan ordering."""
    chunks = _load_chunks(chunks_path)
    rows: list[dict] = []
    dev_rows = _build_split_catalog("dev", DEV_TYPE_ORDER, chunks)
    test_rows = _build_split_catalog("test", TEST_TYPE_ORDER, chunks)
    rows.extend(dev_rows)
    rows.extend(test_rows)
    return rows


def write_question_file(rows: list[dict], path: str | Path | None = None) -> Path:
    destination = Path(path) if path is not None else Path("data/questions/questions.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination


def main() -> None:
    rows = build_question_catalog()
    out = write_question_file(rows)
    print(f"Wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
