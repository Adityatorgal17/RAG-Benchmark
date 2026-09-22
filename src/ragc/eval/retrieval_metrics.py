"""Local retrieval and faithfulness metrics for the benchmark.

These metrics are deterministic and do not depend on an external judge. They support
pre-flight checks and local evaluation before the configured pointwise judge is used.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re

from ragc.eval.nli_faith import NLIFaith


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _snippet_in_text(snippet: str, text: str) -> bool:
    return _normalize_whitespace(snippet) in _normalize_whitespace(text)


def _chunk_lookup_from_result(result: dict, chunk_store: dict[str, list[str]]) -> list[str]:
    ids: list[str] = []
    for item in result.get("final_evidence", []) or []:
        chunk_id = item.get("chunk_id")
        if chunk_id:
            ids.append(chunk_id)
    for item in result.get("candidate_ids", []) or []:
        if item not in ids:
            ids.append(item)
    return ids


def _texts_for_chunk_ids(chunk_ids: list[str], chunk_store: dict[str, list[str]]) -> list[str]:
    texts: list[str] = []
    for chunk_id in chunk_ids:
        value = chunk_store.get(chunk_id, [])
        texts.extend(value if isinstance(value, list) else [str(value)])
    return texts


def _all_store_texts(chunk_store: dict[str, list[str]]) -> list[str]:
    texts: list[str] = []
    for value in chunk_store.values():
        texts.extend(value if isinstance(value, list) else [str(value)])
    return texts


def _answer_sentences(answer: str) -> list[str]:
    cleaned = re.sub(r"\[[^\[\]]+\]", "", answer or "")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned) if len(s.split()) >= 6]


def compute_question_metrics(question: dict, result: dict, chunk_store: dict[str, list[str]], *, nli: NLIFaith | None = None) -> dict:
    """Compute deterministic retrieval and faithfulness metrics for one question."""
    gold_evidence = question.get("gold_evidence") or []
    candidate_ids = result.get("candidate_ids") or []
    final_evidence = result.get("final_evidence") or []
    final_ids = [item.get("chunk_id") for item in final_evidence if item.get("chunk_id")]
    candidate_texts = _texts_for_chunk_ids(candidate_ids, chunk_store)
    if candidate_ids and not candidate_texts:
        candidate_texts = _all_store_texts(chunk_store)

    candidate_recall = 0.0
    if gold_evidence:
        found = 0
        for ev in gold_evidence:
            snippet = ev.get("snippet", "")
            doc_id = ev.get("doc_id")
            if any(_snippet_in_text(snippet, text) for text in candidate_texts):
                found += 1
        candidate_recall = found / len(gold_evidence)

    context_recall = 0.0
    if gold_evidence:
        found = 0
        for ev in gold_evidence:
            snippet = ev.get("snippet", "")
            doc_id = ev.get("doc_id")
            for item in final_evidence:
                if item.get("chunk_id") and _snippet_in_text(snippet, (item.get("text") or "") + " " + (item.get("header") or "")):
                    found += 1
                    break
        context_recall = found / len(gold_evidence)

    context_precision = 0.0
    if final_evidence:
        gold_doc_ids = {ev.get("doc_id") for ev in gold_evidence if ev.get("doc_id")}
        precision_hits = 0
        for item in final_evidence:
            chunk_id = item.get("chunk_id") or ""
            doc_id = chunk_id.rsplit("#", 1)[0] if "#" in chunk_id else item.get("doc_id", "")
            if doc_id in gold_doc_ids:
                precision_hits += 1
            elif item.get("source_type") in question.get("gold_source_types", []):
                precision_hits += 1
        context_precision = precision_hits / len(final_evidence)

    faithfulness = 0.0
    if final_evidence:
        model = nli or NLIFaith()
        evidence_texts = [((item.get("header") or "") + " " + (item.get("text") or "")).strip() for item in final_evidence]
        scored = []
        for sentence in _answer_sentences(result.get("answer") or ""):
            scored.append(max((model.entail_prob(text, sentence) for text in evidence_texts if text), default=0.0))
        faithfulness = sum(scored) / len(scored) if scored else 0.0

    return {
        "qid": question.get("qid"),
        "candidate_recall": round(candidate_recall, 4),
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
        "faithfulness": round(faithfulness, 4),
    }


def summarize_metrics(records: list[dict]) -> dict:
    """Average deterministic local metrics across records."""
    if not records:
        return {"candidate_recall": 0.0, "context_recall": 0.0, "context_precision": 0.0, "faithfulness": 0.0}

    def avg(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in records) / len(records)

    return {
        "candidate_recall": round(avg("candidate_recall"), 4),
        "context_recall": round(avg("context_recall"), 4),
        "context_precision": round(avg("context_precision"), 4),
        "faithfulness": round(avg("faithfulness"), 4),
    }


def write_local_metrics_csv(records: list[dict], output_path: str | Path) -> Path:
    """Write a CSV file of local metric records."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["qid", "candidate_recall", "context_recall", "context_precision", "faithfulness"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, 0.0) for key in fieldnames})
    return path
