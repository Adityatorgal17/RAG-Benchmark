"""Local retrieval and faithfulness metrics for the benchmark.

These metrics are deterministic and do not depend on an external judge. They support
pre-flight checks and local evaluation before the Groq pointwise judge is used.
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


def compute_question_metrics(question: dict, result: dict, chunk_store: dict[str, list[str]], *, nli: NLIFaith | None = None) -> dict:
    """Compute deterministic retrieval and faithfulness metrics for one question."""
    gold_evidence = question.get("gold_evidence") or []
    candidate_ids = result.get("candidate_ids") or []
    final_evidence = result.get("final_evidence") or []
    final_ids = [item.get("chunk_id") for item in final_evidence if item.get("chunk_id")]

    candidate_recall = 0.0
    if gold_evidence:
        found = 0
        for ev in gold_evidence:
            snippet = ev.get("snippet", "")
            doc_id = ev.get("doc_id")
            matches = chunk_store.get(doc_id, [])
            if any(_snippet_in_text(snippet, text) for text in matches):
                found += 1
            elif any(_snippet_in_text(snippet, text) for chunk_id in candidate_ids for text in chunk_store.get(chunk_id, []) if chunk_id in chunk_store):
                found += 1
        candidate_recall = found / len(gold_evidence)

    context_recall = 0.0
    if gold_evidence:
        found = 0
        for ev in gold_evidence:
            snippet = ev.get("snippet", "")
            doc_id = ev.get("doc_id")
            if doc_id in chunk_store and any(_snippet_in_text(snippet, text) for text in chunk_store[doc_id]):
                found += 1
            else:
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
            chunk_id = item.get("chunk_id")
            if chunk_id and any(doc_id == chunk_id for doc_id in gold_doc_ids):
                precision_hits += 1
            elif item.get("source_type") in question.get("gold_source_types", []):
                precision_hits += 1
        context_precision = precision_hits / len(final_evidence)

    faithfulness = 0.0
    if final_evidence:
        model = nli or NLIFaith()
        scored = []
        for item in final_evidence:
            text = (item.get("header") or "") + " " + (item.get("text") or "")
            if not text.strip():
                continue
            val = model.entail_prob(text, (question.get("question") or "").strip())
            scored.append(val)
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
