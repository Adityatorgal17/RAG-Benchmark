"""Validation utilities for the benchmark question catalog.

The project freezes its question set once the authoring pass is complete. This module
validates the JSONL file against the benchmark rules and can also check snippets against
actual chunk text when the corpus has been materialized.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EXPECTED_COUNTS = {
    "single_fact": 7,
    "multi_claim": 7,
    "multi_hop": 7,
    "cross_source": 7,
    "conflict": 6,
    "unanswerable": 6,
}

EXPECTED_QID_SEQUENCE = [
    "d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10",
    "t01", "t02", "t03", "t04", "t05", "t06", "t07", "t08", "t09", "t10",
    "t11", "t12", "t13", "t14", "t15", "t16", "t17", "t18", "t19", "t20",
    "t21", "t22", "t23", "t24", "t25", "t26", "t27", "t28", "t29", "t30",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _load_chunk_lookup(chunks_path: Path) -> dict[str, list[str]]:
    if not chunks_path.exists():
        return {}

    lookup: dict[str, list[str]] = {}
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = entry.get("doc_id")
            if doc_id:
                lookup.setdefault(doc_id, []).append((entry.get("header") or "") + "\n" + (entry.get("text") or ""))
    return lookup


def validate_question_file(path: str | Path) -> dict:
    """Validate a question JSONL file and return a summary dictionary."""
    file_path = Path(path)
    results: list[dict] = []
    if not file_path.exists():
        return {"ok": False, "error": f"Missing question file: {file_path}", "total": 0, "counts": {}}

    with file_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"Invalid JSON at line {lineno}: {exc}", "total": 0, "counts": {}}
            results.append(row)

    counts = {k: 0 for k in EXPECTED_COUNTS}
    seen_qids = set()
    ok = True
    errors: list[str] = []

    observed_qids = []
    for row in results:
        qid = row.get("qid")
        if not qid or not re.fullmatch(r"(?:d|t)\d{2}", str(qid)):
            ok = False
            errors.append(f"Bad qid: {qid!r}")
            continue
        observed_qids.append(str(qid))
        if qid in seen_qids:
            ok = False
            errors.append(f"Duplicate qid: {qid}")
        seen_qids.add(qid)

        split = row.get("split")
        if split not in {"dev", "test"} or (qid.startswith("d") and split != "dev") or (qid.startswith("t") and split != "test"):
            ok = False
            errors.append(f"Split mismatch for {qid}: {split!r}")

        qtype = row.get("type")
        if qtype not in EXPECTED_COUNTS:
            ok = False
            errors.append(f"Unknown question type for {qid}: {qtype!r}")
            continue
        counts[qtype] += 1

        answerable = bool(row.get("answerable", False))
        if answerable:
            if row.get("gold_answer") in (None, ""):
                ok = False
                errors.append(f"Missing gold answer for {qid}")
            if not row.get("gold_claims"):
                ok = False
                errors.append(f"Missing gold claims for {qid}")
            if not row.get("gold_evidence"):
                ok = False
                errors.append(f"Missing gold evidence for {qid}")
            claims = row.get("gold_claims") or []
            claim_ids = []
            for claim in claims:
                claim_id = claim.get("id")
                claim_ids.append(claim_id)
                if not claim_id:
                    ok = False
                    errors.append(f"Missing claim id for {qid}")
            if len(set(claim_ids)) != len(claim_ids):
                ok = False
                errors.append(f"Duplicate claim ids for {qid}")
        else:
            if row.get("gold_answer") != "INSUFFICIENT_EVIDENCE":
                ok = False
                errors.append(f"Unanswerable gold answer must be INSUFFICIENT_EVIDENCE for {qid}")
            if row.get("gold_claims"):
                ok = False
                errors.append(f"Unanswerable questions should not have gold claims for {qid}")
            if row.get("gold_evidence"):
                ok = False
                errors.append(f"Unanswerable questions should not have gold evidence for {qid}")

        for ev in row.get("gold_evidence") or []:
            if not ev.get("doc_id") or not ev.get("snippet"):
                ok = False
                errors.append(f"Malformed gold evidence for {qid}")

    for qtype, expected in EXPECTED_COUNTS.items():
        actual = counts.get(qtype, 0)
        if actual != expected:
            ok = False
            errors.append(f"Type count mismatch for {qtype}: expected {expected}, got {actual}")

    if len(results) != 40:
        ok = False
        errors.append(f"Expected 40 questions, got {len(results)}")

    if observed_qids != EXPECTED_QID_SEQUENCE:
        ok = False
        errors.append("Question qids are not in the plan-ordered sequence: expected d01..d10 then t01..t30")

    data_root = file_path.parent.parent if file_path.parent.name == "questions" else Path("data")
    chunks_path = data_root / "processed" / "chunks.jsonl"
    chunk_lookup = _load_chunk_lookup(chunks_path)
    if chunk_lookup:
        for row in results:
            for ev in row.get("gold_evidence") or []:
                doc_id = ev.get("doc_id")
                snippet = ev.get("snippet")
                if not doc_id or not snippet:
                    continue
                found = False
                for text in chunk_lookup.get(doc_id, []):
                    if _normalize(snippet) in _normalize(text):
                        found = True
                        break
                if not found:
                    ok = False
                    errors.append(f"Snippet for {row.get('qid')} not found in doc_id {doc_id}")

    return {"ok": ok, "total": len(results), "counts": counts, "errors": errors}


def main() -> None:
    file_path = Path("data/questions/questions.jsonl")
    summary = validate_question_file(file_path)
    if summary["ok"]:
        print(f"Validated {summary['total']} questions: {summary['counts']}")
    else:
        print(f"INVALID question file: {file_path}")
        for err in summary.get("errors", []):
            print(f"- {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
