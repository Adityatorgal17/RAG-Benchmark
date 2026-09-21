"""Fetch corpus sources and build processed corpus artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragc.corpus.build_index import CorpusBuilder
from ragc.corpus.fetch import CorpusFetcher, dry_run_manifest
from ragc.eval.make_questions import build_question_catalog, write_question_file
from ragc.eval.validate_questions import validate_question_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch, verify, and materialize the benchmark corpus.")
    parser.add_argument("--dry-run", action="store_true", help="List manifest targets without downloading.")
    parser.add_argument("--check-urls", action="store_true", help="Check vendor/threat-intel URL health.")
    parser.add_argument("--force", action="store_true", help="Re-download sources even if raw files exist.")
    parser.add_argument("--no-index", action="store_true", help="Fetch and chunk only; skip index build.")
    parser.add_argument("--skip-questions", action="store_true", help="Do not generate data/questions/questions.jsonl after chunking.")
    return parser


def _write_questions(chunks_path: Path) -> dict[str, object]:
    questions_path = ROOT / "data" / "questions" / "questions.jsonl"
    rows = build_question_catalog(chunks_path=chunks_path)
    write_question_file(rows, questions_path)
    status = validate_question_file(questions_path)
    return {"path": str(questions_path), "rows": len(rows), "valid": bool(status.get("ok")), "errors": status.get("errors", [])}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        rows = dry_run_manifest()
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.get("source_type", "unknown")] = counts.get(row.get("source_type", "unknown"), 0) + 1
        print(json.dumps({"sources": len(rows), "counts": counts}, indent=2, sort_keys=True))
        return 0

    if args.check_urls:
        print(json.dumps(CorpusFetcher().check_manifest_urls(), indent=2, sort_keys=True))
        return 0

    builder = CorpusBuilder()
    if args.no_index:
        docs, chunks, raw_paths = builder.build_documents(force=args.force)
        lock_path = builder.build_lock(raw_paths)
        payload = {"raw_files": len(raw_paths), "documents": len(docs), "chunks": len(chunks), "lock_path": str(lock_path)}
        if not args.skip_questions:
            payload["questions"] = _write_questions(builder.chunks_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    summary = builder.build_all(force=args.force)
    payload = dict(summary.__dict__)
    if not args.skip_questions:
        payload["questions"] = _write_questions(builder.chunks_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
