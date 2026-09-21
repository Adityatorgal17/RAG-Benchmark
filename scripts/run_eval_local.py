"""Compute deterministic local metrics for saved benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragc.config import load_config
from ragc.eval.nli_faith import NLIFaith
from ragc.eval.retrieval_metrics import compute_question_metrics


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_chunks(path: Path) -> dict[str, list[str]]:
    chunks: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        chunks[str(chunk_id)] = [((row.get("header") or "") + "\n" + (row.get("text") or "")).strip()]
    return chunks


def _result_files(results_root: Path, *, system: str | None, split: str) -> list[Path]:
    raw_root = results_root / "raw"
    if not raw_root.exists():
        return []
    paths = sorted((raw_root / system).glob("*.json")) if system else sorted(raw_root.glob("*/*.json"))
    kept: list[Path] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("split", split) == split:
            kept.append(path)
    return kept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local retrieval/context/faithfulness metrics.")
    parser.add_argument("--system", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    questions_path = Path(cfg.paths.questions)
    chunks_path = Path(cfg.paths.chunks)
    results_root = Path(cfg.paths.results)

    if not questions_path.exists():
        raise SystemExit(f"Missing questions file: {questions_path}")
    if not chunks_path.exists():
        raise SystemExit(f"Missing chunks file: {chunks_path}")

    questions = {row["qid"]: row for row in _read_jsonl(questions_path) if row.get("split") == args.split}
    chunk_store = _load_chunks(chunks_path)
    nli_cfg = getattr(cfg, "nli", None)
    nli = NLIFaith(
        model_name=getattr(nli_cfg, "model", "cross-encoder/nli-deberta-v3-xsmall"),
        max_length=int(getattr(nli_cfg, "max_length", 480)),
    )

    records: list[dict] = []
    for path in _result_files(results_root, system=args.system, split=args.split):
        payload = json.loads(path.read_text(encoding="utf-8"))
        question = questions.get(payload.get("qid"))
        if question is None:
            continue
        metrics = compute_question_metrics(question, payload, chunk_store, nli=nli)
        metrics["system"] = payload.get("system") or path.parent.name
        records.append(metrics)

    output = Path(args.output) if args.output else results_root / "judged" / "local_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["system", "qid", "candidate_recall", "context_recall", "context_precision", "faithfulness"]
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, 0.0) for field in fields})
    print(f"Wrote {len(records)} local metric rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
