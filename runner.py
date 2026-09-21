"""Entry point for running a single system over a question split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragc.config import load_config
from ragc.llm.clients import build_main_llm
from ragc.llm.keypool import QuotaExhausted
from ragc.retrieval.reranker import Reranker
from ragc.retrieval.retriever import Retriever
from ragc.systems.registry import get_system_factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a benchmark system for a question subset.")
    parser.add_argument("--system", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qids", default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--subset", default=None, choices=["test24"])
    parser.add_argument("--fake", action="store_true", help="Use FakeLLM for a dry run instead of provider APIs.")
    return parser


def _load_questions(path: str | Path, *, split: str, qids: list[str] | None, subset: str | None, limit: int | None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}. Build data/questions/questions.jsonl first.")
    wanted = set(qids or [])
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if split and row.get("split") != split:
                continue
            if wanted and row.get("qid") not in wanted:
                continue
            if subset == "test24" and row.get("qid") in {"t05", "t10", "t15", "t20", "t25", "t30"}:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _needs_reranker(system_id: str) -> bool:
    return system_id in {"S2", "S3", "S6c", "A1", "A2a", "A2b"}


def run_system(system_id: str, qids: list[str] | None = None, split: str = "test", limit: int | None = None, max_calls: int | None = None, subset: str | None = None, fake: bool = False):
    """Run a system over the requested questions.

    This is intentionally small and resumable: it skips any output file that already
    exists and relies on the underlying system implementations to write JSON results.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    cfg = load_config()
    questions = _load_questions(cfg.paths.questions, split=split, qids=qids, subset=subset, limit=limit)
    if not questions:
        print(f"No questions matched split={split!r}, qids={qids!r}, subset={subset!r}")
        return 0

    llm = build_main_llm(cfg, fake=fake)
    retriever = Retriever(cfg)
    reranker = None
    if _needs_reranker(system_id):
        rerank_cfg = getattr(cfg, "reranker", None)
        reranker = Reranker(
            getattr(rerank_cfg, "model", "BAAI/bge-reranker-base"),
            batch_size=int(getattr(rerank_cfg, "batch_size", 16)),
            max_length=int(getattr(rerank_cfg, "max_length", 384)),
        )
    factory = get_system_factory(system_id)
    results_root = Path(cfg.paths.results) / "raw" / system_id
    results_root.mkdir(parents=True, exist_ok=True)
    new_results = 0
    calls_before = len(getattr(llm, "calls", []))
    for row in questions:
        qid = row["qid"]
        out_path = results_root / f"{qid}.json"
        if out_path.exists():
            print(f"skip {system_id} {qid}: {out_path}")
            continue
        system = factory(cfg=cfg, llm=llm, max_calls=int(cfg.budget.max_llm_calls), retriever=retriever, reranker=reranker)
        result = system.run(qid, row["question"])
        payload = result.model_dump()
        for key in ["split", "type", "answerable", "gold_answer", "gold_claims", "gold_evidence", "gold_source_types"]:
            if key in row:
                payload[key] = row[key]
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        new_results += 1
        calls_now = len(getattr(llm, "calls", []))
        print(f"wrote {system_id} {qid}: calls={payload.get('usage', {}).get('llm_calls', 0)} path={out_path}")
        if max_calls is not None and calls_now - calls_before >= max_calls:
            print(f"stopping after --max-calls={max_calls}")
            break
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    qids = [qid.strip() for qid in args.qids.split(",") if qid.strip()] if args.qids else None
    try:
        return run_system(args.system, qids=qids, split=args.split, limit=args.limit, max_calls=args.max_calls, subset=args.subset, fake=args.fake)
    except QuotaExhausted:
        print(f"Quota exhausted for {args.system} on split={args.split}")
        return 75


if __name__ == "__main__":
    sys.exit(main())
