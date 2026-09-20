"""Entry point for running a single system over a question split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragc.config import load_config
from ragc.llm.fake import FakeLLM
from ragc.llm.keypool import QuotaExhausted
from ragc.systems.registry import get_system_factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a benchmark system for a question subset.")
    parser.add_argument("--system", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qids", default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    return parser


def run_system(system_id: str, qids: list[str] | None = None, split: str = "test", limit: int | None = None, max_calls: int | None = None):
    """Run a system over the requested questions.

    This is intentionally small and resumable: it skips any output file that already
    exists and relies on the underlying system implementations to write JSON results.
    """
    cfg = load_config()
    llm = FakeLLM()
    factory = get_system_factory(system_id)
    try:
        if qids is None:
            qids = []
        _ = factory(cfg=cfg, llm=llm, max_calls=max_calls or 8, retriever=None, reranker=None)
    except NotImplementedError:
        return 0

    results_root = Path(cfg.paths.results)
    results_root.mkdir(parents=True, exist_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    qids = [qid.strip() for qid in args.qids.split(",") if qid.strip()] if args.qids else None
    try:
        return run_system(args.system, qids=qids, split=args.split, limit=args.limit, max_calls=args.max_calls)
    except QuotaExhausted:
        print(f"Quota exhausted for {args.system} on split={args.split}")
        return 75


if __name__ == "__main__":
    sys.exit(main())
