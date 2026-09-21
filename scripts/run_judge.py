"""Entry point for running the pointwise judge over saved system outputs.

The project expects judging results to be append-only and resumable. This script tracks
which question/system pairs are already present in results/judged/pointwise.jsonl and skips
those entries instead of rewriting the file. It also raises a quota signal when the judge
provider reports a daily limit so the caller can stop gracefully and retry later.
"""

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

from ragc.config import load_config
from ragc.eval.judge import judge_answer
from ragc.llm.clients import build_judge_llm
from ragc.llm.keypool import QuotaExhausted


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _result_pairs(results_root: Path, system: str | None = None, split: str | None = None) -> list[tuple[str, str, dict]]:
    pairs: list[tuple[str, str, dict]] = []
    if not results_root.exists():
        return pairs
    for result_file in sorted(results_root.rglob("*.json")):
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        qid = payload.get("qid")
        system_name = payload.get("system") or result_file.parent.name
        if not qid or not system_name:
            continue
        if system is not None and system_name != system:
            continue
        if split is not None and payload.get("split") not in {split, None}:
            if payload.get("split") != split:
                continue
        pairs.append((qid, system_name, payload))
    return pairs


def run_judge(*, system: str | None = None, qids: list[str] | None = None, split: str = "test", max_tokens: int = 90000, fake: bool = False):
    """Judge saved answers and append new pointwise results in a resumable way."""
    cfg = load_config()
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    results_root = Path(cfg.paths.results)
    judged_dir = results_root / "judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    pointwise_path = judged_dir / "pointwise.jsonl"

    seen: set[tuple[str, str]] = set()
    for row in _load_jsonl(pointwise_path):
        if "qid" in row and "system" in row:
            seen.add((str(row["qid"]), str(row["system"])))

    llm = build_judge_llm(cfg, fake=fake)
    judge_rows: list[dict] = []
    used_tokens = 0
    for qid, system_name, payload in _result_pairs(results_root, system=system, split=split):
        if qids is not None and qid not in set(qids):
            continue
        key = (qid, system_name)
        if key in seen:
            continue
        question = payload.get("question") or ""
        gold_answer = payload.get("gold_answer") or "INSUFFICIENT_EVIDENCE"
        gold_claims = payload.get("gold_claims") or []
        answer = payload.get("answer") or ""
        result = judge_answer(llm, question, gold_answer, gold_claims, answer)
        used_tokens += 650
        if used_tokens > max_tokens:
            raise QuotaExhausted(f"Judge token budget exceeded: {used_tokens} > {max_tokens}")
        row = {
            "qid": qid,
            "system": system_name,
            "question": question,
            "correctness": float(result.get("correctness", 0)) / 2.0,
            "claim_coverage": (sum(1 for v in (result.get("claims") or {}).values() if v) / max(len(result.get("claims") or {}), 1)) if result.get("claims") else 0.0,
            "abstained": bool(result.get("abstained", False)),
            "contradiction_rate": 1.0 if result.get("contradicts_gold") else 0.0,
            "judgement": result,
        }
        judge_rows.append(row)
        seen.add(key)

    with pointwise_path.open("a", encoding="utf-8") as handle:
        for row in judge_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(judge_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pointwise LLM judging over saved system results.")
    parser.add_argument("--system", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--qids", default=None)
    parser.add_argument("--max-tokens", type=int, default=90000)
    parser.add_argument("--fake", action="store_true", help="Use FakeLLM instead of Groq for a dry run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    qids = [qid.strip() for qid in args.qids.split(",") if qid.strip()] if args.qids else None
    try:
        count = run_judge(system=args.system, qids=qids, split=args.split, max_tokens=args.max_tokens, fake=args.fake)
        print(f"Judged {count} answers")
        return 0
    except QuotaExhausted:
        print(f"Judge quota exhausted for split={args.split}; stopping before token cap")
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
