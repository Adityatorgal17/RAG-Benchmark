"""Pilot runner for the benchmark's lightweight sanity check.

The T16 pilot is intentionally narrow: it runs a tiny subset of dev questions through the
lowest-friction systems, collects the measured call counts, tokens, and runtime, and writes
Markdown for the plan's Section 11.8 budget calibration. The logic is kept cheap and
resumable so it can be run before the full benchmark allocation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable

from ragc.config import load_config


def _estimate_tokens(question: str, answer: str) -> int:
    return max(1, len((question + " " + answer).split()))


def run_pilot(qrows: Iterable[dict], *, systems: list[tuple[str, Callable]], llm_factory, retriever=None, judge_fn=None, output_path: str | Path = "results/pilot.md") -> dict:
    """Run a small set of dev questions through the requested systems and emit a pilot table."""
    rows: list[dict] = []
    system_names = [name for name, _ in systems]
    for system_name, factory in systems:
        per_question: list[dict] = []
        for row in qrows:
            qid = row.get("qid")
            question = row.get("question")
            start = time.perf_counter()
            llm = llm_factory()
            system = factory(load_config(), llm=llm, retriever=retriever, reranker=None)
            result = system.run(qid, question)
            elapsed = time.perf_counter() - start

            judge_value = None
            if judge_fn is not None:
                gold_answer = row.get("gold_answer") or "INSUFFICIENT_EVIDENCE"
                gold_claims = row.get("gold_claims") or []
                judge_value = judge_fn(llm, question, gold_answer, gold_claims, result.answer)

            calls = int(result.usage.get("llm_calls", 0))
            if judge_value is not None:
                calls += 1
            token_count = int(max(1, result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0)))
            if judge_value is not None:
                token_count += 650

            per_question.append(
                {
                    "qid": qid,
                    "calls": calls,
                    "tokens": token_count,
                    "seconds": round(elapsed, 3),
                    "judge": judge_value,
                    "answer": result.answer,
                }
            )
        rows.append({"system": system_name, "rows": per_question})

    markdown_lines = [
        "# Pilot run summary",
        "",
        "| System | Mean calls | Mean tokens | Mean seconds | Questions |",
        "|---|---:|---:|---:|---:|",
    ]
    for system_row in rows:
        records = system_row["rows"]
        mean_calls = round(mean(r["calls"] for r in records), 2) if records else 0.0
        mean_tokens = round(mean(r["tokens"] for r in records), 2) if records else 0.0
        mean_seconds = round(mean(r["seconds"] for r in records), 2) if records else 0.0
        markdown_lines.append(f"| {system_row['system']} | {mean_calls} | {mean_tokens} | {mean_seconds} | {len(records)} |")

    markdown_lines.extend(["", "| System | qid | calls | tokens | seconds | judge |", "|---|---|---:|---:|---:|---|"])
    for system_row in rows:
        for record in system_row["rows"]:
            judge_label = str(record["judge"].get("correctness", "n/a")) if isinstance(record["judge"], dict) else "n/a"
            markdown_lines.append(f"| {system_row['system']} | {record['qid']} | {record['calls']} | {record['tokens']} | {record['seconds']} | {judge_label} |")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {"systems": system_names, "rows": rows, "markdown": "\n".join(markdown_lines) + "\n"}


def main() -> None:
    cfg = load_config()
    qpath = Path(cfg.paths.questions)
    qrows = []
    if qpath.exists():
        with qpath.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") == "dev":
                    qrows.append(row)
    qrows = qrows[:3]

    from ragc.eval.judge import judge_answer
    from ragc.llm.fake import FakeLLM
    from ragc.systems.s_1_none import S1NoRetrievalSystem
    from ragc.systems.s0_naive import S0NaiveSystem

    systems = [
        ("S-1", lambda cfg, llm, retriever=None, reranker=None, **kwargs: S1NoRetrievalSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs)),
        ("S0", lambda cfg, llm, retriever=None, reranker=None, **kwargs: S0NaiveSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs)),
    ]
    run_pilot(qrows, systems=systems, llm_factory=FakeLLM, judge_fn=judge_answer, output_path="results/pilot.md")


if __name__ == "__main__":
    main()
