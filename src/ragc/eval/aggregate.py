"""Aggregation helpers for benchmark results.

The implementation is intentionally lightweight but deterministic: it scans a results
folder for saved JSON payloads, aggregates the per-question metrics, and writes the
Markdown tables that are later consumed by the final report.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_result_files(results_root: str | Path) -> list[dict]:
    root = Path(results_root)
    if not root.exists():
        return []

    rows: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _bootstrap_ci(values: list[float], *, seed: int = 42, n_resamples: int = 1000, confidence: float = 0.95) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    arr = [float(v) for v in values]
    sample_mean = mean(arr)
    boot = []
    for _ in range(n_resamples):
        sample = [rng.choice(arr) for _ in arr]
        boot.append(mean(sample))
    boot.sort()
    lower_idx = max(0, math.floor((1 - confidence) * n_resamples / 2))
    upper_idx = min(n_resamples - 1, math.ceil((1 + confidence) * n_resamples / 2) - 1)
    return round(boot[lower_idx] - sample_mean, 4), round(boot[upper_idx] - sample_mean, 4)


def _metric_rows(rows: list[dict]) -> list[dict]:
    records: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        system = str(row.get("system") or "unknown")
        qid = str(row.get("qid") or "unknown")
        if qid == "unknown":
            continue
        records.setdefault(system, {}).setdefault("qid", []).append(qid)
        for key in ["correctness", "candidate_recall", "context_recall", "context_precision", "faithfulness", "latency_s", "llm_calls", "prompt_tokens", "completion_tokens", "absent"]:
            if key in row:
                records[system].setdefault(key, []).append(_safe_float(row.get(key), 0.0))
        usage = row.get("usage") or {}
        if isinstance(usage, dict):
            records[system].setdefault("llm_calls", []).append(_safe_float(usage.get("llm_calls", 0), 0.0))
            records[system].setdefault("prompt_tokens", []).append(_safe_float(usage.get("prompt_tokens", 0), 0.0))
            records[system].setdefault("completion_tokens", []).append(_safe_float(usage.get("completion_tokens", 0), 0.0))
            records[system].setdefault("latency_s", []).append(_safe_float(usage.get("latency_s", 0), 0.0))

    summary_rows: list[dict] = []
    for system in sorted(records):
        metrics = records[system]
        correctness = metrics.get("correctness", [0.0])
        total_tokens = [float(a) + float(b) for a, b in zip(metrics.get("prompt_tokens", [0.0] * len(correctness)), metrics.get("completion_tokens", [0.0] * len(correctness)))]
        summary_rows.append({
            "system": system,
            "mean_correctness": round(mean(correctness), 4) if correctness else 0.0,
            "mean_calls": round(mean(metrics.get("llm_calls", [0.0])), 4) if metrics.get("llm_calls") else 0.0,
            "mean_tokens": round(mean(total_tokens), 4) if total_tokens else 0.0,
            "mean_latency": round(mean(metrics.get("latency_s", [0.0])), 4) if metrics.get("latency_s") else 0.0,
            "quality_per_1000_tokens": round((mean(correctness) / max(mean(total_tokens), 1.0)) * 1000.0, 4) if correctness and total_tokens else 0.0,
        })
    return summary_rows


def aggregate_results(results_root: str | Path, *, output_dir: str | Path | None = None) -> dict:
    """Aggregate per-question JSON results and write a summary table."""
    rows = _load_result_files(results_root)
    systems = sorted({str(row.get("system") or "unknown") for row in rows})
    summary_rows = _metric_rows(rows)

    output = Path(output_dir) if output_dir is not None else Path(results_root) / "tables"
    output.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Main summary",
        "",
        "| System | Mean correctness | Mean calls | Mean tokens | Mean latency | Quality / 1k tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['system']} | {row['mean_correctness']:.4f} | {row['mean_calls']:.2f} | {row['mean_tokens']:.2f} | {row['mean_latency']:.2f} | {row['quality_per_1000_tokens']:.4f} |"
        )
    (output / "main_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_type_lines = [
        "# Per-question-type summary",
        "",
        "| System | Type | Mean correctness |",
        "|---|---|---:|",
    ]
    type_map: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        qtype = str(row.get("type") or row.get("question_type") or "unknown")
        system = str(row.get("system") or "unknown")
        type_map.setdefault((system, qtype), []).append(_safe_float(row.get("correctness"), 0.0))
    for (system, qtype), values in sorted(type_map.items()):
        by_type_lines.append(f"| {system} | {qtype} | {mean(values):.4f} |")
    (output / "by_type.md").write_text("\n".join(by_type_lines) + "\n", encoding="utf-8")

    return {"systems": systems, "rows": summary_rows, "tables_dir": output, "n_rows": len(rows)}


__all__ = ["aggregate_results"]
