"""Aggregation helpers for benchmark results.

Aggregation is intentionally strict: it combines raw system outputs with judged metrics
and local retrieval metrics, and never treats missing evaluation artifacts as zero.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = [
    "correctness",
    "claim_coverage",
    "candidate_recall",
    "context_recall",
    "context_precision",
    "faithfulness",
    "contradiction_rate",
]


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return mean(clean) if clean else None


def _fmt(value: float | None, *, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _load_raw_results(results_root: str | Path) -> dict[tuple[str, str], dict]:
    root = Path(results_root) / "raw"
    rows: dict[tuple[str, str], dict] = {}
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        qid = str(payload.get("qid") or path.stem)
        system = str(payload.get("system") or path.parent.name)
        rows[(qid, system)] = payload
    return rows


def _load_pointwise(results_root: str | Path) -> dict[tuple[str, str], dict]:
    path = Path(results_root) / "judged" / "pointwise.jsonl"
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = payload.get("qid")
            system = payload.get("system")
            if qid and system:
                rows[(str(qid), str(system))] = payload
    return rows


def _load_local_metrics(results_root: str | Path) -> dict[tuple[str, str], dict]:
    path = Path(results_root) / "judged" / "local_metrics.csv"
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for payload in reader:
            qid = payload.get("qid")
            system = payload.get("system")
            if qid and system:
                rows[(str(qid), str(system))] = payload
    return rows


def load_merged_rows(results_root: str | Path) -> list[dict]:
    """Return one merged row per raw system result.

    Raw outputs are the source of coverage. Judged/local metrics are merged by
    `(qid, system)` when present. Missing metrics remain absent.
    """
    raw = _load_raw_results(results_root)
    judged = _load_pointwise(results_root)
    local = _load_local_metrics(results_root)
    rows: list[dict] = []

    for key, payload in sorted(raw.items(), key=lambda item: (item[0][1], item[0][0])):
        qid, system = key
        merged = dict(payload)
        merged["qid"] = qid
        merged["system"] = system

        judge_row = judged.get(key, {})
        for metric in ["correctness", "claim_coverage", "contradiction_rate"]:
            value = _safe_float(judge_row.get(metric))
            if value is not None:
                merged[metric] = value
        if "abstained" in judge_row:
            merged["judge_abstained"] = bool(judge_row.get("abstained"))

        local_row = local.get(key, {})
        for metric in ["candidate_recall", "context_recall", "context_precision", "faithfulness"]:
            value = _safe_float(local_row.get(metric))
            if value is not None:
                merged[metric] = value

        rows.append(merged)
    return rows


def _bootstrap_ci(values: list[float], *, seed: int = 42, n_resamples: int = 10000, confidence: float = 0.95) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    sample_mean = mean(values)
    boot = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in values]
        boot.append(mean(sample))
    boot.sort()
    lower_idx = max(0, math.floor((1 - confidence) * n_resamples / 2))
    upper_idx = min(n_resamples - 1, math.ceil((1 + confidence) * n_resamples / 2) - 1)
    return round(boot[lower_idx] - sample_mean, 4), round(boot[upper_idx] - sample_mean, 4)


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("system") or "unknown"), []).append(row)
    return grouped


def _metric_rows(rows: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []
    for system, system_rows in sorted(_group_rows(rows).items()):
        calls = [_safe_float((row.get("usage") or {}).get("llm_calls")) for row in system_rows if isinstance(row.get("usage"), dict)]
        prompt = [_safe_float((row.get("usage") or {}).get("prompt_tokens")) for row in system_rows if isinstance(row.get("usage"), dict)]
        completion = [_safe_float((row.get("usage") or {}).get("completion_tokens")) for row in system_rows if isinstance(row.get("usage"), dict)]
        latency = [_safe_float((row.get("usage") or {}).get("latency_s")) for row in system_rows if isinstance(row.get("usage"), dict)]
        total_tokens = [
            float(p) + float(c)
            for p, c in zip(prompt, completion)
            if p is not None and c is not None
        ]
        correctness_values = [_safe_float(row.get("correctness")) for row in system_rows]
        correctness = _mean(correctness_values)
        correctness_clean = [float(v) for v in correctness_values if v is not None]
        ci_low, ci_high = _bootstrap_ci(correctness_clean) if correctness_clean else (None, None)

        row = {
            "system": system,
            "n_results": len(system_rows),
            "n_judged": len(correctness_clean),
            "mean_correctness": correctness,
            "correctness_ci_low": ci_low,
            "correctness_ci_high": ci_high,
            "mean_calls": _mean(calls),
            "mean_tokens": _mean(total_tokens),
            "mean_latency": _mean(latency),
            "quality_per_1000_tokens": None,
        }
        if correctness is not None and row["mean_tokens"]:
            row["quality_per_1000_tokens"] = correctness / max(float(row["mean_tokens"]), 1.0) * 1000.0
        for metric in ["claim_coverage", "candidate_recall", "context_recall", "context_precision", "faithfulness", "contradiction_rate"]:
            row[f"mean_{metric}"] = _mean([_safe_float(item.get(metric)) for item in system_rows])
        summary_rows.append(row)
    return summary_rows


def aggregate_results(results_root: str | Path, *, output_dir: str | Path | None = None) -> dict:
    """Aggregate raw, judged, and local metric artifacts into Markdown tables."""
    rows = load_merged_rows(results_root)
    systems = sorted({str(row.get("system") or "unknown") for row in rows})
    summary_rows = _metric_rows(rows)

    output = Path(output_dir) if output_dir is not None else Path(results_root) / "tables"
    output.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Main summary",
        "",
        "| System | Results | Judged | Correctness | 95% CI delta | Calls | Tokens | Latency | Claim coverage | Candidate recall | Context recall | Context precision | Faithfulness | Contradiction | Quality / 1k tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        ci = "n/a" if row["correctness_ci_low"] is None else f"{row['correctness_ci_low']:+.4f}/{row['correctness_ci_high']:+.4f}"
        lines.append(
            "| {system} | {n_results} | {n_judged} | {correctness} | {ci} | {calls} | {tokens} | {latency} | {claim} | {cand} | {ctxr} | {ctxp} | {faith} | {contra} | {quality} |".format(
                system=row["system"],
                n_results=row["n_results"],
                n_judged=row["n_judged"],
                correctness=_fmt(row["mean_correctness"]),
                ci=ci,
                calls=_fmt(row["mean_calls"], digits=2),
                tokens=_fmt(row["mean_tokens"], digits=2),
                latency=_fmt(row["mean_latency"], digits=2),
                claim=_fmt(row["mean_claim_coverage"]),
                cand=_fmt(row["mean_candidate_recall"]),
                ctxr=_fmt(row["mean_context_recall"]),
                ctxp=_fmt(row["mean_context_precision"]),
                faith=_fmt(row["mean_faithfulness"]),
                contra=_fmt(row["mean_contradiction_rate"]),
                quality=_fmt(row["quality_per_1000_tokens"]),
            )
        )
    (output / "main_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_type_lines = [
        "# Per-question-type summary",
        "",
        "| System | Type | Judged | Mean correctness |",
        "|---|---|---:|---:|",
    ]
    type_map: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = _safe_float(row.get("correctness"))
        if value is None:
            continue
        qtype = str(row.get("type") or row.get("question_type") or "unknown")
        system = str(row.get("system") or "unknown")
        type_map.setdefault((system, qtype), []).append(value)
    for (system, qtype), values in sorted(type_map.items()):
        by_type_lines.append(f"| {system} | {qtype} | {len(values)} | {mean(values):.4f} |")
    (output / "by_type.md").write_text("\n".join(by_type_lines) + "\n", encoding="utf-8")

    missing = {
        "pointwise": sum(1 for row in rows if _safe_float(row.get("correctness")) is None),
        "local_metrics": sum(1 for row in rows if _safe_float(row.get("faithfulness")) is None),
    }
    return {"systems": systems, "rows": summary_rows, "merged_rows": rows, "tables_dir": output, "n_rows": len(rows), "missing": missing}


__all__ = ["aggregate_results", "load_merged_rows"]
