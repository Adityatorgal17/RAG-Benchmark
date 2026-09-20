"""Simple plotting utilities for benchmark aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_rows(results_root: str | Path) -> list[dict]:
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


def _system_mean(rows: list[dict], key: str = "correctness") -> dict[str, float]:
    by_system: dict[str, list[float]] = {}
    for row in rows:
        system = str(row.get("system") or "unknown")
        value = row.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        by_system.setdefault(system, []).append(numeric)
    return {system: sum(values) / max(len(values), 1) for system, values in sorted(by_system.items())}


def generate_figures(results_root: str | Path, *, output_dir: str | Path | None = None) -> dict:
    """Generate the standard benchmark figures and return output paths."""
    rows = _load_rows(results_root)
    out = Path(output_dir) if output_dir is not None else Path(results_root) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    system_means = _system_mean(rows)
    if system_means:
        labels = list(system_means.keys())
        values = list(system_means.values())
        ax.bar(labels, values, color="#4c72b0")
        ax.set_ylabel("Correctness")
        ax.set_title("Correctness by system")
        ax.set_ylim(0, 1.1)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "correctness_by_system.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    by_type: dict[str, list[float]] = {}
    for row in rows:
        qtype = str(row.get("type") or "unknown")
        system = str(row.get("system") or "unknown")
        value = row.get("correctness", 0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        by_type.setdefault((system, qtype), []).append(numeric)
    if by_type:
        systems = sorted({system for system, _ in by_type})
        qtypes = sorted({qtype for _, qtype in by_type})
        heat = [[sum(by_type.get((system, qtype), [0])) / max(len(by_type.get((system, qtype), [0])), 1) for qtype in qtypes] for system in systems]
        ax.imshow(heat, cmap="viridis")
        ax.set_xticks(range(len(qtypes)))
        ax.set_xticklabels(qtypes, rotation=45, ha="right")
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels(systems)
        ax.set_title("Correctness by system x type")
    else:
        ax.text(0.5, 0.5, "No type data", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "correctness_by_type.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    tokens = []
    correctness = []
    for row in rows:
        usage = row.get("usage") or {}
        total_tokens = 0.0
        if isinstance(usage, dict):
            total_tokens = float(usage.get("prompt_tokens", 0) or 0) + float(usage.get("completion_tokens", 0) or 0)
        try:
            correctness_val = float(row.get("correctness", 0) or 0)
        except (TypeError, ValueError):
            correctness_val = 0.0
        tokens.append(total_tokens)
        correctness.append(correctness_val)
    if tokens:
        ax.scatter(tokens, correctness)
        ax.set_xlabel("Mean total tokens")
        ax.set_ylabel("Correctness")
        ax.set_title("Tokens vs correctness")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "tokens_vs_correctness.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if rows:
        ax.bar(["candidate recall", "context recall"], [
            sum(float(row.get("candidate_recall", 0) or 0) for row in rows) / max(len(rows), 1),
            sum(float(row.get("context_recall", 0) or 0) for row in rows) / max(len(rows), 1),
        ], color=["#55a868", "#c44e52"])
        ax.set_title("Retrieval funnel")
    else:
        ax.text(0.5, 0.5, "No retrieval data", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "retrieval_funnel.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if rows:
        faithfulness = [float(row.get("faithfulness", 0) or 0) for row in rows]
        contradiction = [1.0 if float(row.get("contradicts_gold", 0) or 0) else 0.0 for row in rows]
        ax.bar(["Faithfulness", "Contradiction"], [sum(faithfulness) / max(len(faithfulness), 1), sum(contradiction) / max(len(contradiction), 1)], color=["#8172b3", "#ccB974"])
        ax.set_title("Faithfulness and contradiction rate")
    else:
        ax.text(0.5, 0.5, "No NLI data", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "faithfulness_contradiction.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    values = []
    for row in rows:
        usage = row.get("usage") or {}
        values.append(float(usage.get("latency_s", 0) or 0))
    if values:
        ax.boxplot(values)
        ax.set_yscale("log")
        ax.set_title("Latency per system")
    else:
        ax.text(0.5, 0.5, "No latency data", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "latency_boxplot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if rows:
        abstain_values = [1.0 if row.get("abstained") else 0.0 for row in rows]
        ax.bar(["Abstention", "False abstain"], [sum(abstain_values) / max(len(abstain_values), 1), 0.0], color=["#64b5cd", "#ff8f6b"])
        ax.set_title("Abstention")
    else:
        ax.text(0.5, 0.5, "No abstention data", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "abstention.png", dpi=150)
    plt.close(fig)

    files = [
        "correctness_by_system.png",
        "correctness_by_type.png",
        "tokens_vs_correctness.png",
        "retrieval_funnel.png",
        "faithfulness_contradiction.png",
        "latency_boxplot.png",
        "abstention.png",
    ]
    return {"output_dir": out, "files": [str(out / name) for name in files]}


__all__ = ["generate_figures"]
