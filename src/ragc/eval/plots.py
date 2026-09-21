"""Plotting utilities for real benchmark metrics."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from ragc.eval.aggregate import load_merged_rows

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency should exist in project env
    raise RuntimeError("matplotlib is required for report figures. Install requirements.txt.") from exc


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _usage(row: dict, key: str) -> float | None:
    usage = row.get("usage") or {}
    if not isinstance(usage, dict):
        return None
    return _float(usage.get(key))


def _require_metric(rows: list[dict], key: str) -> None:
    if not any(_float(row.get(key)) is not None for row in rows):
        raise RuntimeError(f"Cannot plot {key}: no real metric values found. Run the required evaluation script first.")


def _group_mean(rows: list[dict], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _float(row.get(metric))
        if value is None:
            continue
        grouped.setdefault(str(row.get("system") or "unknown"), []).append(value)
    return {system: mean(values) for system, values in sorted(grouped.items()) if values}


def _save_empty_guard(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("No raw system results found. Run scripts/run_system.py before generating figures.")


def generate_figures(results_root: str | Path, *, output_dir: str | Path | None = None) -> dict:
    """Generate benchmark figures from real raw/judged/local metric artifacts."""
    rows = load_merged_rows(results_root)
    _save_empty_guard(rows)

    out = Path(output_dir) if output_dir is not None else Path(results_root) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    _require_metric(rows, "correctness")
    correctness = _group_mean(rows, "correctness")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(correctness.keys()), list(correctness.values()), color="#4c72b0")
    ax.set_ylabel("Correctness")
    ax.set_title("Correctness by system")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = out / "correctness_by_system.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(str(path))

    by_type: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = _float(row.get("correctness"))
        if value is None:
            continue
        by_type.setdefault((str(row.get("system") or "unknown"), str(row.get("type") or "unknown")), []).append(value)
    if not by_type:
        raise RuntimeError("Cannot plot correctness by type: no judged correctness values found.")
    systems = sorted({system for system, _ in by_type})
    qtypes = sorted({qtype for _, qtype in by_type})
    heat = [[mean(by_type[(system, qtype)]) if (system, qtype) in by_type else float("nan") for qtype in qtypes] for system in systems]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(heat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(qtypes)))
    ax.set_xticklabels(qtypes, rotation=45, ha="right")
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems)
    ax.set_title("Correctness by system x type")
    fig.tight_layout()
    path = out / "correctness_by_type.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(str(path))

    token_points: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        corr = _float(row.get("correctness"))
        prompt = _usage(row, "prompt_tokens")
        completion = _usage(row, "completion_tokens")
        if corr is None or prompt is None or completion is None:
            continue
        token_points.setdefault(str(row.get("system") or "unknown"), []).append((prompt + completion, corr))
    if not token_points:
        raise RuntimeError("Cannot plot tokens vs correctness: missing judged correctness or usage tokens.")
    fig, ax = plt.subplots(figsize=(7, 4))
    for system, values in sorted(token_points.items()):
        ax.scatter([mean([v[0] for v in values])], [mean([v[1] for v in values])], label=system)
    ax.set_xlabel("Mean total tokens")
    ax.set_ylabel("Correctness")
    ax.set_title("Tokens vs correctness")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "tokens_vs_correctness.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(str(path))

    for metric in ["candidate_recall", "context_recall", "faithfulness", "contradiction_rate"]:
        _require_metric(rows, metric)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["candidate recall", "context recall"],
        [
            mean([float(row["candidate_recall"]) for row in rows if _float(row.get("candidate_recall")) is not None]),
            mean([float(row["context_recall"]) for row in rows if _float(row.get("context_recall")) is not None]),
        ],
        color=["#55a868", "#c44e52"],
    )
    ax.set_title("Retrieval funnel")
    fig.tight_layout()
    path = out / "retrieval_funnel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["Faithfulness", "Contradiction"],
        [
            mean([float(row["faithfulness"]) for row in rows if _float(row.get("faithfulness")) is not None]),
            mean([float(row["contradiction_rate"]) for row in rows if _float(row.get("contradiction_rate")) is not None]),
        ],
        color=["#8172b3", "#ccb974"],
    )
    ax.set_title("Faithfulness and contradiction rate")
    fig.tight_layout()
    path = out / "faithfulness_contradiction.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(str(path))

    latency_values = [_usage(row, "latency_s") for row in rows]
    latency_values = [v for v in latency_values if v is not None and v > 0]
    if not latency_values:
        raise RuntimeError("Cannot plot latency: no real usage.latency_s values found.")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(latency_values)
    ax.set_yscale("log")
    ax.set_title("Latency distribution")
    fig.tight_layout()
    path = out / "latency_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(str(path))

    abstention_rows = [row for row in rows if "abstained" in row]
    if not abstention_rows:
        raise RuntimeError("Cannot plot abstention: raw result rows do not include abstained flags.")
    u_rows = [row for row in abstention_rows if str(row.get("type")) == "unanswerable"]
    answerable_rows = [row for row in abstention_rows if str(row.get("type")) != "unanswerable"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["U abstain rate", "answerable false abstain"],
        [
            mean([1.0 if row.get("abstained") else 0.0 for row in u_rows]) if u_rows else float("nan"),
            mean([1.0 if row.get("abstained") else 0.0 for row in answerable_rows]) if answerable_rows else float("nan"),
        ],
        color=["#64b5cd", "#ff8f6b"],
    )
    ax.set_title("Abstention")
    fig.tight_layout()
    path = out / "abstention.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(str(path))

    return {"output_dir": out, "files": files}


__all__ = ["generate_figures"]
