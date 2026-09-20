"""Build benchmark report tables from saved JSON results."""

from __future__ import annotations

import json
from pathlib import Path

from ragc.eval.aggregate import aggregate_results


def _read_result_payloads(results_root: str | Path) -> dict[str, dict]:
    root = Path(results_root)
    payloads: dict[str, dict] = {}
    if not root.exists():
        return payloads
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        qid = str(data.get("qid") or path.stem)
        system = str(data.get("system") or path.parent.name)
        payloads[(qid, system)] = data
    return payloads


def check_consistency(results_root: str | Path, *, systems: list[str] | None = None, qids: list[str] | None = None) -> dict[str, list[str]]:
    """Return missing qids by system for the requested benchmark coverage check."""
    root = Path(results_root)
    seen = _read_result_payloads(root)
    if systems is None:
        systems = sorted({system for (_, system) in seen})
    if qids is None:
        qids = sorted({qid for (qid, _) in seen})
    present_by_system: dict[str, set[str]] = {system: set() for system in systems}
    for (qid, system), payload in seen.items():
        if system in present_by_system:
            present_by_system[system].add(qid)

    missing: dict[str, list[str]] = {}
    for system in systems:
        missing_qids = [qid for qid in qids if qid not in present_by_system.get(system, set())]
        if missing_qids:
            missing[system] = missing_qids
    return missing


def make_report(results_root: str | Path) -> dict:
    """Create the report tables in results/tables and return the output metadata."""
    root = Path(results_root)
    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_results(root, output_dir=tables_dir)
    table_lines = [
        "# Benchmark summary",
        "",
        "| System | Mean correctness | Mean calls | Mean tokens | Mean latency | Quality / 1k tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        table_lines.append(f"| {row['system']} | {row['mean_correctness']:.4f} | {row['mean_calls']:.2f} | {row['mean_tokens']:.2f} | {row['mean_latency']:.2f} | {row['quality_per_1000_tokens']:.4f} |")
    (tables_dir / "summary.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    return {"results_root": root, "tables_dir": tables_dir, "summary": summary}


def main() -> int:
    root = Path("results")
    report = make_report(root)
    missing = check_consistency(root)
    print(f"Report written to {report['tables_dir']}")
    print(f"Missing coverage entries: {sum(len(v) for v in missing.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
