import json
from pathlib import Path

from ragc.eval.aggregate import aggregate_results
from ragc.eval.plots import generate_figures


def _write_result(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_results_writes_tables_and_figures(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sys_a = results_dir / "s1.json"
    sys_b = results_dir / "s2.json"
    _write_result(sys_a, {"qid": "t01", "system": "S0", "split": "test", "answer": "A", "correctness": 1.0, "usage": {"llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 50, "latency_s": 0.5}})
    _write_result(sys_a, {"qid": "t02", "system": "S0", "split": "test", "answer": "A", "correctness": 0.5, "usage": {"llm_calls": 1, "prompt_tokens": 200, "completion_tokens": 80, "latency_s": 0.7}})
    _write_result(sys_a, {"qid": "t03", "system": "S0", "split": "test", "answer": "A", "correctness": 1.0, "usage": {"llm_calls": 1, "prompt_tokens": 150, "completion_tokens": 60, "latency_s": 0.8}})
    _write_result(sys_b, {"qid": "t01", "system": "S1", "split": "test", "answer": "B", "correctness": 0.5, "usage": {"llm_calls": 2, "prompt_tokens": 220, "completion_tokens": 70, "latency_s": 0.9}})
    _write_result(sys_b, {"qid": "t02", "system": "S1", "split": "test", "answer": "B", "correctness": 1.0, "usage": {"llm_calls": 2, "prompt_tokens": 240, "completion_tokens": 90, "latency_s": 1.0}})
    _write_result(sys_b, {"qid": "t03", "system": "S1", "split": "test", "answer": "B", "correctness": 0.0, "usage": {"llm_calls": 2, "prompt_tokens": 260, "completion_tokens": 100, "latency_s": 1.1}})

    out_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    summary = aggregate_results(results_dir, output_dir=out_dir)
    assert summary["systems"] == ["S0", "S1"]
    assert (out_dir / "main_summary.md").exists()
    assert (out_dir / "by_type.md").exists()

    generated = generate_figures(results_dir, output_dir=figures_dir)
    assert generated["files"]
    assert (figures_dir / "correctness_by_system.png").exists()
