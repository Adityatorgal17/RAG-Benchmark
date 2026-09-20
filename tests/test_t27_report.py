import json
from pathlib import Path

from scripts.make_report import check_consistency, make_report


def test_make_report_writes_markdown_tables_and_consistency_zero(tmp_path):
    base = tmp_path / "results"
    (base / "systems").mkdir(parents=True)
    (base / "tables").mkdir(parents=True)
    for qid in ["t01", "t02", "t03"]:
        payload = {
            "qid": qid,
            "system": "S0",
            "split": "test",
            "question": f"Question {qid}",
            "answer": "Answer",
            "gold_answer": "Answer",
            "gold_claims": [],
            "usage": {"llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 50, "latency_s": 0.1},
        }
        (base / "systems" / f"{qid}_S0.json").write_text(json.dumps(payload), encoding="utf-8")

    # create a complete set for two systems, one missing qid to confirm detection
    for qid in ["t01", "t02"]:
        payload = {"qid": qid, "system": "S1", "split": "test", "question": f"Question {qid}", "answer": "Answer", "gold_answer": "Answer", "gold_claims": [], "usage": {"llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 50, "latency_s": 0.1}}
        (base / "systems" / f"{qid}_S1.json").write_text(json.dumps(payload), encoding="utf-8")

    report = make_report(base)
    assert report["tables_dir"].exists()
    assert (base / "tables" / "summary.md").exists()

    missing = check_consistency(base, systems=["S0", "S1"], qids=["t01", "t02", "t03"])
    assert missing == {"S1": ["t03"]}

