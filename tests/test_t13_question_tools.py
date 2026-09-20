from pathlib import Path

from ragc.eval.make_questions import build_question_catalog, write_question_file
from ragc.eval.validate_questions import validate_question_file


def test_question_catalog_has_expected_shape_and_counts():
    rows = build_question_catalog()
    assert len(rows) == 40

    counts = {}
    for row in rows:
        counts[row["type"]] = counts.get(row["type"], 0) + 1

    assert counts == {
        "single_fact": 7,
        "multi_claim": 7,
        "multi_hop": 7,
        "cross_source": 7,
        "conflict": 6,
        "unanswerable": 6,
    }

    assert [row["qid"] for row in rows[:10]] == ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]
    assert [row["type"] for row in rows[:10]] == [
        "single_fact", "single_fact",
        "multi_claim", "multi_claim",
        "multi_hop", "multi_hop",
        "cross_source", "cross_source",
        "conflict", "unanswerable",
    ]
    assert [row["qid"] for row in rows[10:15]] == ["t01", "t02", "t03", "t04", "t05"]
    assert all("qid" in row and row["qid"] for row in rows)
    assert any(row["split"] == "dev" for row in rows)
    assert any(row["split"] == "test" for row in rows)


def test_validation_accepts_built_question_file(tmp_path):
    out_path = tmp_path / "questions.jsonl"
    rows = build_question_catalog()
    write_question_file(rows, out_path)
    status = validate_question_file(out_path)

    assert status["ok"] is True
    assert status["total"] == 40
    assert status["counts"]["single_fact"] == 7
