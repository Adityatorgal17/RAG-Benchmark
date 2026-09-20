import json
from pathlib import Path

from ragc.eval.retrieval_metrics import compute_question_metrics, summarize_metrics


def test_local_metrics_compute_recall_and_precision_for_handmade_result():
    question = {
        "qid": "d01",
        "split": "dev",
        "type": "single_fact",
        "answerable": True,
        "gold_evidence": [{"doc_id": "nvd_1", "snippet": "affected versions include 10.2.9-h1"}],
        "gold_source_types": ["nvd", "vendor"],
    }
    chunks = {
        "nvd_1": [
            "NVD summary: affected versions include 10.2.9-h1 and later.",
            "Other unrelated text.",
        ],
        "vendor_1": [
            "Vendor advisory: upgrade now.",
        ],
    }
    result = {
        "qid": "d01",
        "candidate_ids": ["c1", "c2"],
        "final_evidence": [
            {"chunk_id": "c1", "source_type": "nvd", "header": "NVD", "text": "affected versions include 10.2.9-h1 and later."},
            {"chunk_id": "c2", "source_type": "vendor", "header": "Vendor", "text": "upgrade now."},
        ],
    }

    metrics = compute_question_metrics(question, result, chunks)

    assert metrics["candidate_recall"] == 1.0
    assert metrics["context_recall"] == 1.0
    assert metrics["context_precision"] == 1.0
    assert "faithfulness" in metrics
    assert 0.0 <= metrics["faithfulness"] <= 1.0


def test_summarize_metrics_and_csv_write(tmp_path):
    records = [
        {"qid": "d01", "candidate_recall": 1.0, "context_recall": 1.0, "context_precision": 0.5, "faithfulness": 0.8},
        {"qid": "d02", "candidate_recall": 0.5, "context_recall": 0.5, "context_precision": 1.0, "faithfulness": 0.4},
    ]

    summary = summarize_metrics(records)
    assert summary["candidate_recall"] == 0.75
    assert summary["context_recall"] == 0.75
    assert summary["context_precision"] == 0.75
    assert summary["faithfulness"] == 0.6

    out_path = tmp_path / "local_metrics.csv"
    from ragc.eval.retrieval_metrics import write_local_metrics_csv

    write_local_metrics_csv(records, out_path)
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "qid" in text
    assert "candidate_recall" in text
