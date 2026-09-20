from pathlib import Path

from ragc.eval.pilot import run_pilot
from ragc.llm.fake import FakeLLM
from ragc.systems.s0_naive import S0NaiveSystem
from ragc.systems.s_1_none import S1NoRetrievalSystem


class StubRetriever:
    def __init__(self):
        self.hits = [{"chunk_id": "c1", "score": 1.0, "rank": 1, "via": "dense"}]

    def dense(self, query, k, source_types=None):
        return [type("Hit", (), {"chunk_id": "c1", "score": 1.0, "rank": 1, "via": "dense"})()]

    def get_chunk(self, chunk_id):
        return {"chunk_id": chunk_id, "header": "Evidence", "text": "The affected version is 10.2.9-h1.", "source_type": "nvd"}


def test_run_pilot_creates_markdown_and_uses_dev_questions(tmp_path):
    qrows = [
        {"qid": "d01", "question": "Which version is affected?"},
        {"qid": "d02", "question": "What is the CVE?"},
        {"qid": "d03", "question": "What is the mitigation?"},
    ]
    systems = [
        ("S-1", lambda cfg, llm, retriever=None, reranker=None, **kwargs: S1NoRetrievalSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs)),
        ("S0", lambda cfg, llm, retriever=None, reranker=None, **kwargs: S0NaiveSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs)),
    ]

    result = run_pilot(qrows, systems=systems, llm_factory=FakeLLM, retriever=StubRetriever(), output_path=tmp_path / "pilot.md")

    assert result["systems"] == ["S-1", "S0"]
    assert len(result["rows"]) == 2
    assert "| System |" in result["markdown"]
    assert "d01" in result["markdown"]
    assert (tmp_path / "pilot.md").exists()
