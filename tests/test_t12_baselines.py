from types import SimpleNamespace

from ragc.llm.fake import FakeLLM
from ragc.schemas import Hit
from ragc.systems.s_1_none import S1NoRetrievalSystem
from ragc.systems.s0_naive import S0NaiveSystem


class StubRetriever:
    def __init__(self):
        self.hits = [
            Hit(chunk_id=f"c{i}", score=float(10 - i), rank=i + 1, via="dense")
            for i in range(5)
        ]

    def dense(self, query: str, k: int, source_types=None):
        return self.hits[:k]

    def get_chunk(self, chunk_id: str):
        return {
            "chunk_id": chunk_id,
            "header": "vuln",
            "text": f"Evidence for {chunk_id}.",
            "source_type": "nvd",
            "title": "Example",
            "section": "Overview",
        }


def test_s1_no_retrieval_baseline_uses_one_generation_call():
    llm = FakeLLM()
    system = S1NoRetrievalSystem(cfg=SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1)), llm=llm, retriever=None, reranker=None)

    result = system.run("q1", "What is the CVE?")

    assert result.system == "S-1"
    assert result.candidate_ids == []
    assert result.final_evidence == []
    assert system.budget.used == 1
    assert len(llm.calls) == 1


def test_s0_naive_rag_returns_five_ranked_evidence_items():
    llm = FakeLLM()
    retriever = StubRetriever()
    system = S0NaiveSystem(cfg=SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1)), llm=llm, retriever=retriever, reranker=None)

    result = system.run("q2", "Which version is affected?")

    assert result.system == "S0"
    assert len(result.candidate_ids) == 5
    assert len(result.final_evidence) == 5
    assert [item.chunk_id for item in result.final_evidence] == ["c0", "c1", "c2", "c3", "c4"]
    assert system.budget.used == 1
    assert len(llm.calls) == 1
