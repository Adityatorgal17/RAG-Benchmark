from types import SimpleNamespace

from ragc.llm.fake import FakeLLM, FakeLLMResult
from ragc.schemas import Hit


class StubRetriever:
    def __init__(self):
        self.chunks = {
            "c1": {"chunk_id": "c1", "header": "Vendor advisory", "text": "The affected version is 10.2.9-h1.", "source_type": "vendor"},
            "c2": {"chunk_id": "c2", "header": "NVD", "text": "This is a command injection flaw.", "source_type": "nvd"},
        }

    def hybrid(self, query: str, k_dense: int, k_bm25: int, k_out: int, source_types=None):
        if "affected version" in query.lower():
            return [Hit(chunk_id="c1", score=1.0, rank=1, via="hybrid")]
        return [Hit(chunk_id="c2", score=1.0, rank=1, via="hybrid")]

    def get_chunk(self, chunk_id: str):
        return self.chunks.get(chunk_id, {})


class S6aSequenceLLM(FakeLLM):
    def __init__(self, decision=None, queries=None, sufficiency=None, final_text="Affected version is 10.2.9-h1. [c1]"):
        super().__init__()
        self._responses = iter([
            decision or '{"retrieve": true, "complexity": "simple"}',
            *([query for query in (queries or [])]),
            *(sufficiency or []),
            final_text,
        ])

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "system": system})
        text = next(self._responses)
        return FakeLLMResult(text=text, prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)


def test_s6a_simple_question_uses_one_query_round():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1), paths=SimpleNamespace())
    llm = S6aSequenceLLM(
        decision='{"retrieve": true, "complexity": "simple"}',
        queries=['{"query": "affected version"}'],
        sufficiency=[],
        final_text="Affected version is 10.2.9-h1. [c1]",
    )
    retriever = StubRetriever()

    from ragc.systems.s6a_acquisition import S6aAcquisitionSystem

    system = S6aAcquisitionSystem(cfg=cfg, llm=llm, retriever=retriever)
    result = system.run("d01", "Which version is affected?")

    assert result.system == "S6a"
    assert result.final_evidence[0].chunk_id == "c1"
    assert system.budget.used == 3
    assert len(llm.calls) == 3


def test_s6a_duplicate_query_stops_the_rounds_and_keeps_budget():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1), paths=SimpleNamespace())
    llm = S6aSequenceLLM(
        decision='{"retrieve": true, "complexity": "multi_fact"}',
        queries=['{"query": "affected version"}', '{"query": "affected version"}'],
        sufficiency=['{"sufficient": false, "missing": "vendor details"}'],
        final_text="Affected version is 10.2.9-h1. [c1]",
    )
    retriever = StubRetriever()

    from ragc.systems.s6a_acquisition import S6aAcquisitionSystem

    system = S6aAcquisitionSystem(cfg=cfg, llm=llm, retriever=retriever)
    result = system.run("d02", "Which version is affected?")

    assert result.system == "S6a"
    assert system.budget.used <= 7
    assert len(llm.calls) <= 7
    assert result.final_evidence
