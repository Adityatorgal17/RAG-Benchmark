from types import SimpleNamespace

from ragc.llm.fake import FakeLLM, FakeLLMResult
from ragc.schemas import Hit


class S6cRetriever:
    def __init__(self, *, restricted_empty: bool = True):
        self.restricted_empty = restricted_empty
        self.calls = []
        self.chunks = {
            "c1": {"chunk_id": "c1", "header": "Vendor advisory", "text": "Affected version is 10.2.9-h1.", "source_type": "vendor"},
            "c2": {"chunk_id": "c2", "header": "NVD", "text": "This is a command injection flaw.", "source_type": "nvd"},
        }

    def hybrid(self, query: str, k_dense: int, k_bm25: int, k_out: int, source_types=None):
        self.calls.append({"query": query, "source_types": source_types})
        if source_types == ["vendor"] and self.restricted_empty:
            return []
        return [Hit(chunk_id="c1", score=5.0, rank=1, via="hybrid")]

    def get_chunk(self, chunk_id: str):
        return self.chunks.get(chunk_id, {})


class S6cLLM(FakeLLM):
    def __init__(self, planner_payload: str, refine_payload: str, decide_payload: str, final_answer: str = "Affected version is 10.2.9-h1. [c1]"):
        super().__init__()
        self.responses = iter([
            planner_payload,
            refine_payload,
            decide_payload,
            final_answer,
        ])

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "system": system})
        text = next(self.responses)
        return FakeLLMResult(text=text, prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)


def test_s6c_routes_to_source_filter_and_falls_back_once():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1), paths=SimpleNamespace())
    retriever = S6cRetriever(restricted_empty=True)
    llm = S6cLLM(
        planner_payload='{"sub_questions":[{"id":"s1","question":"Which version is affected?","sources":["vendor"]}]}',
        refine_payload='{"notes":[{"fact":"version is 10.2.9-h1","chunk_ids":["c1"],"sub_question":"s1"}],"conflicts":[]}',
        decide_payload='{"sufficient": true, "gaps": []}',
        final_answer='Affected version is 10.2.9-h1. [c1]',
    )

    from ragc.systems.s6c_hybrid import S6cHybridSystem

    system = S6cHybridSystem(cfg=cfg, llm=llm, retriever=retriever)
    result = system.run("d01", "Which version is affected?")

    assert result.system == "S6c"
    assert result.final_evidence[0].chunk_id == "c1"
    assert retriever.calls[0]["source_types"] == ["vendor"]
    assert retriever.calls[1]["source_types"] is None
    assert system.budget.used <= 7
    assert len(llm.calls) == 4


def test_a1_disables_routing_and_keeps_retrieval_unrestricted():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1), paths=SimpleNamespace())
    retriever = S6cRetriever(restricted_empty=False)
    llm = S6cLLM(
        planner_payload='{"sub_questions":[{"id":"s1","question":"Which version is affected?"}]}',
        refine_payload='{"notes":[{"fact":"version is 10.2.9-h1","chunk_ids":["c1"],"sub_question":"s1"}],"conflicts":[]}',
        decide_payload='{"sufficient": true, "gaps": []}',
        final_answer='Affected version is 10.2.9-h1. [c1]',
    )

    from ragc.systems.s6c_hybrid import S6cHybridSystem

    system = S6cHybridSystem(cfg=cfg, llm=llm, retriever=retriever, use_routing=False)
    result = system.run("d02", "Which version is affected?")

    assert result.system == "A1"
    assert all(call["source_types"] is None for call in retriever.calls)
    assert system.budget.used <= 7
    assert len(llm.calls) == 4
