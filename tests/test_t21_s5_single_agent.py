from types import SimpleNamespace

from ragc.llm.fake import FakeLLM, FakeLLMResult
from ragc.schemas import Hit
from ragc.systems.s5_single_agent import S5SingleAgentSystem


class StubRetriever:
    def __init__(self):
        self.chunks = {
            "c1": {"chunk_id": "c1", "header": "Vendor", "text": "The affected version is 10.2.9-h1.", "source_type": "vendor"},
            "c2": {"chunk_id": "c2", "header": "NVD", "text": "This is a command injection flaw.", "source_type": "nvd"},
        }

    def hybrid(self, query: str, k_dense: int, k_bm25: int, k_out: int, source_types=None):
        return [Hit(chunk_id="c1", score=1.0, rank=1, via="hybrid")]

    def get_chunk(self, chunk_id: str):
        return self.chunks.get(chunk_id, {})


class AgentSequenceLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self._responses = iter([
            '{"thought": "Need version info.", "new_facts": [{"fact": "version 10.2.9-h1", "chunk_id": "c1"}], "useful_chunk_ids": ["c1"], "action": "search", "query": "affected version"}',
            '{"thought": "The version is known.", "new_facts": [], "useful_chunk_ids": ["c1"], "action": "answer"}',
            "The affected version is 10.2.9-h1. [c1]",
        ])

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "system": system})
        text = next(self._responses)
        return FakeLLMResult(text=text, prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)


def test_s5_single_agent_searches_until_duplicate_then_answers():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1), paths=SimpleNamespace())
    llm = AgentSequenceLLM()
    retriever = StubRetriever()
    system = S5SingleAgentSystem(cfg=cfg, llm=llm, retriever=retriever)

    result = system.run("d01", "Which version is affected?")

    assert result.system == "S5"
    assert len(llm.calls) == 3
    assert result.final_evidence[0].chunk_id == "c1"
    assert system.budget.used == 3
