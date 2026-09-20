from types import SimpleNamespace

from ragc.llm.fake import FakeLLM, FakeLLMResult
from ragc.schemas import Hit
from ragc.systems.s1_pre import S1PreSystem


class StubRetriever:
    def __init__(self):
        self.chunk_ids = [f"c{i}" for i in range(1, 11)]

    def dense(self, query: str, k: int, source_types=None):
        return [Hit(chunk_id=chunk_id, score=float(10 - idx), rank=idx + 1, via="dense") for idx, chunk_id in enumerate(self.chunk_ids[:k])]

    def bm25(self, query: str, k: int, source_types=None):
        return [Hit(chunk_id=chunk_id, score=float(9 - idx), rank=idx + 1, via="bm25") for idx, chunk_id in enumerate(self.chunk_ids[:k])]

    def passage(self, passage: str, k: int, source_types=None):
        return [Hit(chunk_id=chunk_id, score=float(8 - idx), rank=idx + 1, via="hyde") for idx, chunk_id in enumerate(self.chunk_ids[:k])]

    def get_chunk(self, chunk_id: str):
        return {
            "chunk_id": chunk_id,
            "header": "Security advisory",
            "text": f"Evidence for {chunk_id}.",
            "source_type": "vendor",
            "title": "Example",
            "section": "Overview",
        }


class JsonFakeLLM(FakeLLM):
    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "system": system})
        if json_mode:
            text = '{"rewrites": ["patched version issue", "CVE affected product"], "hyde": "The advisory explains the firewall vulnerability and the patched version is 10.2.9-h1."}'
            return FakeLLMResult(text=text, prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)
        return FakeLLMResult(text="Affected versions include 10.2.9-h1. [c1]", prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)


def test_s1_pre_retrieval_uses_rewrite_plus_hyde_and_fuses_lists():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1))
    llm = JsonFakeLLM()
    retriever = StubRetriever()
    system = S1PreSystem(cfg=cfg, llm=llm, retriever=retriever)

    result = system.run("d01", "Which version is affected?")

    assert result.system == "S1"
    assert len(result.candidate_ids) >= 1
    assert len(result.final_evidence) == 5
    assert system.budget.used == 2
    assert len(llm.calls) == 2
