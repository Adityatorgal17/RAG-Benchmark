from types import SimpleNamespace

from ragc.llm.fake import FakeLLM, FakeLLMResult
from ragc.schemas import Hit
from ragc.systems.s3_decomp import S3DecompSystem


class StubRetriever:
    def __init__(self):
        self.chunks = {
            "c1": {"chunk_id": "c1", "header": "Vendor", "text": "The affected version is 10.2.9-h1.", "source_type": "vendor"},
            "c2": {"chunk_id": "c2", "header": "NVD", "text": "This affects PAN-OS.", "source_type": "nvd"},
            "c3": {"chunk_id": "c3", "header": "KEV", "text": "The issue is exploited in the wild.", "source_type": "kev"},
        }

    def hybrid(self, query, k_dense, k_bm25, k_out, source_types=None):
        if "version" in query.lower():
            hits = ["c1", "c3"]
        elif "product" in query.lower():
            hits = ["c2"]
        else:
            hits = ["c1", "c2", "c3"]
        return [Hit(chunk_id=chunk_id, score=1.0, rank=idx + 1, via="hybrid") for idx, chunk_id in enumerate(hits[:k_out])]

    def get_chunk(self, chunk_id):
        return self.chunks.get(chunk_id, {})


class StubReranker:
    def score(self, pairs):
        return [0.95 for _ in pairs]


class JsonFakeLLM(FakeLLM):
    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "system": system})
        if json_mode:
            text = '{"sub_queries": ["What version is affected?", "What product is affected?"]}'
            return FakeLLMResult(text=text, prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)
        return FakeLLMResult(text="Affected versions include 10.2.9-h1. [c1]", prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)


def test_s3_decomposition_uses_sub_queries_and_query_of_mapping():
    cfg = SimpleNamespace(
        budget=SimpleNamespace(max_llm_calls=8, reserve_final=1),
        post=SimpleNamespace(tau=0.10, compress_tokens=1500),
        retrieval=SimpleNamespace(s2=SimpleNamespace(keep_k=6)),
        paths=SimpleNamespace(),
    )
    llm = JsonFakeLLM()
    retriever = StubRetriever()
    system = S3DecompSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=StubReranker())

    result = system.run("d01", "Which version is affected?")

    assert result.system == "S3"
    assert len(result.candidate_ids) >= 1
    assert len(result.final_evidence) >= 1
    assert system.budget.used == 2
    assert len(llm.calls) == 2
