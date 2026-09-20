from types import SimpleNamespace

from ragc.llm.fake import FakeLLM, FakeLLMResult
from ragc.schemas import Hit


class StubRetriever:
    def __init__(self):
        self.chunks = {
            "c1": {"chunk_id": "c1", "header": "Vendor advisory", "text": "The affected version is 10.2.9-h1.", "source_type": "vendor"},
            "c2": {"chunk_id": "c2", "header": "NVD", "text": "This is a command injection flaw.", "source_type": "nvd"},
            "c3": {"chunk_id": "c3", "header": "KEV", "text": "Exploit is active.", "source_type": "kev"},
        }

    def dense(self, query: str, k: int):
        return [Hit(chunk_id="c1", score=5.0, rank=1, via="dense"), Hit(chunk_id="c2", score=4.0, rank=2, via="dense"), Hit(chunk_id="c3", score=3.0, rank=3, via="dense")]

    def get_chunk(self, chunk_id: str):
        return self.chunks.get(chunk_id, {})


class S6bLLM(FakeLLM):
    def __init__(self, *, aligned: bool = True):
        super().__init__()
        self.aligned = aligned
        self.responses = iter([
            '{"scores":[{"chunk_id":"c1","score":5,"reason":"directly answers"},{"chunk_id":"c2","score":4,"reason":"supports it"},{"chunk_id":"c3","score":3,"reason":"related"}]}',
            '{"notes":[{"fact":"version is 10.2.9-h1","chunk_ids":["c1"]}], "conflicts":[], "gaps":[]}',
            'Affected version is 10.2.9-h1. [c1]',
            '{"aligned": true, "critiques": []}',
        ])

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "system": system})
        if self.aligned:
            text = next(self.responses)
            return FakeLLMResult(text=text, prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)
        # non-aligned path: first reconcile says not aligned, then final regenerate again, then second reconcile says aligned
        if not hasattr(self, "_non_aligned_state"):
            self._non_aligned_state = iter([
                '{"scores":[{"chunk_id":"c1","score":5,"reason":"directly answers"},{"chunk_id":"c2","score":4,"reason":"supports it"},{"chunk_id":"c3","score":3,"reason":"related"}]}',
                '{"notes":[{"fact":"version is 10.2.9-h1","chunk_ids":["c1"]}], "conflicts":[], "gaps":[]}',
                'Affected version is 10.2.9-h1. [c1]',
                '{"aligned": false, "critiques":["missing notes"]}',
                'Affected version is 10.2.9-h1. [c1]',
                '{"aligned": true, "critiques": []}',
            ])
            self.responses = self._non_aligned_state
        text = next(self.responses)
        return FakeLLMResult(text=text, prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)


def test_s6b_aligned_on_first_reconciliation_uses_four_calls():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1), paths=SimpleNamespace())
    llm = S6bLLM(aligned=True)
    retriever = StubRetriever()

    from ragc.systems.s6b_utilization import S6bUtilizationSystem

    system = S6bUtilizationSystem(cfg=cfg, llm=llm, retriever=retriever)
    result = system.run("d01", "Which version is affected?")

    assert result.system == "S6b"
    assert result.final_evidence[0].chunk_id == "c1"
    assert system.budget.used == 4
    assert len(llm.calls) == 4


def test_s6b_never_aligned_uses_budget_cap():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1), paths=SimpleNamespace())
    llm = S6bLLM(aligned=False)
    retriever = StubRetriever()

    from ragc.systems.s6b_utilization import S6bUtilizationSystem

    system = S6bUtilizationSystem(cfg=cfg, llm=llm, retriever=retriever)
    result = system.run("d02", "Which version is affected?")

    assert result.system == "S6b"
    assert system.budget.used <= 7
    assert len(llm.calls) <= 7
    assert result.final_evidence
