from types import SimpleNamespace

from ragc.llm.fake import FakeLLM, FakeLLMResult
from ragc.schemas import Hit
from ragc.systems.s4_llm_centric import S4LLMCentricSystem


class StubRetriever:
    def __init__(self):
        self.chunks = {
            f"c{i}": {"chunk_id": f"c{i}", "header": f"Section {i}", "text": f"Evidence {i}.", "source_type": "vendor"}
            for i in range(1, 9)
        }

    def dense(self, query: str, k: int, source_types=None):
        if "missing" in query.lower():
            return [Hit(chunk_id="c8", score=1.0, rank=1, via="dense")]
        return [Hit(chunk_id=f"c{i}", score=float(9 - i), rank=i, via="dense") for i in range(1, min(k, 8) + 1)]

    def get_chunk(self, chunk_id: str):
        return self.chunks.get(chunk_id, {})


class SequenceLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self._responses = iter([
            '{"retrieve": true, "reason": "need docs"}',
            '{"notes": [{"chunk_id": "c1", "relevance": "relevant", "note": "version 10.2.9-h1", "answers_directly": true}, {"chunk_id": "c2", "relevance": "relevant", "note": "patched version", "answers_directly": false}, {"chunk_id": "c3", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c4", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c5", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c6", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c7", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c8", "relevance": "irrelevant", "note": "", "answers_directly": false}] }',
            "The affected version is 10.2.9-h1. [c1]",
            '{"supported": "partial", "useful": 3, "unsupported_claims": ["not all versions"], "missing_parts": ["missing"]}',
            'The fixed version is 10.2.9-h1 and the patch is available. [c1]'
        ])

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "system": system})
        text = next(self._responses)
        return FakeLLMResult(text=text, prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)


def test_s4_llm_centric_gate_notes_critique_and_repair_flow():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1), paths=SimpleNamespace())
    llm = SequenceLLM()
    system = S4LLMCentricSystem(cfg=cfg, llm=llm, retriever=StubRetriever())

    result = system.run("d01", "Which version is affected?")

    assert result.system == "S4"
    assert len(llm.calls) == 5
    assert len(result.final_evidence) >= 1
    assert any(item.chunk_id == "c8" for item in result.final_evidence)


class IrrelevantLLM(SequenceLLM):
    def __init__(self):
        super().__init__()
        self._responses = iter([
            '{"retrieve": true, "reason": "need docs"}',
            '{"notes": [{"chunk_id": "c1", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c2", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c3", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c4", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c5", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c6", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c7", "relevance": "irrelevant", "note": "", "answers_directly": false}, {"chunk_id": "c8", "relevance": "irrelevant", "note": "", "answers_directly": false}] }',
            'INSUFFICIENT_EVIDENCE',
        ])


def test_s4_llm_centric_irrelevant_evidence_yields_empty_selection():
    cfg = SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1), paths=SimpleNamespace())
    llm = IrrelevantLLM()
    system = S4LLMCentricSystem(cfg=cfg, llm=llm, retriever=StubRetriever())

    result = system.run("d02", "Which version is affected?")

    assert result.system == "S4"
    assert result.final_evidence == []
