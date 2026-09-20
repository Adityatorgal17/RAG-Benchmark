from types import SimpleNamespace

from ragc.llm.fake import FakeLLM
from ragc.systems.post_refine import post_refine
from ragc.systems.s2_post import S2PostSystem, A2aSystem, A2bSystem
from scripts.calibrate_tau import calibrate_tau


class StubRetriever:
    def __init__(self):
        self.chunks = {
            "c1": {"chunk_id": "c1", "header": "Vendor", "text": "The affected version is 10.2.9-h1.", "source_type": "vendor"},
            "c2": {"chunk_id": "c2", "header": "NVD", "text": "A command injection flaw affects PAN-OS.", "source_type": "nvd"},
            "c3": {"chunk_id": "c3", "header": "KEV", "text": "The issue is exploited in the wild.", "source_type": "kev"},
        }

    def hybrid(self, query, k_dense, k_bm25, k_out, source_types=None):
        return [type("Hit", (), {"chunk_id": chunk_id, "score": 1.0, "rank": idx + 1, "via": "hybrid"})() for idx, chunk_id in enumerate(["c1", "c2", "c3"])]

    def get_chunk(self, chunk_id):
        return self.chunks.get(chunk_id, {})


class StubReranker:
    def score(self, pairs):
        scores = []
        for q, p in pairs:
            key = (q, p)
            if "10.2.9-h1" in p:
                scores.append(0.91)
            elif "command injection" in p:
                scores.append(0.70)
            elif "exploited in the wild" in p:
                scores.append(0.40)
            else:
                scores.append(0.10)
        return scores


def test_post_refine_keeps_top_two_when_all_scores_below_tau():
    cfg = SimpleNamespace(post=SimpleNamespace(tau=0.80, compress_tokens=1500), paths=SimpleNamespace())
    items = post_refine(
        StubReranker(),
        StubRetriever(),
        ["Which version is affected?"],
        ["c1", "c2", "c3"],
        None,
        cfg,
        do_filter=True,
        do_compress=False,
    )
    assert [item.chunk_id for item in items] == ["c1", "c2"]


def test_s2_and_ablation_variants_run_and_keep_budget():
    cfg = SimpleNamespace(
        budget=SimpleNamespace(max_llm_calls=8, reserve_final=1),
        post=SimpleNamespace(tau=0.80, compress_tokens=1500),
        paths=SimpleNamespace(),
    )
    retriever = StubRetriever()
    llm = FakeLLM()

    s2 = S2PostSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=StubReranker())
    result = s2.run("d01", "Which version is affected?")
    assert result.system == "S2"
    assert len(result.candidate_ids) == 3
    assert len(result.final_evidence) >= 1
    assert s2.budget.used == 1

    a2a = A2aSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=StubReranker())
    a2a_result = a2a.run("d02", "What is the weakness?")
    assert a2a_result.system == "A2a"

    a2b = A2bSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=StubReranker())
    a2b_result = a2b.run("d03", "What is the exploitation status?")
    assert a2b_result.system == "A2b"


def test_calibrate_tau_clips_to_plan_bounds():
    tau = calibrate_tau([0.01, 0.05, 0.2, 0.4])
    assert 0.02 <= tau <= 0.30
    assert tau == 0.02
