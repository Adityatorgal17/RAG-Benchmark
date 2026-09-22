"""S2 post-retrieval system and its S2 ablation variants.

This module implements the core post-retrieval pipeline from the plan: hybrid retrieval,
reranking, optional tau-based filtering, optional sentence compression, and one final
LLM generation pass. The ablation variants A2a and A2b reuse the same logic with flags.
"""

from __future__ import annotations

from ragc.generation.generator import extract_citations
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System
from ragc.systems.post_refine import post_refine


class S2PostSystem(System):
    name = "S2"

    def __init__(self, cfg=None, llm=None, retriever=None, reranker=None, *, do_filter=True, do_compress=True, **kwargs):
        super().__init__(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs)
        self.do_filter = do_filter
        self.do_compress = do_compress

    def run(self, qid: str, question: str) -> SystemResult:
        if self.retriever is None:
            hits = []
        else:
            dense_k = int(self.retrieval_param("s2", "dense", 30))
            bm25_k = int(self.retrieval_param("s2", "bm25", 30))
            candidates_k = int(self.retrieval_param("s2", "candidates", 30))
            hits = self.retriever.hybrid(question, dense_k, bm25_k, candidates_k)
        candidates = [hit.chunk_id for hit in hits]
        items = post_refine(self.reranker, self.retriever, [question], candidates, None, self.cfg, do_filter=self.do_filter, do_compress=self.do_compress)
        self.budget.spend_final(1)
        answer, shown = self.generate_answer(question, items, tag="s2")
        abstained = answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_ids=extract_citations(answer),
            candidate_ids=candidates,
            final_evidence=shown,
            usage=self.usage(),
            trace=self.trace,
            errors=self.errors,
        )


class A2aSystem(S2PostSystem):
    name = "A2a"

    def __init__(self, cfg=None, llm=None, retriever=None, reranker=None, **kwargs):
        super().__init__(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, do_filter=False, do_compress=False, **kwargs)


class A2bSystem(S2PostSystem):
    name = "A2b"

    def __init__(self, cfg=None, llm=None, retriever=None, reranker=None, **kwargs):
        super().__init__(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, do_filter=True, do_compress=False, **kwargs)
