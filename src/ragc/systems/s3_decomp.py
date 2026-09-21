"""S3: query decomposition + post-retrieval refinement.

This follows the benchmark contract: one JSON decomposer call, a hybrid retrieval pass for
both the original question and each sub-query, then rerank/filter/compress those candidates
before a final answer generation call.
"""

from __future__ import annotations

from ragc.generation.generator import extract_citations, generate
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System
from ragc.systems.post_refine import post_refine


class S3DecompSystem(System):
    name = "S3"

    def run(self, qid: str, question: str) -> SystemResult:
        if self.llm is not None:
            try:
                payload = self.call_json(
                    "s3_decomp",
                    "You decompose questions about cybersecurity vulnerabilities for a document collection.",
                    f"Question: {question}\nReturn JSON with sub_queries.",
                    ["sub_queries"],
                )
                data = payload.get("text") or "{}"
                if isinstance(data, str):
                    import json

                    parsed = json.loads(data)
                else:
                    parsed = data
            except Exception:
                parsed = {}
        else:
            parsed = {}

        sub_queries = parsed.get("sub_queries", []) if isinstance(parsed, dict) else []
        if not isinstance(sub_queries, list):
            sub_queries = []
        sub_queries = [str(q).strip() for q in sub_queries if str(q).strip()]
        sub_queries = sub_queries[:3]
        if not sub_queries:
            sub_queries = [question]

        queries = [question] + sub_queries
        unique_queries: list[str] = []
        seen: set[str] = set()
        for q in queries:
            if q not in seen:
                unique_queries.append(q)
                seen.add(q)
        queries = unique_queries

        query_of: dict[str, list[str]] = {}
        candidates: list[str] = []
        if self.retriever is not None:
            for q in queries:
                hits = self.retriever.hybrid(q, 20, 20, 10)
                for hit in hits:
                    chunk_id = hit.chunk_id
                    query_of.setdefault(chunk_id, [])
                    if q not in query_of[chunk_id]:
                        query_of[chunk_id].append(q)
                    if chunk_id not in candidates:
                        candidates.append(chunk_id)

        items = post_refine(self.reranker, self.retriever, queries, candidates, query_of, self.cfg, do_filter=True, do_compress=True)

        self.budget.spend(1)
        answer, shown = generate(self.llm, question, items, tag="s3")
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
        )
