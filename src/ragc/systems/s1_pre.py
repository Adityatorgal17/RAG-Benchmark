"""S1: pre-retrieval query rewriting plus HyDE.

The system follows the benchmark contract exactly: one JSON call to generate a set of
rewrites and a hypothetical document, then dense/BM25 retrieval over every query, RRF
fusion of the ranked lists, and one final generation pass.
"""

from __future__ import annotations

from ragc.generation.generator import extract_citations
from ragc.retrieval.hybrid import rrf
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System


class S1PreSystem(System):
    name = "S1"

    def run(self, qid: str, question: str) -> SystemResult:
        if self.llm is not None:
            try:
                payload = self.call_json(
                    "s1_rewrite",
                    "You rewrite search queries for a collection of cybersecurity documents. Respond only with a plain JSON object, no tools or function calls.",
                    f'Question: {question}\n\nReturn JSON: {{"rewrites": ["rewrite1", "rewrite2"], "hyde": "hypothetical document passage that would answer the question"}}',
                    ["rewrites", "hyde"],
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

        rewrites = parsed.get("rewrites", []) if isinstance(parsed, dict) else []
        if not isinstance(rewrites, list):
            rewrites = []
        max_rewrites = int(self.retrieval_param("s1", "max_rewrites", 3))
        per_query_dense = int(self.retrieval_param("s1", "per_query_dense", 10))
        per_query_bm25 = int(self.retrieval_param("s1", "per_query_bm25", 10))
        hyde_dense = int(self.retrieval_param("s1", "hyde_dense", 10))
        final_k = int(self.retrieval_param("s1", "final_k", 5))
        rrf_k = int(self.cfg_get("retrieval", "rrf_k", default=60))

        rewrites = [str(r).strip() for r in rewrites if str(r).strip()]
        rewrites = rewrites[:max_rewrites]
        hyde = parsed.get("hyde", "") if isinstance(parsed, dict) else ""
        hyde = str(hyde).strip()

        queries = [question] + rewrites
        all_lists: list[list[str]] = []
        candidate_ids: list[str] = []

        if self.retriever is not None:
            for q in queries:
                dense_hits = self.retriever.dense(q, per_query_dense)
                bm25_hits = self.retriever.bm25(q, per_query_bm25)
                all_lists.append([hit.chunk_id for hit in dense_hits])
                all_lists.append([hit.chunk_id for hit in bm25_hits])
                candidate_ids.extend(hit.chunk_id for hit in dense_hits)
                candidate_ids.extend(hit.chunk_id for hit in bm25_hits)

            if hyde:
                hyde_hits = self.retriever.passage(hyde, hyde_dense) if hasattr(self.retriever, "passage") else self.retriever.dense(hyde, hyde_dense)
                all_lists.append([hit.chunk_id for hit in hyde_hits])
                candidate_ids.extend(hit.chunk_id for hit in hyde_hits)

        fused = rrf(all_lists, k=rrf_k) if all_lists else []
        ordered_ids = list(dict.fromkeys(fused))[:final_k]

        final_evidence: list[EvidenceItem] = []
        for chunk_id in ordered_ids:
            if self.retriever is None:
                continue
            chunk = self.retriever.get_chunk(chunk_id)
            if not chunk:
                continue
            final_evidence.append(
                EvidenceItem(
                    chunk_id=chunk_id,
                    header=chunk.get("header", ""),
                    text=chunk.get("text", ""),
                    source_type=chunk.get("source_type", ""),
                )
            )

        self.budget.spend_final(1)
        answer, shown = self.generate_answer(question, final_evidence, tag="s1")
        abstained = answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_ids=extract_citations(answer),
            candidate_ids=list(dict.fromkeys(candidate_ids)),
            final_evidence=shown,
            usage=self.usage(),
            trace=self.trace,
            errors=self.errors,
        )
