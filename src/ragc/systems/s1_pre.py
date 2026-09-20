"""S1: pre-retrieval query rewriting plus HyDE.

The system follows the benchmark contract exactly: one JSON call to generate a set of
rewrites and a hypothetical document, then dense/BM25 retrieval over every query, RRF
fusion of the ranked lists, and one final generation pass.
"""

from __future__ import annotations

from ragc.generation.generator import generate
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
                    "You rewrite search queries for a collection of cybersecurity documents.",
                    f"Question: {question}\nReturn JSON with rewrites and hyde.",
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
        rewrites = [str(r).strip() for r in rewrites if str(r).strip()]
        rewrites = rewrites[:3]
        hyde = parsed.get("hyde", "") if isinstance(parsed, dict) else ""
        hyde = str(hyde).strip()

        queries = [question] + rewrites
        all_lists: list[list[str]] = []
        candidate_ids: list[str] = []

        if self.retriever is not None:
            for q in queries:
                dense_hits = self.retriever.dense(q, 10)
                bm25_hits = self.retriever.bm25(q, 10)
                all_lists.append([hit.chunk_id for hit in dense_hits])
                all_lists.append([hit.chunk_id for hit in bm25_hits])
                candidate_ids.extend(hit.chunk_id for hit in dense_hits)
                candidate_ids.extend(hit.chunk_id for hit in bm25_hits)

            if hyde:
                hyde_hits = self.retriever.passage(hyde, 10) if hasattr(self.retriever, "passage") else self.retriever.dense(hyde, 10)
                all_lists.append([hit.chunk_id for hit in hyde_hits])
                candidate_ids.extend(hit.chunk_id for hit in hyde_hits)

        fused = rrf(all_lists, k=60) if all_lists else []
        ordered_ids = list(dict.fromkeys(fused))[:5]

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

        self.budget.spend(1)
        answer, shown = generate(self.llm, question, final_evidence, tag="s1")
        abstained = answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_ids=[item.chunk_id for item in shown],
            candidate_ids=list(dict.fromkeys(candidate_ids)),
            final_evidence=shown,
            usage={"llm_calls": self.budget.used, "prompt_tokens": 0, "completion_tokens": 0, "local_s": 0.0, "wall_s": 0.0},
            trace=self.trace,
        )
