"""S0: naive retrieval-and-read baseline.

This baseline performs a single dense retrieval pass, converts the top-k hits into
compact evidence items, and then calls the shared generator once. It is the canonical
reference system for the naive RAG paradigm in the benchmark.
"""

from __future__ import annotations

from ragc.generation.generator import extract_citations
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System


class S0NaiveSystem(System):
    name = "S0"

    def run(self, qid: str, question: str) -> SystemResult:
        k = int(self.cfg_get("retrieval", "s0_k", default=5))
        hits = self.retriever.dense(question, k=k) if self.retriever is not None else []
        final_evidence: list[EvidenceItem] = []
        for hit in hits:
            chunk = self.retriever.get_chunk(hit.chunk_id) if self.retriever is not None else {}
            if not chunk:
                continue
            final_evidence.append(
                EvidenceItem(
                    chunk_id=hit.chunk_id,
                    header=chunk.get("header", ""),
                    text=chunk.get("text", ""),
                    source_type=chunk.get("source_type", ""),
                )
            )

        self.budget.spend_final(1)
        answer, shown = self.generate_answer(question, final_evidence, tag="s0")
        abstained = answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_ids=extract_citations(answer),
            candidate_ids=[hit.chunk_id for hit in hits],
            final_evidence=shown,
            usage=self.usage(),
            trace=self.trace,
            errors=self.errors,
        )
