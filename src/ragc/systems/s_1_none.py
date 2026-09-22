"""S-1: no-retrieval baseline.

This baseline measures parametric knowledge leakage by answering directly from the
LLM without any retrieved evidence. It uses the no-evidence generator path and keeps
its call budget to a single final generation turn.
"""

from __future__ import annotations

from ragc.generation.generator import extract_citations
from ragc.schemas import SystemResult
from ragc.systems.base import System


class S1NoRetrievalSystem(System):
    name = "S-1"

    def run(self, qid: str, question: str) -> SystemResult:
        self.budget.spend_final(1)
        answer, shown = self.generate_answer(question, [], no_evidence_mode=True, tag="s1")
        abstained = answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_ids=[],
            candidate_ids=[],
            final_evidence=[],
            usage=self.usage(),
            trace=self.trace,
            errors=self.errors,
        )
