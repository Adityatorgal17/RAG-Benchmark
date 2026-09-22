"""S6a: acquisition-driven multi-agent retrieval loop.

This follows the plan's Retrieval-Decision -> Query -> Sufficiency orchestration:
- decide whether retrieval is needed and the search complexity;
- issue up to a bounded number of focused queries;
- stop when a duplicate query is proposed or the sufficiency agent says enough;
- answer using the best accumulated evidence.
"""

from __future__ import annotations

import json
import re

from ragc.generation.generator import extract_citations
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System


class S6aAcquisitionSystem(System):
    name = "S6a"

    def _normalize_query(self, query: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (query or "").lower())

    def _parse_json_object(self, payload):
        if isinstance(payload, dict):
            return payload
        if not isinstance(payload, str):
            return {}
        text = payload.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        return {}

    def run(self, qid: str, question: str) -> SystemResult:
        decision = {"retrieve": True, "complexity": "simple"}
        if self.llm is not None:
            try:
                payload = self.call_json(
                    "s6a_decision",
                    "You are the Retrieval-Decision agent. Decide whether a search is needed and how complex the retrieval loop should be.",
                    "Question: " + question + "\nReturn JSON with retrieve boolean and complexity one of simple, multi_fact, multi_hop.",
                    ["retrieve", "complexity"],
                )
                decision = self._parse_json_object(payload.get("text") if isinstance(payload, dict) else payload)
            except Exception as exc:
                self.record_error("s6a_decision_error", exc)
                decision = {"retrieve": True, "complexity": "simple"}

        if not bool(decision.get("retrieve", True)):
            self.budget.spend_final(1)
            answer, shown = self.generate_answer(question, [], no_evidence_mode=True, tag="s6a_no_retrieve")
            return SystemResult(
                system=self.name,
                qid=qid,
                question=question,
                answer=answer,
                abstained=answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"),
                cited_ids=[],
                candidate_ids=[],
                final_evidence=shown,
                usage=self.usage(),
                trace=self.trace,
                errors=self.errors,
            )

        complexity = str(decision.get("complexity", "")).strip().lower()
        configured_max_rounds = int(self.retrieval_param("s6a", "max_rounds", 3))
        simple_rounds = int(self.retrieval_param("s6a", "simple_rounds", 1))
        multi_fact_rounds = int(self.retrieval_param("s6a", "multi_fact_rounds", 2))
        multi_hop_rounds = int(self.retrieval_param("s6a", "multi_hop_rounds", configured_max_rounds))
        max_rounds = {"simple": simple_rounds, "multi_fact": multi_fact_rounds, "multi_hop": multi_hop_rounds}.get(complexity, multi_fact_rounds)
        max_rounds = max(1, min(max_rounds, configured_max_rounds))
        dense_k = int(self.retrieval_param("s6a", "dense", 30))
        bm25_k = int(self.retrieval_param("s6a", "bm25", 30))
        search_k = int(self.retrieval_param("s6a", "k", 5))
        final_k = int(self.retrieval_param("s6a", "final_k", 6))

        queries: list[str] = []
        evidence_map: dict[str, dict] = {}
        missing = ""

        for round_idx in range(1, max_rounds + 1):
            try:
                payload = self.call_json(
                    "s6a_query",
                    "You are the Query agent. Write one focused search query for a document collection.",
                    "Question: " + question + "\nPrevious queries: " + (", ".join(queries) if queries else "(none)") + "\nMissing information: " + (missing or "(none)") + "\nReturn JSON with one query.",
                    ["query"],
                )
                step = self._parse_json_object(payload.get("text") if isinstance(payload, dict) else payload)
            except Exception as exc:
                self.record_error("s6a_query_error", exc)
                break

            query = str(step.get("query", "")).strip()
            if not query:
                break

            norm = self._normalize_query(query)
            if norm and norm in {self._normalize_query(q) for q in queries}:
                break

            queries.append(query)

            if self.retriever is None:
                break

            hits = self.retriever.hybrid(query, dense_k, bm25_k, search_k)
            for hit in hits:
                chunk = self.retriever.get_chunk(hit.chunk_id)
                if not chunk:
                    continue
                score = float(getattr(hit, "score", 0.0) or 0.0)
                current = evidence_map.get(hit.chunk_id)
                if current is None or score > float(current.get("score", -1.0)):
                    evidence_map[hit.chunk_id] = {
                        "chunk_id": hit.chunk_id,
                        "header": chunk.get("header", ""),
                        "text": chunk.get("text", ""),
                        "source_type": chunk.get("source_type", ""),
                        "score": score,
                    }

            if round_idx >= max_rounds:
                break

            try:
                sufficiency_payload = self.call_json(
                    "s6a_suff",
                    "You are the Sufficiency agent. Judge whether the retrieved evidence is enough to answer the question completely.",
                    "Question: " + question + f"\nEvidence (top {final_k} by retrieval score):\n" + "\n\n".join(
                        f"{item['chunk_id']}: {item['header']}\n{(item['text'] or '')[:600]}"
                        for item in sorted(evidence_map.values(), key=lambda d: float(d.get('score', 0.0)), reverse=True)[:final_k]
                    ) + "\nReturn JSON with sufficient boolean and missing note when insufficient.",
                    ["sufficient", "missing"],
                )
                sufficiency = self._parse_json_object(sufficiency_payload.get("text") if isinstance(sufficiency_payload, dict) else sufficiency_payload)
            except Exception as exc:
                self.record_error("s6a_sufficiency_error", exc)
                sufficiency = {"sufficient": True, "missing": ""}

            if bool(sufficiency.get("sufficient", False)):
                break

            missing = str(sufficiency.get("missing", "")).strip()

        ranked = sorted(evidence_map.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)[:final_k]
        final_evidence = [
            EvidenceItem(
                chunk_id=str(item["chunk_id"]),
                header=str(item.get("header", "")),
                text=str(item.get("text", "")),
                source_type=str(item.get("source_type", "")),
            )
            for item in ranked
        ]

        if not final_evidence:
            self.budget.spend_final(1)
            answer, shown = self.generate_answer(question, [], no_evidence_mode=True, tag="s6a_no_evidence")
            return SystemResult(
                system=self.name,
                qid=qid,
                question=question,
                answer=answer,
                abstained=answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"),
                cited_ids=[],
                candidate_ids=[],
                final_evidence=shown,
                usage=self.usage(),
                trace=self.trace,
                errors=self.errors,
            )

        self.budget.spend_final(1)
        answer, shown = self.generate_answer(question, final_evidence, tag="s6a_final")
        abstained = answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_ids=extract_citations(answer),
            candidate_ids=list(evidence_map.keys()),
            final_evidence=shown,
            usage=self.usage(),
            trace=self.trace,
            errors=self.errors,
        )
