"""S5: single-agent search loop with scratchpad-based reasoning.

This follows the plan's ReAct-style loop: an LLM agent can issue up to 5 actions,
search with a hybrid tool when needed, track useful chunk ids, and finally generate an
answer from the evidence it judged useful.
"""

from __future__ import annotations

import json
import re

from ragc.generation.generator import generate
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System


class S5SingleAgentSystem(System):
    name = "S5"

    def _normalize_query(self, query: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (query or "").lower())

    def run(self, qid: str, question: str) -> SystemResult:
        scratch: list[dict[str, str]] = []
        useful: list[str] = []
        queries: list[str] = []
        latest: list[dict] = []
        search_count = 0

        for _ in range(5):
            if self.llm is None:
                break
            try:
                payload = self.call_json(
                    "s5_agent",
                    "You are a research agent investigating a cybersecurity question with a search tool over a fixed document collection.",
                    "Question: " + question + "\n\nScratchpad (facts gathered so far, with chunk ids):\n" + (json.dumps(scratch) if scratch else "(empty)") + "\n\nPrevious search queries: " + (", ".join(queries) if queries else "(none)") + "\n\nLatest search results:\n" + (json.dumps(latest) if latest else "(none yet)") + "\n\nReturn JSON with thought, new_facts, useful_chunk_ids, action, and query when action=search.",
                    ["thought", "new_facts", "useful_chunk_ids", "action"],
                )
                text = payload.get("text") or "{}"
                if isinstance(text, str):
                    step = json.loads(text)
                else:
                    step = text
            except Exception:
                break

            if not isinstance(step, dict):
                break

            if isinstance(step.get("new_facts"), list):
                for fact in step["new_facts"]:
                    if isinstance(fact, dict):
                        chunk_id = str(fact.get("chunk_id", "")).strip()
                        fact_text = str(fact.get("fact", "")).strip()
                        if chunk_id and fact_text:
                            scratch.append({"fact": fact_text, "chunk_id": chunk_id})

            useful_ids = step.get("useful_chunk_ids") or []
            if isinstance(useful_ids, list):
                for chunk_id in useful_ids:
                    chunk_id = str(chunk_id).strip()
                    if chunk_id and chunk_id not in useful:
                        useful.append(chunk_id)

            action = str(step.get("action", "answer")).strip().lower()
            query = str(step.get("query", "")).strip()
            if action == "answer":
                break

            if action == "search":
                if not query:
                    break
                norm = self._normalize_query(query)
                if norm and (norm in {self._normalize_query(q) for q in queries} or search_count >= 4):
                    break
                queries.append(query)
                if self.retriever is not None:
                    latest = [
                        {"chunk_id": hit.chunk_id, "header": self.retriever.get_chunk(hit.chunk_id).get("header", ""), "text": self.retriever.get_chunk(hit.chunk_id).get("text", "")}
                        for hit in self.retriever.hybrid(query, 30, 30, 4)
                    ]
                search_count += 1
                continue

            break

        final_evidence: list[EvidenceItem] = []
        if useful:
            for chunk_id in useful:
                chunk = self.retriever.get_chunk(chunk_id) if self.retriever is not None else {}
                if not chunk:
                    continue
                final_evidence.append(EvidenceItem(chunk_id=chunk_id, header=chunk.get("header", ""), text=chunk.get("text", ""), source_type=chunk.get("source_type", "")))
        elif latest and self.retriever is not None:
            for hit in self.retriever.hybrid(queries[-1], 30, 30, 3) if queries else []:
                chunk = self.retriever.get_chunk(hit.chunk_id)
                if not chunk:
                    continue
                final_evidence.append(EvidenceItem(chunk_id=hit.chunk_id, header=chunk.get("header", ""), text=chunk.get("text", ""), source_type=chunk.get("source_type", "")))

        if not useful and not latest and not queries:
            self.budget.spend(1)
            answer, shown = generate(self.llm, question, [], no_evidence_mode=True, tag="s5_no_search")
            return SystemResult(
                system=self.name,
                qid=qid,
                question=question,
                answer=answer,
                abstained=answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"),
                cited_ids=[],
                candidate_ids=[],
                final_evidence=shown,
                usage={"llm_calls": self.budget.used, "prompt_tokens": 0, "completion_tokens": 0, "local_s": 0.0, "wall_s": 0.0},
                trace=self.trace,
            )

        self.budget.spend(1)
        answer, shown = generate(self.llm, question, final_evidence, tag="s5_final")
        abstained = answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_ids=[item.chunk_id for item in shown],
            candidate_ids=[item.get("chunk_id") for item in latest] if latest else [],
            final_evidence=shown,
            usage={"llm_calls": self.budget.used, "prompt_tokens": 0, "completion_tokens": 0, "local_s": 0.0, "wall_s": 0.0},
            trace=self.trace,
        )
