"""S5: single-agent search loop with scratchpad-based reasoning.

This follows the plan's ReAct-style loop: an LLM agent can issue up to 5 actions,
search with a hybrid tool when needed, track useful chunk ids, and finally generate an
answer from the evidence it judged useful.
"""

from __future__ import annotations

import json
import re

from ragc.generation.generator import extract_citations
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System


class S5SingleAgentSystem(System):
    name = "S5"

    def _extract_action_and_query(self, step: dict) -> tuple[str, str]:
        """Extract action and optional query from agent step.
        
        Args:
            step: The parsed agent response
            
        Returns:
            Tuple of (action, query) where query may be empty
        """
        action = str(step.get("action", "")).strip().lower()
        query = str(step.get("query", "")).strip()
        return action, query

    def _validate_action(self, action: str, valid_actions: list[str]) -> bool:
        """Validate that action is one of the allowed actions.
        
        Args:
            action: The action to validate
            valid_actions: List of valid actions
            
        Returns:
            True if action is valid, False otherwise
        """
        return action in valid_actions

    def _normalize_query(self, query: str) -> str:
        """Normalize query for duplicate detection.
        
        Args:
            query: The query to normalize
            
        Returns:
            Normalized query string
        """
        return re.sub(r"[^a-z0-9]+", "", (query or "").lower())

    def _validate_query(self, query: str, previous_queries: list[str]) -> bool:
        """Validate query is not empty and not duplicate.
        
        Args:
            query: The query to validate
            previous_queries: List of previously issued queries
            
        Returns:
            True if query is valid, False otherwise
        """
        if not query:
            return False
        
        norm = self._normalize_query(query)
        if not norm:
            return False
        
        for prev_query in previous_queries:
            if norm == self._normalize_query(prev_query):
                return False
        
        return True

    def run(self, qid: str, question: str) -> SystemResult:
        scratch: list[dict[str, str]] = []
        useful: list[str] = []
        queries: list[str] = []
        latest: list[dict] = []
        candidate_ids: list[str] = []
        search_count = 0
        dense_k = int(self.retrieval_param("s5", "dense", 30))
        bm25_k = int(self.retrieval_param("s5", "bm25", 30))
        tool_k = int(self.retrieval_param("s5", "k", 4))
        max_searches = int(self.retrieval_param("s5", "max_searches", 4))
        max_agent_calls = int(self.retrieval_param("s5", "max_agent_calls", 5))

        for step_num in range(max_agent_calls):
            if self.llm is None:
                break
            
            try:
                # Summarise latest results compactly to keep prompt short.
                # Full JSON dumps of chunk texts made the prompt too long and
                # caused the model to try tool-calling on subsequent steps.
                if latest:
                    latest_summary = "\n".join(
                        f"- [{item.get('chunk_id','')}] {item.get('header','')} | {(item.get('text') or '')[:80]}..."
                        for item in latest[:tool_k]
                    )
                else:
                    latest_summary = "(none yet — issue a search action first)"

                scratch_summary = "\n".join(
                    f"- [{s.get('chunk_id','')}] {s.get('fact','')}"
                    for s in scratch[-6:]  # last 6 facts only
                ) if scratch else "(empty)"

                payload = self.call_json(
                    f"s5_agent_step_{step_num}",
                    "You are a research agent answering a cybersecurity question using a fixed document collection. Respond ONLY with a raw JSON object — no markdown, no function calls, no tools.",
                    f"Question: {question}\n\nFacts gathered so far:\n{scratch_summary}\n\nPrevious search queries: {', '.join(queries) if queries else '(none)'}\n\nLatest search results:\n{latest_summary}\n\nChoose action=\"search\" to look up more, or action=\"answer\" when ready.\n\nReturn exactly: {{\"thought\": \"...\", \"action\": \"search\" or \"answer\", \"query\": \"search query (only when action=search)\"}}",
                    ["thought", "action"],
                )
                text = payload.get("text") or ""
                step = None
                if isinstance(text, str) and text.strip():
                    try:
                        step = json.loads(text)
                    except json.JSONDecodeError:
                        # Base class already retried — try regex extraction as last resort
                        import re as _re
                        m = _re.search(r'\{.*?\}', text, _re.DOTALL)
                        if m:
                            try:
                                step = json.loads(m.group(0))
                            except json.JSONDecodeError:
                                pass
                elif isinstance(text, dict):
                    step = text
            except Exception as e:
                # Only log unexpected errors (quota, network).
                # json_validate_failed / tool_use_failed are handled by base class.
                err_str = str(e)
                if not any(k in err_str for k in ("json_validate_failed", "tool_use_failed")):
                    self.record_error(f"s5_agent_error_{step_num}", e)
                step = None

            if not isinstance(step, dict):
                # Could not get a valid step response — do a fallback hybrid search
                # with the question (or last query) and break out of the agent loop.
                if self.retriever is not None and not queries:
                    try:
                        queries.append(question)
                        latest = [
                            {
                                "chunk_id": hit.chunk_id,
                                "header": self.retriever.get_chunk(hit.chunk_id).get("header", ""),
                                "text": self.retriever.get_chunk(hit.chunk_id).get("text", ""),
                                "score": hit.score,
                                "rank": hit.rank,
                                "via": hit.via,
                            }
                            for hit in self.retriever.hybrid(question, dense_k, bm25_k, tool_k)
                        ]
                        for item in latest:
                            cid = item.get("chunk_id")
                            if cid and cid not in candidate_ids:
                                candidate_ids.append(cid)
                        self.trace.append({"tag": f"s5_fallback_search_{step_num}", "query": question})
                    except Exception as fe:
                        self.record_error(f"s5_fallback_error_{step_num}", fe)
                break

            thought = step.get("thought", "")
            new_facts = step.get("new_facts", [])
            useful_ids = step.get("useful_chunk_ids", [])
            
            if isinstance(new_facts, list):
                for fact in new_facts:
                    if isinstance(fact, dict):
                        chunk_id = str(fact.get("chunk_id", "")).strip()
                        fact_text = str(fact.get("fact", "")).strip()
                        if chunk_id and fact_text:
                            scratch.append({"fact": fact_text, "chunk_id": chunk_id})

            if isinstance(useful_ids, list):
                for chunk_id in useful_ids:
                    chunk_id = str(chunk_id).strip()
                    if chunk_id and chunk_id not in useful:
                        useful.append(chunk_id)

            action, query = self._extract_action_and_query(step)
            
            if not self._validate_action(action, ["search", "answer"]):
                # If we already have evidence/queries, treat unrecognized action as "answer"
                if queries or useful:
                    self.trace.append({"tag": f"s5_invalid_action_defaulted_to_answer", "action": action, "step": step_num})
                    break
                # No evidence yet — do a fallback search and break
                if self.retriever is not None:
                    try:
                        queries.append(question)
                        latest = [{"chunk_id": h.chunk_id, "header": self.retriever.get_chunk(h.chunk_id).get("header",""), "text": self.retriever.get_chunk(h.chunk_id).get("text",""), "score": h.score, "rank": h.rank, "via": h.via} for h in self.retriever.hybrid(question, dense_k, bm25_k, tool_k)]
                        for item in latest:
                            cid = item.get("chunk_id")
                            if cid and cid not in candidate_ids:
                                candidate_ids.append(cid)
                    except Exception:
                        pass
                self.trace.append({"tag": f"s5_invalid_action_fallback", "action": action, "step": step_num})
                break

            if action == "answer":
                self.trace.append({
                    "tag": f"s5_agent_chooses_answer_{step_num}",
                    "thought": thought,
                    "useful_chunks": len(useful),
                    "queries_issued": len(queries),
                    "step": step_num
                })
                break

            if action == "search":
                if not self._validate_query(query, queries):
                    # Duplicate or empty query — skip silently and continue the loop
                    self.trace.append({"tag": f"s5_skip_duplicate_query", "query": query, "step": step_num})
                    continue

                norm = self._normalize_query(query)
                if search_count >= max_searches:
                    self.trace.append({
                        "tag": f"s5_agent_max_searches_{step_num}",
                        "search_count": search_count,
                        "max_searches": max_searches,
                        "step": step_num
                    })
                    break

                queries.append(query)
                search_count += 1
                
                if self.retriever is not None:
                    latest = [
                        {
                            "chunk_id": hit.chunk_id,
                            "header": self.retriever.get_chunk(hit.chunk_id).get("header", ""),
                            "text": self.retriever.get_chunk(hit.chunk_id).get("text", ""),
                            "score": hit.score,
                            "rank": hit.rank,
                            "via": hit.via
                        }
                        for hit in self.retriever.hybrid(query, dense_k, bm25_k, tool_k)
                    ]
                    for item in latest:
                        chunk_id = item.get("chunk_id")
                        if chunk_id and chunk_id not in candidate_ids:
                            candidate_ids.append(chunk_id)
                
                self.trace.append({
                    "tag": f"s5_agent_search_{step_num}",
                    "query": query,
                    "results_count": len(latest) if latest else 0,
                    "useful_chunks": len(useful),
                    "step": step_num,
                    "n_facts": len([f for f in scratch if f.get("chunk_id") in [h.get("chunk_id") for h in (latest or [])]]) if latest else 0
                })

        final_evidence: list[EvidenceItem] = []
        if useful:
            for chunk_id in useful:
                chunk = self.retriever.get_chunk(chunk_id) if self.retriever is not None else {}
                if not chunk:
                    continue
                final_evidence.append(EvidenceItem(
                    chunk_id=chunk_id,
                    header=chunk.get("header", ""),
                    text=chunk.get("text", ""),
                    source_type=chunk.get("source_type", ""),
                    note=f"Useful from S5 scratchpad"
                ))
        elif latest and self.retriever is not None and queries:
            for item in latest[:tool_k]:
                chunk_id = str(item.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                chunk = self.retriever.get_chunk(chunk_id)
                if not chunk:
                    continue
                final_evidence.append(EvidenceItem(
                    chunk_id=chunk_id,
                    header=chunk.get("header", ""),
                    text=chunk.get("text", ""),
                    source_type=chunk.get("source_type", ""),
                    note=f"Latest search result from query '{queries[-1]}'"
                ))
            if not final_evidence:
                for hit in self.retriever.hybrid(queries[-1], dense_k, bm25_k, tool_k):
                    if hit.chunk_id not in candidate_ids:
                        candidate_ids.append(hit.chunk_id)
                    chunk = self.retriever.get_chunk(hit.chunk_id)
                    if not chunk:
                        continue
                    final_evidence.append(EvidenceItem(
                        chunk_id=hit.chunk_id,
                        header=chunk.get("header", ""),
                        text=chunk.get("text", ""),
                        source_type=chunk.get("source_type", ""),
                        note=f"Latest search result from query '{queries[-1]}'"
                    ))

        if not useful and not latest and not queries:
            self.budget.spend_final(1)
            answer, shown = self.generate_answer(question, [], no_evidence_mode=True, tag="s5_no_search")
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
        answer, shown = self.generate_answer(question, final_evidence, tag="s5_final")
        abstained = answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        
        self.trace.append({
            "tag": "s5_final_generation",
            "final_evidence_count": len(final_evidence),
            "useful_chunks_before": len(useful),
            "total_searches": search_count,
            "queries_issued": len(queries),
            "answer_length": len(answer),
            "abstained": abstained
        })
        
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_ids=extract_citations(answer),
            candidate_ids=candidate_ids,
            final_evidence=shown,
            usage=self.usage(),
            trace=self.trace,
            errors=self.errors,
        )
