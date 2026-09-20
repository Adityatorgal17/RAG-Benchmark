"""S6c: hybrid dynamic multi-agent system with optional source routing.

This follows the plan's S6c flow:
- planner / router splits the question into sub-questions and source targets;
- retrieval is restricted to those sources when routing is enabled;
- on empty restricted results, retry once without a source filter and log a fallback;
- refinement merges new notes/conflicts into a shared knowledge state;
- decision decides whether the evidence is sufficient;
- final generation uses the cited note evidence plus any flagged conflicts.
"""

from __future__ import annotations

import json

from ragc.generation.generator import generate
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System


class S6cHybridSystem(System):
    name = "S6c"

    def __init__(self, cfg=None, llm=None, retriever=None, reranker=None, *, max_calls: int | None = None, reserve_final: int = 1, use_routing: bool = True):
        super().__init__(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, max_calls=max_calls, reserve_final=reserve_final)
        self.use_routing = use_routing
        self.name = "S6c" if use_routing else "A1"

    def _json(self, payload):
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

    def _plan_subquestions(self, question: str):
        if self.llm is None:
            return [{"id": "s1", "question": question, "sources": []}]

        try:
            if self.use_routing:
                payload = self.call_json(
                    "s6c_plan",
                    "You are the Planner and Router agent. Break a question into sub-questions and route each to the most likely source types.",
                    "Question: " + question + "\nReturn JSON with a 'sub_questions' list where each item has an 'id', a 'question', and 'sources' from: nvd, kev, vendor, threatintel, attack.",
                    ["sub_questions"],
                )
            else:
                payload = self.call_json(
                    "s6c_plan_noroute",
                    "You are the Planner agent. Break a question into self-contained sub-questions without source routing.",
                    "Question: " + question + "\nReturn JSON with a 'sub_questions' list where each item has an 'id' and a 'question'.",
                    ["sub_questions"],
                )
            data = self._json(payload.get("text") if isinstance(payload, dict) else payload)
        except Exception:
            data = {}

        items = data.get("sub_questions", []) if isinstance(data, dict) else []
        if not isinstance(items, list) or not items:
            return [{"id": "s1", "question": question, "sources": []}]

        cleaned = []
        for idx, item in enumerate(items[:4], start=1):
            if not isinstance(item, dict):
                continue
            q = str(item.get("question") or question).strip()
            if not q:
                continue
            sources = item.get("sources") if isinstance(item.get("sources"), list) else []
            if self.use_routing:
                sources = [str(src).strip() for src in sources if str(src).strip()]
                if len(sources) > 2:
                    sources = sources[:2]
            else:
                sources = []
            cleaned.append({"id": str(item.get("id") or f"s{idx}"), "question": q, "sources": sources})
        if not cleaned:
            return [{"id": "s1", "question": question, "sources": []}]
        return cleaned

    def _collect_round(self, sub_questions: list[dict], seen: set[str], *, question: str):
        grouped: dict[str, list[dict]] = {}
        for idx, sub_q in enumerate(sub_questions, start=1):
            sid = str(sub_q.get("id") or f"s{idx}")
            grouped[sid] = []
        routing = []

        for sub_q in sub_questions:
            sid = str(sub_q.get("id") or "s1")
            sub_question = str(sub_q.get("question") or question).strip()
            sources = list(sub_q.get("sources") or []) if self.use_routing else []
            if not sub_question:
                continue

            if self.retriever is None:
                grouped[sid] = []
                continue

            hits = self.retriever.hybrid(sub_question, 20, 20, 15, source_types=sources if self.use_routing and sources else None)
            if self.use_routing and sources and not hits:
                hits = self.retriever.hybrid(sub_question, 20, 20, 15, source_types=None)
                routing.append((sid, tuple(sources), "route_fallback"))
            else:
                routing.append((sid, tuple(sources), "route" if self.use_routing else "unrestricted"))

            for hit in hits:
                chunk_id = str(hit.chunk_id)
                if chunk_id in seen:
                    continue
                chunk = self.retriever.get_chunk(chunk_id)
                if not chunk:
                    continue
                seen.add(chunk_id)
                grouped.setdefault(sid, []).append({
                    "chunk_id": chunk_id,
                    "header": chunk.get("header", ""),
                    "text": chunk.get("text", ""),
                    "source_type": chunk.get("source_type", ""),
                    "score": float(getattr(hit, "score", 0.0) or 0.0),
                })

        return grouped, routing, seen

    def run(self, qid: str, question: str) -> SystemResult:
        sub_questions = self._plan_subquestions(question)
        state_notes: list[dict] = []
        state_conflicts: list[dict] = []
        seen: set[str] = set()

        for round_idx in range(3):
            grouped, routing, seen = self._collect_round(sub_questions, seen, question=question)
            if not any(grouped.values()):
                break

            new_passages = []
            for sid, items in grouped.items():
                if not items:
                    continue
                new_passages.append({"sub_question": sid, "passages": items})

            if self.llm is None:
                break

            try:
                payload = self.call_json(
                    "s6c_refine",
                    "You are the Refinement agent. Turn newly retrieved passages into notes and merge them into the shared knowledge state.",
                    "Question: " + question + "\nKnowledge notes so far (facts only):\n" + (json.dumps(state_notes) if state_notes else "(none)") + "\nNew passages grouped by sub-question:\n" + json.dumps(new_passages) + "\nReturn JSON with 'notes' and 'conflicts'.",
                    ["notes", "conflicts"],
                )
                refine = self._json(payload.get("text") if isinstance(payload, dict) else payload)
            except Exception:
                refine = {"notes": [], "conflicts": []}

            new_notes = refine.get("notes", []) if isinstance(refine, dict) else []
            if isinstance(new_notes, list):
                for note in new_notes:
                    if isinstance(note, dict):
                        state_notes.append(note)

            new_conflicts = refine.get("conflicts", []) if isinstance(refine, dict) else []
            if isinstance(new_conflicts, list):
                for conflict in new_conflicts:
                    if isinstance(conflict, dict):
                        state_conflicts.append(conflict)

            if round_idx >= 2:
                break

            try:
                payload = self.call_json(
                    "s6c_decide",
                    "You are the Decision agent. Decide whether the collected notes answer the question completely.",
                    "Question: " + question + "\nNotes:\n" + json.dumps(state_notes) + "\nReturn JSON with 'sufficient' boolean and 'gaps' list of {'question', 'sources'}.",
                    ["sufficient", "gaps"],
                )
                decision = self._json(payload.get("text") if isinstance(payload, dict) else payload)
            except Exception:
                decision = {"sufficient": True, "gaps": []}

            if bool(decision.get("sufficient", False)):
                break

            gaps = decision.get("gaps", []) if isinstance(decision, dict) else []
            if not isinstance(gaps, list) or not gaps:
                break

            sub_questions = []
            for gap in gaps[:2]:
                if not isinstance(gap, dict):
                    continue
                gap_question = str(gap.get("question") or "").strip()
                if not gap_question:
                    continue
                planned_sources = list(gap.get("sources") or []) if self.use_routing else []
                if self.use_routing:
                    planned_sources = [str(src).strip() for src in planned_sources if str(src).strip()][:2]
                sub_questions.append({"id": f"g{len(sub_questions)+1}", "question": gap_question, "sources": planned_sources})

            if not sub_questions:
                break

        evidence: list[EvidenceItem] = []
        ordered_ids: list[str] = []
        for note in state_notes:
            chunk_ids = note.get("chunk_ids") if isinstance(note, dict) else []
            if not isinstance(chunk_ids, list):
                continue
            for chunk_id in chunk_ids:
                chunk_id = str(chunk_id).strip()
                if not chunk_id or chunk_id in ordered_ids:
                    continue
                chunk = self.retriever.get_chunk(chunk_id) if self.retriever is not None else {}
                if not chunk:
                    continue
                ordered_ids.append(chunk_id)
                evidence.append(EvidenceItem(chunk_id=chunk_id, header=chunk.get("header", ""), text=chunk.get("text", ""), note=str(note.get("fact", "")).strip(), source_type=chunk.get("source_type", "")))

        if not evidence and self.retriever is not None:
            for chunk_id in sorted(seen):
                chunk = self.retriever.get_chunk(chunk_id)
                if not chunk:
                    continue
                evidence.append(EvidenceItem(chunk_id=chunk_id, header=chunk.get("header", ""), text=chunk.get("text", ""), source_type=chunk.get("source_type", "")))

        extra = ""
        if state_conflicts:
            extra = "Conflicts flagged during refinement: " + "; ".join(str(conflict.get("claim", "")) for conflict in state_conflicts if isinstance(conflict, dict) and str(conflict.get("claim", "")).strip())

        self.budget.spend(1)
        answer, shown = generate(self.llm, question, evidence, extra=extra, tag="s6c_final")
        abstained = answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_ids=[item.chunk_id for item in shown],
            candidate_ids=sorted(seen),
            final_evidence=shown,
            usage={"llm_calls": self.budget.used, "prompt_tokens": 0, "completion_tokens": 0, "local_s": 0.0, "wall_s": 0.0},
            trace=self.trace,
        )
