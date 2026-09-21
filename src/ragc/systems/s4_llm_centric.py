"""S4: LLM-centric retrieval decision and Chain-of-Note refinement.

This follows the plan's prompt-driven control flow: gate decision, relevance audit,
then a draft answer, critique, and an optional repair pass that appends new evidence.
"""

from __future__ import annotations

import json

from ragc.generation.generator import extract_citations, generate
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System


class S4LLMCentricSystem(System):
    name = "S4"

    def run(self, qid: str, question: str) -> SystemResult:
        gate = {"retrieve": True}
        if self.llm is not None:
            try:
                payload = self.call_json(
                    "s4_gate",
                    "You decide whether answering a question requires searching a fixed collection of documents.",
                    f"Question: {question}\nReturn JSON with retrieve boolean.",
                    ["retrieve"],
                )
                text = payload.get("text") or "{}"
                if isinstance(text, str):
                    gate = json.loads(text)
            except Exception:
                gate = {"retrieve": True}

        if not bool(gate.get("retrieve", True)):
            self.budget.spend(1)
            answer, shown = generate(self.llm, question, [], no_evidence_mode=True, tag="s4_gate_false")
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
            )

        hits = self.retriever.dense(question, 8) if self.retriever is not None else []
        candidate_ids = [hit.chunk_id for hit in hits]

        notes_payload = {"notes": []}
        if self.llm is not None and hits:
            passages = []
            for hit in hits:
                chunk = self.retriever.get_chunk(hit.chunk_id) if self.retriever is not None else {}
                if not chunk:
                    continue
                passages.append({"chunk_id": hit.chunk_id, "header": chunk.get("header", ""), "text": chunk.get("text", "")})
            try:
                payload = self.call_json(
                    "s4_notes",
                    "You read retrieved passages critically before answering.",
                    "Question: " + question + "\n\nPassages:\n" + "\n\n".join(f"{p['chunk_id']}: {p['header']}\n{p['text']}" for p in passages) + "\n\nReturn JSON with notes for each passage.",
                    ["notes"],
                )
                text = payload.get("text") or "{}"
                if isinstance(text, str):
                    notes_payload = json.loads(text)
            except Exception:
                notes_payload = {"notes": []}

        note_map = {
            str(entry.get("chunk_id")): entry
            for entry in notes_payload.get("notes", [])
            if isinstance(entry, dict) and str(entry.get("chunk_id", "")).strip()
        }

        notes_valid = isinstance(notes_payload, dict) and isinstance(notes_payload.get("notes", []), list)

        relevant_ids: list[str] = []
        if hits:
            for hit in hits:
                item = note_map.get(hit.chunk_id)
                if item and str(item.get("relevance", "irrelevant")).lower() in {"relevant", "partial"}:
                    relevant_ids.append(hit.chunk_id)
        if not relevant_ids and hits and not notes_valid:
            relevant_ids = [hit.chunk_id for hit in hits]

        evidence: list[EvidenceItem] = []
        for chunk_id in relevant_ids:
            chunk = self.retriever.get_chunk(chunk_id) if self.retriever is not None else {}
            if not chunk:
                continue
            item = note_map.get(chunk_id, {})
            evidence.append(
                EvidenceItem(
                    chunk_id=chunk_id,
                    header=chunk.get("header", ""),
                    text=chunk.get("text", ""),
                    note=str(item.get("note", "")) if isinstance(item, dict) else "",
                    source_type=chunk.get("source_type", ""),
                )
            )

        if not evidence:
            self.budget.spend(1)
            answer, shown = generate(self.llm, question, evidence, tag="s4_final")
            return SystemResult(
                system=self.name,
                qid=qid,
                question=question,
                answer=answer,
                abstained=answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"),
                cited_ids=extract_citations(answer),
                candidate_ids=candidate_ids,
                final_evidence=shown,
                usage=self.usage(),
                trace=self.trace,
            )

        self.budget.spend(1)
        answer1, _ = generate(self.llm, question, evidence, tag="s4_draft")

        critique = {"supported": "full", "useful": 5, "unsupported_claims": [], "missing_parts": []}
        if self.llm is not None and not answer1.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
            try:
                payload = self.call_json(
                    "s4_critique",
                    "You audit a draft answer against its evidence.",
                    "Question: " + question + "\n\nEvidence:\n" + "\n\n".join(f"{item.chunk_id}: {item.header}\n{item.text}" for item in evidence) + "\n\nDraft answer:\n" + answer1 + "\n\nReturn JSON critique.",
                    ["supported", "useful", "unsupported_claims", "missing_parts"],
                )
                text = payload.get("text") or "{}"
                if isinstance(text, str):
                    critique = json.loads(text)
            except Exception:
                critique = {"supported": "full", "useful": 5, "unsupported_claims": [], "missing_parts": []}

        extra = ""
        if not answer1.strip().upper().startswith("INSUFFICIENT_EVIDENCE") and critique.get("supported") != "full":
            missing = critique.get("missing_parts") or critique.get("unsupported_claims") or [question]
            query = str(missing[0]).strip() if isinstance(missing, list) and missing else question
            if query:
                new_hits = self.retriever.dense(query, 3) if self.retriever is not None else []
                for hit in new_hits:
                    if hit.chunk_id not in [item.chunk_id for item in evidence]:
                        chunk = self.retriever.get_chunk(hit.chunk_id) if self.retriever is not None else {}
                        if chunk:
                            evidence.append(EvidenceItem(chunk_id=hit.chunk_id, header=chunk.get("header", ""), text=chunk.get("text", ""), source_type=chunk.get("source_type", "")))
                extra = (
                    "Reviewer feedback on your previous draft. Unsupported claims: "
                    + str(critique.get("unsupported_claims", []))
                    + ". Missing parts: "
                    + str(critique.get("missing_parts", []))
                    + ". Rewrite the answer using ONLY the evidence above: remove unsupported claims and cover the missing parts if the evidence allows."
                )

        self.budget.spend(1)
        final_answer, shown = generate(self.llm, question, evidence, extra=extra, tag="s4_final")
        abstained = final_answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=final_answer,
            abstained=abstained,
            cited_ids=extract_citations(final_answer),
            candidate_ids=candidate_ids,
            final_evidence=shown,
            usage=self.usage(),
            trace=self.trace,
        )
