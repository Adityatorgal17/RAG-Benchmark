"""S6b: utilization-driven multi-agent review and reconciliation loop.

This matches the plan's evaluator -> note-taker -> generator -> reconcile pattern:
- score the top dense hits;
- keep the best chunk set, preserving a note for each kept passage;
- generate a draft answer with optional conflict feedback;
- iterate reconciliation up to 2 times until aligned or budget is exhausted.
"""

from __future__ import annotations

import json

from ragc.generation.generator import extract_citations
from ragc.schemas import EvidenceItem, SystemResult
from ragc.systems.base import System


class S6bUtilizationSystem(System):
    name = "S6b"

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

    def run(self, qid: str, question: str) -> SystemResult:
        retrieval_k = int(self.retrieval_param("s6b", "k", 10))
        min_keep = int(self.retrieval_param("s6b", "min_keep", 2))
        max_keep = int(self.retrieval_param("s6b", "max_keep", 6))
        score_threshold = int(self.retrieval_param("s6b", "score_threshold", 3))
        max_regen = int(self.retrieval_param("s6b", "max_regen", 2))

        if self.retriever is None:
            self.budget.spend_final(1)
            answer, shown = self.generate_answer(question, [], no_evidence_mode=True, tag="s6b_no_retrieval")
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

        hits = self.retriever.dense(question, retrieval_k)
        if not hits:
            self.budget.spend_final(1)
            answer, shown = self.generate_answer(question, [], no_evidence_mode=True, tag="s6b_no_hits")
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

        try:
            payload = self.call_json(
                "s6b_eval",
                "You are the Evaluator agent. Rate how useful each retrieved passage is for answering the question.",
                "Question: " + question + "\nPassages:\n" + "\n\n".join(f"{idx}: {self.retriever.get_chunk(hit.chunk_id).get('header', '')}\n{self.retriever.get_chunk(hit.chunk_id).get('text', '')}" for idx, hit in enumerate(hits, 1)) + "\nReturn JSON with a 'scores' list of {chunk_id, score, reason} objects.",
                ["scores"],
            )
            eval_data = self._json(payload.get("text") if isinstance(payload, dict) else payload)
        except Exception as exc:
            self.record_error("s6b_eval_error", exc)
            eval_data = {"scores": []}

        scores = eval_data.get("scores", []) if isinstance(eval_data, dict) else []
        kept: list[dict] = []
        score_map = {str(item.get("chunk_id", "")): int(item.get("score", 0)) for item in scores if isinstance(item, dict) and str(item.get("chunk_id", "")).strip()}
        for hit in hits:
            chunk_id = str(hit.chunk_id)
            score = score_map.get(chunk_id, 0)
            if score >= score_threshold:
                kept.append({"chunk_id": chunk_id, "score": score})
        if len(kept) < min_keep:
            kept = [{"chunk_id": str(hit.chunk_id), "score": float(getattr(hit, "score", 0) or 0)} for hit in hits[:min_keep]]
        kept = sorted(kept, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_keep]

        evidence: list[EvidenceItem] = []
        for item in kept:
            chunk_id = str(item["chunk_id"])
            chunk = self.retriever.get_chunk(chunk_id)
            if not chunk:
                continue
            evidence.append(EvidenceItem(chunk_id=chunk_id, header=chunk.get("header", ""), text=chunk.get("text", ""), source_type=chunk.get("source_type", "")))

        notes_payload = {"notes": [], "conflicts": [], "gaps": []}
        if self.llm is not None and evidence:
            try:
                payload = self.call_json(
                    "s6b_notes",
                    "You are the Note-taking agent. Summarize the useful evidence into facts, conflicts, and gaps.",
                    "Question: " + question + "\nEvidence:\n" + "\n\n".join(f"{item.chunk_id}: {item.header}\n{item.text}" for item in evidence) + "\nReturn JSON with notes, conflicts, and gaps.",
                    ["notes", "conflicts", "gaps"],
                )
                notes_payload = self._json(payload.get("text") if isinstance(payload, dict) else payload)
            except Exception as exc:
                self.record_error("s6b_notes_error", exc)
                notes_payload = {"notes": [], "conflicts": [], "gaps": []}

        notes = notes_payload.get("notes", []) if isinstance(notes_payload, dict) else []
        conflicts = notes_payload.get("conflicts", []) if isinstance(notes_payload, dict) else []

        for item in evidence:
            facts = []
            for note in notes:
                if isinstance(note, dict) and str(item.chunk_id) in [str(cid) for cid in (note.get("chunk_ids") or [])]:
                    facts.append(str(note.get("fact", "")).strip())
            item.note = " / ".join([fact for fact in facts if fact])

        extra = ""
        if conflicts:
            extra = "Conflicts flagged by the note-taker: " + "; ".join(str(c.get("claim", "")) for c in conflicts if isinstance(c, dict))

        self.budget.spend(1)
        answer, shown = self.generate_answer(question, evidence, extra=extra, tag="s6b_draft")

        aligned = True
        critiques = []
        for _ in range(max_regen):
            try:
                payload = self.call_json(
                    "s6b_recon",
                    "You are the Reconciliation agent. Decide whether the answer matches the evidence and complete coverage.",
                    "Question: " + question + "\nEvidence:\n" + "\n\n".join(f"{item.chunk_id}: {item.header}\n{item.text}" for item in evidence) + "\nDraft answer:\n" + answer + "\nReturn JSON with aligned boolean and critiques list.",
                    ["aligned", "critiques"],
                )
                recon = self._json(payload.get("text") if isinstance(payload, dict) else payload)
            except Exception as exc:
                self.record_error("s6b_recon_error", exc)
                recon = {"aligned": True, "critiques": []}

            aligned = bool(recon.get("aligned", True))
            critiques = recon.get("critiques", []) if isinstance(recon, dict) else []
            if aligned:
                break

            extra = extra + "\nCritiques of the previous draft: " + "; ".join(str(c) for c in critiques if c)
            if not self.budget.can_spend(1):
                self.trace.append({"tag": "s6b_regen_budget_exhausted", "critiques": critiques})
                break
            self.budget.spend(1)
            answer, shown = self.generate_answer(question, evidence, extra=extra, tag="s6b_regen")

        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer=answer,
            abstained=answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"),
            cited_ids=extract_citations(answer),
            candidate_ids=[item.chunk_id for item in evidence],
            final_evidence=shown,
            usage=self.usage(),
            trace=self.trace,
            errors=self.errors,
        )
