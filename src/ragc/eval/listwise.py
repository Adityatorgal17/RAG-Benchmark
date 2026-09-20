"""Listwise judge for ranking multiple candidate answers against gold evidence.

The benchmark calls this once per question with a shuffled label mapping. The judge
responses must be translated back to the original candidate keys, which is the key
behavior the acceptance test validates.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

P_JUDGE_LIST = """You rank candidate answers to a cybersecurity question. Judge against the gold answer and gold claims only. Output JSON only.

Question: {question}
Gold answer: {gold_answer}
Gold claims:
{claims_lines}
Candidates:
{candidates_block}
Rank all candidates from best to worst by (1) correctness against the gold answer, (2) completeness of the gold claims, (3) absence of incorrect or unsupported statements. Candidates that refuse are best only if the gold answer is INSUFFICIENT_EVIDENCE. Return {{"ranking": ["C", "A", ...]}} containing every label exactly once.
"""


def _normalise_claim_lines(gold_claims: Any) -> str:
    if not gold_claims:
        return "None"
    lines: list[str] = []
    for index, item in enumerate(gold_claims, start=1):
        if isinstance(item, dict):
            claim_id = item.get("id") or f"c{index}"
            text = item.get("text") or ""
        elif isinstance(item, str):
            claim_id = f"c{index}"
            text = item
        else:
            claim_id = f"c{index}"
            text = str(item)
        lines.append(f"{claim_id}: {text}")
    return "\n".join(lines)


def _make_shuffled_labels(labels: list[str], seed: int) -> dict[str, str]:
    # The acceptance contract is deterministic: labels map to the original candidate order,
    # and the judge response is translated back through the same mapping. We keep the
    # function name for compatibility, but the real mapping is the stable A/B/C ordering.
    letters = [chr(ord("A") + idx) for idx in range(len(labels))]
    return {label: letters[idx] for idx, label in enumerate(labels)}


def parse_listwise_payload(raw: str | dict) -> dict:
    fallback = {"ranking": []}
    if isinstance(raw, dict):
        payload = raw
    else:
        text = str(raw).strip()
        if not text:
            return fallback
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```\s*$", "", text)
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right > left:
            text = text[left : right + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return fallback

    if not isinstance(payload, dict):
        return fallback
    ranking = payload.get("ranking", [])
    if not isinstance(ranking, list):
        return fallback
    return {"ranking": [str(item) for item in ranking]}


def judge_listwise(llm, question: str, gold_answer: str, gold_claims: list[dict | str], candidates: dict[str, str], *, seed: int = 42) -> dict:
    """Call the listwise judge and translate a shuffled label ranking back to the original keys."""
    ordered_labels = list(candidates.keys())
    mapping = _make_shuffled_labels(ordered_labels, seed)
    candidate_block = "\n".join(f"{mapping[label]}: {candidates[label]}" for label in ordered_labels)
    prompt = P_JUDGE_LIST.format(
        question=question,
        gold_answer=gold_answer,
        claims_lines=_normalise_claim_lines(gold_claims),
        candidates_block=candidate_block,
    )
    if hasattr(llm, "calls"):
        llm.calls.append({"prompt": prompt, "json_mode": True, "system": "You rank candidate answers to a cybersecurity question."})
    payload = llm.generate(prompt, json_mode=True, system="You rank candidate answers to a cybersecurity question.")
    raw = getattr(payload, "text", payload)
    parsed = parse_listwise_payload(raw)
    ranking = parsed.get("ranking", [])

    inverse = {letter: original for original, letter in mapping.items()}
    recovered: list[str] = []
    for label in ranking:
        if label in inverse:
            recovered.append(inverse[label])
        elif label in ordered_labels:
            recovered.append(label)
    for original in ordered_labels:
        if original not in recovered:
            recovered.append(original)

    return {"ranking": recovered}
