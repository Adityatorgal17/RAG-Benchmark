"""Pointwise judge for benchmarking answer correctness against gold evidence.

This module follows the benchmark’s fixed prompt contract: it sends one gold-to-candidate
comparison per question and validates that the provider responded with the required JSON
shape. The implementation is deliberately lightweight and deterministic so it can be
checked with FakeLLM before real judge-provider calls are used.
"""

from __future__ import annotations

import json
import re
from typing import Any

P_JUDGE_POINT = """You are a strict grader of answers to cybersecurity questions. Judge ONLY against the gold answer and gold claims; do not use outside knowledge. Ignore citation markers. Output JSON only.

Question: {question}
Gold answer: {gold_answer}
Gold claims:
{claims_lines}
Candidate answer:
{answer}

Return {{"correctness": 0|1|2, "claims": {{"c1": true|false, ...}}, "abstained": true|false, "contradicts_gold": ["short phrase", ...]}}.
correctness: 2 = every key gold fact is present and correct and nothing contradicts the gold answer; 1 = partially correct or incomplete, no contradiction; 0 = wrong, contradicting, empty, or refused although the gold answer is not INSUFFICIENT_EVIDENCE.
If the gold answer is INSUFFICIENT_EVIDENCE: correctness = 2 only if the candidate states that the answer cannot be determined (or replies INSUFFICIENT_EVIDENCE) without asserting an answer; otherwise 0, and "claims" is {{}}.
"abstained" is true if the candidate declines to answer or says the information is not available.
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


def parse_judge_payload(raw: str | dict) -> dict:
    """Parse a raw LLM response into the strict judge payload shape.

    In real evaluation the judge must return the benchmark JSON contract. For local fake
    and pilot runs, a malformed response is treated as an empty/zero verdict instead of
    crashing the whole batch, which keeps the development loop resumable.
    """
    fallback = {"correctness": 0, "claims": {}, "abstained": False, "contradicts_gold": []}

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

    correctness = int(payload.get("correctness", 0))
    claims = payload.get("claims", {})
    if not isinstance(claims, dict):
        claims = {}
    abstained = bool(payload.get("abstained", False))
    contradicts = payload.get("contradicts_gold", [])
    if not isinstance(contradicts, list):
        contradicts = [str(contradicts)]

    return {
        "correctness": correctness,
        "claims": claims,
        "abstained": abstained,
        "contradicts_gold": contradicts,
    }


def judge_answer(llm, question: str, gold_answer: str, gold_claims: list[dict | str], candidate_answer: str) -> dict:
    """Call the pointwise judge and validate the JSON response."""
    claims_lines = _normalise_claim_lines(gold_claims)
    prompt = P_JUDGE_POINT.format(
        question=question,
        gold_answer=gold_answer,
        claims_lines=claims_lines,
        answer=candidate_answer,
    )
    if hasattr(llm, "calls"):
        llm.calls.append({"prompt": prompt, "json_mode": True, "system": "You are a strict grader of answers to cybersecurity questions."})
    payload = llm.generate(prompt, json_mode=True, system="You are a strict grader of answers to cybersecurity questions.")
    raw = getattr(payload, "text", payload)
    parsed = parse_judge_payload(raw)
    return parsed
