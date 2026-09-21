"""Fake LLM used for unit tests and local dry runs.

This module provides a deterministic response stub that mimics the LLM interface without
making external API calls. It is the preferred way to validate logic before exercising
real providers.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from ragc.llm.cache import LLMCache


@dataclass
class FakeLLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


class FakeLLM:
    def __init__(self, cache: LLMCache | None = None):
        self.cache = cache
        self.calls: list[dict] = []

    def _default_text(self, prompt: str, *, json_mode: bool) -> str:
        if json_mode:
            lower = prompt.lower()
            if "rewrites" in lower and "hyde" in lower:
                return '{"rewrites": [], "hyde": ""}'
            if "sub_queries" in lower:
                return '{"sub_queries": []}'
            if "retrieve" in lower and "complexity" in lower:
                return '{"retrieve": true, "complexity": "simple"}'
            if "retrieve" in lower:
                return '{"retrieve": true}'
            if "scores" in lower:
                ids = re.findall(r"([\w.-]+#\d{3}|[\w.-]+-\d{3}|demo-\d{3})", prompt)
                return '{"scores": [' + ", ".join(f'{{"chunk_id": "{cid}", "score": 3, "reason": "related"}}' for cid in dict.fromkeys(ids)) + ']}'
            if "aligned" in lower:
                return '{"aligned": true, "critiques": []}'
            if "notes" in lower and "conflicts" in lower:
                return '{"notes": [], "conflicts": [], "gaps": []}'
            if "notes" in lower:
                return '{"notes": []}'
            if "sufficient" in lower:
                return '{"sufficient": true, "missing": "", "gaps": []}'
            if "ranking" in lower:
                return '{"ranking": []}'
            if "correctness" in lower:
                return '{"correctness": 0, "claims": {}, "abstained": false, "contradicts_gold": []}'
            return "{}"

        citation = re.search(r"\[([^\[\]\s]+#\d{3})\]", prompt)
        if citation:
            chunk_id = citation.group(1)
            evidence_tail = prompt[prompt.find(f"[{chunk_id}]") :]
            version = re.search(r"\b\d+\.\d+(?:\.\d+)?(?:-[a-z0-9]+)?\b", evidence_tail, flags=re.IGNORECASE)
            if version:
                return f"Affected versions include {version.group(0)}. [{chunk_id}]"
            return f"Based on the evidence, the answer is supported. [{chunk_id}]"
        if "No documents are available" in prompt:
            return "INSUFFICIENT_EVIDENCE"
        return "OK"

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None, max_tokens: int | None = None, tag: str = ""):
        payload = {"prompt": prompt, "json_mode": json_mode, "system": system}
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "system": system})
        cached = None
        if self.cache is not None:
            cached = self.cache.get(prompt, json_mode=json_mode, system=system)
        if cached is not None:
            return FakeLLMResult(
                text=cached.get("text", "OK"),
                prompt_tokens=int(cached.get("prompt_tokens", 8) or 8),
                completion_tokens=int(cached.get("completion_tokens", 8) or 8),
                latency_s=float(cached.get("latency_s", 0.1) or 0.1),
                cached=True,
            )

        result = FakeLLMResult(text=self._default_text(prompt, json_mode=json_mode), prompt_tokens=max(1, len(prompt.split())), completion_tokens=8, latency_s=0.1, cached=False)
        if self.cache is not None:
            self.cache.set(
                prompt,
                {
                    "text": result.text,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_s": result.latency_s,
                },
                json_mode=json_mode,
                system=system,
            )
        return result
