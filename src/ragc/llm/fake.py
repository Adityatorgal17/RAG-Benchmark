"""Fake LLM used for unit tests and local dry runs.

This module provides a deterministic response stub that mimics the LLM interface without
making external API calls. It is the preferred way to validate logic before exercising
real providers.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        payload = {"prompt": prompt, "json_mode": json_mode, "system": system}
        self.calls.append({"prompt": prompt, "json_mode": json_mode, "system": system})
        cached = None
        if self.cache is not None:
            cached = self.cache.get(prompt, json_mode=json_mode, system=system)
        if cached is not None:
            return FakeLLMResult(text=cached.get("text", "OK"), prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=True)

        result = FakeLLMResult(text="OK", prompt_tokens=8, completion_tokens=8, latency_s=0.1, cached=False)
        if self.cache is not None:
            self.cache.set(prompt, {"text": result.text}, json_mode=json_mode, system=system)
        return result
