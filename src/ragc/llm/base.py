"""Abstract interface and common exceptions for all LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragc.schemas import LLMResult


class LLMError(RuntimeError):
    """Base class for provider, transport, and malformed-response failures."""


class BadJSON(LLMError):
    """Raised when a JSON-mode call fails validation after repair attempts."""


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None, max_tokens: int | None = None, tag: str = "") -> LLMResult:
        raise NotImplementedError

    def json(self, prompt: str, *, system: str | None = None, required: list[str] | None = None, max_tokens: int | None = None, tag: str = "") -> dict:
        """Generate and parse a JSON object, validating required keys."""
        import json
        import re

        required = required or []
        result = self.generate(prompt, json_mode=True, system=system, max_tokens=max_tokens, tag=tag)
        text = (result.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```\s*$", "", text)
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right > left:
            text = text[left : right + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BadJSON(f"Provider returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise BadJSON("Provider returned JSON, but not an object")
        missing = [key for key in required if key not in payload]
        if missing:
            raise BadJSON(f"Provider JSON missing required keys: {missing}")
        return payload
