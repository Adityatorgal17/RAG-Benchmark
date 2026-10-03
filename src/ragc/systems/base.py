"""Base execution layer for all RAG system variants.

This module defines the common contract shared by every system implementation in the
benchmark: a per-run call budget, a standard result object, and a helper for safe
LLM calls that enforce the project-wide resource limits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from types import SimpleNamespace
from time import perf_counter

from ragc.config import load_config
from ragc.generation.generator import generate
from ragc.llm.base import BaseLLMClient
from ragc.schemas import EvidenceItem, SystemResult


class BudgetExceeded(RuntimeError):
    """Raised when a system attempts to exceed the reserved LLM call budget."""


class CallBudget:
    """Tracks LLM calls across a single system run while reserving the final turn."""

    def __init__(self, max_calls: int, reserve_final: int = 1):
        self.max_calls = max_calls
        self.reserve_final = reserve_final
        self.used = 0

    def can_spend(self, n: int = 1) -> bool:
        return (self.used + n) <= (self.max_calls - self.reserve_final)

    def spend(self, n: int = 1):
        if not self.can_spend(n):
            raise BudgetExceeded(f"Budget would exceed max_calls={self.max_calls} with reserve_final={self.reserve_final}")
        self.used += n
        return self.used

    def spend_final(self, n: int = 1):
        if (self.used + n) > self.max_calls:
            raise BudgetExceeded(f"Final call would exceed max_calls={self.max_calls}")
        self.used += n
        return self.used


@lru_cache(maxsize=1)
def _project_config():
    return load_config()


class System(ABC):
    """Common base class for all system-level agent implementations."""

    name: str = "system"

    def __init__(self, cfg=None, llm: BaseLLMClient | None = None, retriever=None, reranker=None, *, max_calls: int | None = None, reserve_final: int = 1):
        raw_cfg = cfg or SimpleNamespace(budget=SimpleNamespace(max_llm_calls=8, reserve_final=1))
        budget_cfg = getattr(raw_cfg, "budget", SimpleNamespace(max_llm_calls=8, reserve_final=1))
        if max_calls is None:
            max_calls = int(getattr(budget_cfg, "max_llm_calls", 8))
        self.cfg = raw_cfg
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.budget = CallBudget(max_calls=max_calls, reserve_final=reserve_final)
        self.trace: list[dict] = []
        self.errors: list[str] = []
        self._started_at = perf_counter()
        self._llm_results: list[object] = []
        if self.llm is not None:
            try:
                setattr(self.llm, "_ragc_result_sink", self._record_llm_result)
            except Exception:
                pass

    def cfg_get(self, *path: str, default=None):
        """Read a nested config value from namespaces or dicts."""
        value = self._cfg_get_from(self.cfg, *path)
        if value is not None:
            return value
        value = self._cfg_get_from(_project_config(), *path)
        return default if value is None else value

    @staticmethod
    def _cfg_get_from(source, *path: str):
        missing = object()
        current = source
        for key in path:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key, missing)
            else:
                current = getattr(current, key, missing)
            if current is missing:
                return None
        return current

    def retrieval_param(self, section: str, key: str, default):
        return self.cfg_get("retrieval", section, key, default=default)

    def post_param(self, key: str, default):
        return self.cfg_get("post", key, default=default)

    def context_max_tokens(self) -> int:
        return int(self.cfg_get("context", "max_tokens"))

    def generate_answer(self, question: str, items: list[EvidenceItem], *, extra: str = "", no_evidence_mode: bool = False, tag: str = "gen"):
        return generate(
            self.llm,
            question,
            items,
            extra=extra,
            no_evidence_mode=no_evidence_mode,
            tag=tag,
            max_context_tokens=self.context_max_tokens(),
        )

    def record_error(self, tag: str, exc: BaseException | str) -> None:
        message = str(exc)
        self.errors.append(f"{tag}: {message}")
        self.trace.append({"tag": tag, "error": message})

    def _record_llm_result(self, result) -> None:
        self._llm_results.append(result)

    def usage(self) -> dict:
        prompt_tokens = sum(int(getattr(item, "prompt_tokens", 0) or 0) for item in self._llm_results)
        completion_tokens = sum(int(getattr(item, "completion_tokens", 0) or 0) for item in self._llm_results)
        active_latency_s = sum(float(getattr(item, "latency_s", 0.0) or 0.0) for item in self._llm_results)
        wall_s = max(0.0, perf_counter() - self._started_at)
        local_s = max(0.0, wall_s - active_latency_s)
        return {
            "llm_calls": self.budget.used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "active_latency_s": round(active_latency_s, 4),
            "latency_s": round(active_latency_s, 4),
            "local_s": round(local_s, 4),
            "wall_s": round(wall_s, 4),
        }

    def call_json(self, tag: str, system: str, user: str, required: list[str], budget: CallBudget | None = None) -> dict:
        """Execute a JSON-producing LLM call while enforcing the call budget.

        A caller may use this helper to perform an agent step; if the budget is exhausted,
        the method raises a `BudgetExceeded` so the surrounding loop can end cleanly and
        continue to the final generation step.
        """
        import json as _json, re as _re
        active_budget = budget or self.budget
        if not active_budget.can_spend(1):
            raise BudgetExceeded("LLM budget exhausted before agent step")

        start = perf_counter()
        active_budget.spend(1)
        if self.llm is None:
            raise RuntimeError("No LLM client configured for this system")

        try:
            payload = self.llm.generate(user, json_mode=True, system=system)
            self._record_llm_result(payload)
            result = getattr(payload, "text", payload) or ""
        except Exception as _exc:
            # json_validate_failed / tool_use_failed: model couldn't produce JSON
            # in constrained mode. Retry once in plain text mode and extract JSON.
            _exc_str = str(_exc)
            if "json_validate_failed" in _exc_str or "tool_use_failed" in _exc_str or not _exc_str:
                result = ""
            else:
                raise  # re-raise unrelated errors (quota, network, etc.)

        elapsed = perf_counter() - start

        # If model returned empty (or failed JSON mode), retry in plain text mode.
        if not str(result).strip():
            try:
                payload2 = self.llm.generate(
                    user + "\n\nIMPORTANT: Reply with ONLY a raw JSON object. No markdown, no explanation.",
                    json_mode=False,
                    system=system + " Return only a raw JSON object, nothing else.",
                )
                self._record_llm_result(payload2)
                raw = getattr(payload2, "text", payload2) or ""
                # Extract first JSON object from plain text response
                match = _re.search(r'\{.*\}', raw, _re.DOTALL)
                result = match.group(0) if match else raw
            except Exception:
                result = ""  # Callers handle empty/invalid JSON with their own fallbacks

        self.trace.append({"tag": tag, "system": system, "user": user, "latency_s": elapsed, "required": required, "result": result})
        return {"text": result}

    @abstractmethod
    def run(self, qid: str, question: str) -> SystemResult:
        raise NotImplementedError
