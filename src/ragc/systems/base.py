"""Base execution layer for all RAG system variants.

This module defines the common contract shared by every system implementation in the
benchmark: a per-run call budget, a standard result object, and a helper for safe
LLM calls that enforce the project-wide resource limits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from time import perf_counter

from ragc.llm.base import BaseLLMClient
from ragc.schemas import SystemResult


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
        self._started_at = perf_counter()
        self._llm_results: list[object] = []
        if self.llm is not None:
            try:
                setattr(self.llm, "_ragc_result_sink", self._record_llm_result)
            except Exception:
                pass

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
        active_budget = budget or self.budget
        if not active_budget.can_spend(1):
            raise BudgetExceeded("LLM budget exhausted before agent step")

        start = perf_counter()
        active_budget.spend(1)
        if self.llm is None:
            raise RuntimeError("No LLM client configured for this system")

        payload = self.llm.generate(user, json_mode=True, system=system)
        self._record_llm_result(payload)
        elapsed = perf_counter() - start
        result = getattr(payload, "text", payload)
        self.trace.append({"tag": tag, "system": system, "user": user, "latency_s": elapsed, "required": required, "result": result})
        return {"text": result}

    @abstractmethod
    def run(self, qid: str, question: str) -> SystemResult:
        raise NotImplementedError
