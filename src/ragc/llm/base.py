"""Abstract interface for all LLM clients.

This file defines the common contract used by the project for text generation. It keeps
LLM integrations behind a consistent API so prompt logic, caching, retry policies, and
testing can all share the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        raise NotImplementedError
