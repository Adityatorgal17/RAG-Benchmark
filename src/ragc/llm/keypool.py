"""Key rotation and provider fallback helper.

This module manages multiple API keys and retries calls across them when quota or rate
limits are hit. It keeps LLM access resilient without scattering provider logic across
other project modules.
"""

from __future__ import annotations


class QuotaExhausted(RuntimeError):
    pass


class KeyPool:
    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self.index = 0
        self.active_key = self.keys[0] if self.keys else None

    def _next_key(self):
        if not self.keys:
            raise QuotaExhausted("No keys configured")
        self.index = (self.index + 1) % len(self.keys)
        self.active_key = self.keys[self.index]
        return self.active_key

    def run_with_fallback(self, fn):
        if not self.keys:
            raise QuotaExhausted("No keys configured")

        for _ in range(len(self.keys)):
            key = self.keys[self.index]
            self.active_key = key
            try:
                return fn(key)
            except QuotaExhausted:
                self._next_key()
        raise QuotaExhausted("All configured keys are exhausted")
