"""Key discovery and sticky quota failover."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


class QuotaExhausted(RuntimeError):
    pass


class KeyPool:
    def __init__(self, keys: list[str], *, provider: str = "", state_path: str | Path = "cache/keypool_state.json"):
        self.keys = list(keys)
        self.provider = provider
        self.state_path = Path(state_path)
        self.index = 0
        self.active_key = self.keys[0] if self.keys else None
        self._state = self._load_state()
        self._skip_recently_exhausted()

    @classmethod
    def from_env(cls, provider: str, *, state_path: str | Path = "cache/keypool_state.json") -> "KeyPool":
        prefix = provider.upper()
        keys: list[str] = []
        single = os.getenv(f"{prefix}_API_KEY")
        if single:
            keys.append(single)
        idx = 1
        while True:
            value = os.getenv(f"{prefix}_API_KEY{idx}")
            if not value:
                break
            keys.append(value)
            idx += 1
        return cls(keys, provider=provider.lower(), state_path=state_path)

    def _load_state(self) -> dict:
        if not self.provider:
            return {}
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self) -> None:
        if not self.provider:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")

    def _state_key(self, key: str) -> str:
        return f"{self.provider}:{key[:8]}"

    def _is_recently_exhausted(self, key: str) -> bool:
        value = self._state.get(self._state_key(key))
        if not value:
            return False
        try:
            marked = datetime.fromisoformat(value)
        except ValueError:
            return False
        return datetime.now(timezone.utc) - marked < timedelta(hours=24)

    def _mark_exhausted(self, key: str) -> None:
        self._state[self._state_key(key)] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def _skip_recently_exhausted(self) -> None:
        for idx, key in enumerate(self.keys):
            if not self._is_recently_exhausted(key):
                self.index = idx
                self.active_key = key
                return

    def _next_key(self):
        if not self.keys:
            raise QuotaExhausted("No keys configured")
        for _ in range(len(self.keys)):
            self.index = (self.index + 1) % len(self.keys)
            self.active_key = self.keys[self.index]
            if not self._is_recently_exhausted(self.active_key):
                return self.active_key
        raise QuotaExhausted("All configured keys are exhausted")

    def run_with_fallback(self, fn):
        if not self.keys:
            raise QuotaExhausted("No keys configured")

        for _ in range(len(self.keys)):
            key = self.keys[self.index]
            self.active_key = key
            if self._is_recently_exhausted(key):
                self._next_key()
                continue
            try:
                return fn(key)
            except QuotaExhausted:
                self._mark_exhausted(key)
                self._next_key()
        raise QuotaExhausted("All configured keys are exhausted")
