"""SQLite cache for LLM responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class CacheEntry(Base):
    __tablename__ = "llm_cache"
    key = Column(String, primary_key=True)
    value = Column(String)


class LLMCache:
    def __init__(self, path: str | Path = "cache/llm_cache.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.path}")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)

    @staticmethod
    def _hash(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def key_for(self, prompt: str, *, json_mode: bool = False, system: str | None = None, provider: str = "", model: str = "", temperature: float | int | None = None, max_tokens: int | None = None) -> str:
        return self._hash(
            {
                "prompt": prompt,
                "json_mode": json_mode,
                "system": system,
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

    def get(self, prompt: str, *, json_mode: bool = False, system: str | None = None, provider: str = "", model: str = "", temperature: float | int | None = None, max_tokens: int | None = None) -> dict | None:
        key = self.key_for(prompt, json_mode=json_mode, system=system, provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)
        with self.session_factory() as session:
            row = session.get(CacheEntry, key)
            return json.loads(row.value) if row is not None else None

    def set(self, prompt: str, value: dict, *, json_mode: bool = False, system: str | None = None, provider: str = "", model: str = "", temperature: float | int | None = None, max_tokens: int | None = None):
        key = self.key_for(prompt, json_mode=json_mode, system=system, provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)
        with self.session_factory() as session:
            session.merge(CacheEntry(key=key, value=json.dumps(value, sort_keys=True)))
            session.commit()
