"""Real Gemini and Groq LLM clients with cache, retries, and quota failover."""

from __future__ import annotations

import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from ragc.llm.base import BaseLLMClient, LLMError
from ragc.llm.cache import LLMCache
from ragc.llm.keypool import KeyPool, QuotaExhausted
from ragc.schemas import LLMResult


class SlidingWindowRateLimiter:
    def __init__(self, rpm: int = 60):
        self.rpm = max(1, int(rpm or 60))
        self.events: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self.events and now - self.events[0] >= 60:
            self.events.popleft()
        if len(self.events) >= self.rpm:
            delay = 60 - (now - self.events[0])
            if delay > 0:
                time.sleep(delay)
        self.events.append(time.monotonic())


def _looks_daily_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ["per day", "perday", "daily", "quota exceeded"])


def _looks_retryable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ["429", "rate", "timeout", "temporarily", "503", "502", "500"])


def _retry_delay(exc: BaseException, fallback: int) -> int:
    match = re.search(r"retry(?: after| in)?\s*(\d+)", str(exc), flags=re.IGNORECASE)
    if match:
        return max(1, int(match.group(1)))
    return fallback


class ProviderLLM(BaseLLMClient):
    provider = "provider"

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0,
        cache: LLMCache | None = None,
        key_pool: KeyPool | None = None,
        rpm: int = 60,
        default_max_tokens: int = 400,
    ):
        self.model = model
        self.temperature = temperature
        self.cache = cache or LLMCache()
        self.key_pool = key_pool or KeyPool.from_env(self.provider)
        self.default_max_tokens = default_max_tokens
        self._limiters: dict[str, SlidingWindowRateLimiter] = {}
        self.rpm = rpm

    def _limiter_for(self, key: str) -> SlidingWindowRateLimiter:
        if key not in self._limiters:
            self._limiters[key] = SlidingWindowRateLimiter(self.rpm)
        return self._limiters[key]

    def _call_provider(self, key: str, prompt: str, *, system: str | None, json_mode: bool, max_tokens: int) -> LLMResult:
        raise NotImplementedError

    def _call_with_retries(self, key: str, fn: Callable[[], LLMResult]) -> LLMResult:
        delays = [2, 4, 8, 16, 32]
        last_exc: BaseException | None = None
        for attempt, delay in enumerate(delays, start=1):
            try:
                self._limiter_for(key).wait()
                return fn()
            except Exception as exc:  # provider SDK exceptions differ
                if _looks_daily_quota_error(exc):
                    raise QuotaExhausted(str(exc)) from exc
                last_exc = exc
                if not _looks_retryable(exc) or attempt == len(delays):
                    break
                time.sleep(_retry_delay(exc, delay))
        raise LLMError(f"{self.provider} call failed after retries: {last_exc}") from last_exc

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None, max_tokens: int | None = None, tag: str = "") -> LLMResult:
        max_tokens = int(max_tokens or self.default_max_tokens)
        cached = self.cache.get(
            prompt,
            json_mode=json_mode,
            system=system,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        if cached is not None:
            return LLMResult(
                text=str(cached.get("text", "")),
                prompt_tokens=int(cached.get("prompt_tokens", 0) or 0),
                completion_tokens=int(cached.get("completion_tokens", 0) or 0),
                latency_s=float(cached.get("latency_s", 0.0) or 0.0),
                cached=True,
            )

        def invoke(key: str) -> LLMResult:
            return self._call_with_retries(key, lambda: self._call_provider(key, prompt, system=system, json_mode=json_mode, max_tokens=max_tokens))

        result = self.key_pool.run_with_fallback(invoke)
        self.cache.set(
            prompt,
            result.model_dump(),
            json_mode=json_mode,
            system=system,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return result


class GeminiClient(ProviderLLM):
    provider = "gemini"

    def _call_provider(self, key: str, prompt: str, *, system: str | None, json_mode: bool, max_tokens: int) -> LLMResult:
        start = time.perf_counter()
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise LLMError("google-genai is not installed") from exc

        client = genai.Client(api_key=key)
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": self.temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        config = types.GenerateContentConfig(**{k: v for k, v in config_kwargs.items() if v is not None})
        try:
            response = client.models.generate_content(model=self.model, contents=prompt, config=config)
        except Exception as exc:
            if "thinking" in str(exc).lower() and "thinking_config" in config_kwargs:
                config_kwargs.pop("thinking_config", None)
                config = types.GenerateContentConfig(**{k: v for k, v in config_kwargs.items() if v is not None})
                response = client.models.generate_content(model=self.model, contents=prompt, config=config)
            else:
                raise

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        text = getattr(response, "text", "") or ""
        return LLMResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, latency_s=time.perf_counter() - start, cached=False)


class GroqClient(ProviderLLM):
    provider = "groq"

    def _call_provider(self, key: str, prompt: str, *, system: str | None, json_mode: bool, max_tokens: int) -> LLMResult:
        start = time.perf_counter()
        try:
            from groq import Groq
        except Exception as exc:
            raise LLMError("groq is not installed") from exc

        client = Groq(api_key=key)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else ""
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_s=time.perf_counter() - start,
            cached=False,
        )


def _build_provider_llm(role_cfg, *, role_name: str, default_provider: str, default_model: str, default_rpm: int, default_max_tokens: int, cache: LLMCache):
    provider = str(getattr(role_cfg, "provider", default_provider)).strip().lower()
    model = str(getattr(role_cfg, "model", default_model))
    temperature = float(getattr(role_cfg, "temperature", 0))
    rpm = int(getattr(role_cfg, "rpm", default_rpm))
    max_tokens = int(getattr(role_cfg, "max_output_tokens", getattr(role_cfg, "max_output_tokens_answer", default_max_tokens)))

    if provider == "groq":
        pool = KeyPool.from_env("groq")
        if not pool.keys:
            raise QuotaExhausted(f"No Groq API key found for {role_name}. Set GROQ_API_KEY or GROQ_API_KEY1, or pass --fake for a dry run.")
        return GroqClient(
            model=model,
            temperature=temperature,
            cache=cache,
            key_pool=pool,
            rpm=rpm,
            default_max_tokens=max_tokens,
        )

    if provider == "gemini":
        pool = KeyPool.from_env("gemini")
        if not pool.keys:
            raise QuotaExhausted(f"No Gemini API key found for {role_name}. Set GEMINI_API_KEY or GEMINI_API_KEY1, or pass --fake for a dry run.")
        return GeminiClient(
            model=model,
            temperature=temperature,
            cache=cache,
            key_pool=pool,
            rpm=rpm,
            default_max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported {role_name} provider: {provider!r}")


def build_main_llm(cfg, *, fake: bool = False):
    if fake:
        from ragc.llm.fake import FakeLLM

        return FakeLLM(cache=LLMCache(getattr(getattr(cfg, "paths", None), "cache", "cache/llm_cache.sqlite")))
    main = getattr(cfg, "main_llm")
    cache = LLMCache(getattr(getattr(cfg, "paths", None), "cache", "cache/llm_cache.sqlite"))
    return _build_provider_llm(main, role_name="main_llm", default_provider="groq", default_model="openai/gpt-oss-20b", default_rpm=25, default_max_tokens=400, cache=cache)


def build_judge_llm(cfg, *, fake: bool = False):
    if fake:
        from ragc.llm.fake import FakeLLM

        return FakeLLM(cache=LLMCache(getattr(getattr(cfg, "paths", None), "cache", "cache/llm_cache.sqlite")))
    judge = getattr(cfg, "judge_llm")
    cache = LLMCache(getattr(getattr(cfg, "paths", None), "cache", "cache/llm_cache.sqlite"))
    return _build_provider_llm(judge, role_name="judge_llm", default_provider="gemini", default_model="gemini-3.5-flash-lite", default_rpm=10, default_max_tokens=400, cache=cache)
