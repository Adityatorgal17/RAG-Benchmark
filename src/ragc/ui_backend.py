"""Small backend adapter for the single-page RAG explorer UI.

This module exposes a single function used by the UI server to run any of the
project's benchmark systems through the existing system registry and return a
JSON-serializable payload that can be displayed in a lightweight dashboard.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ragc.config import load_config
from ragc.llm.clients import build_main_llm
from ragc.retrieval.reranker import Reranker
from ragc.retrieval.retriever import Retriever
from ragc.systems.registry import get_system_factory


_RERANKER_SYSTEMS = {"S2", "S3", "S6c", "A1", "A2a", "A2b"}
_UI_LLM = None
_UI_LLM_MODE = "unknown"
_UI_RETRIEVER: Retriever | None = None
_UI_RERANKER: Reranker | None = None


def _has_provider_key(provider: str) -> bool:
    prefix = provider.upper()
    return any(
        os.getenv(name)
        for name in [f"{prefix}_API_KEY", *[f"{prefix}_API_KEY{i}" for i in range(1, 10)]]
    )


def _get_retriever(cfg):
    global _UI_RETRIEVER
    if _UI_RETRIEVER is None:
        _UI_RETRIEVER = Retriever(cfg)
    return _UI_RETRIEVER


def _get_llm(cfg):
    global _UI_LLM, _UI_LLM_MODE
    if _UI_LLM is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        force_fake = os.getenv("RAGC_UI_FAKE_LLM", "").strip().lower() in {"1", "true", "yes"}
        force_real = os.getenv("RAGC_UI_REAL_LLM", "").strip().lower() in {"1", "true", "yes"}
        main_provider = str(getattr(getattr(cfg, "main_llm", None), "provider", "groq")).strip().lower()
        use_real = force_real or (_has_provider_key(main_provider) and not force_fake)
        _UI_LLM = build_main_llm(cfg, fake=not use_real)
        _UI_LLM_MODE = "real" if use_real else "fake"
    return _UI_LLM


def get_ui_runtime_status() -> dict[str, Any]:
    """Return runtime details shown by the UI without exposing secrets."""
    cfg = load_config()
    llm = _get_llm(cfg)
    return {
        "backend": "real",
        "retrieval": "real",
        "llm_mode": _UI_LLM_MODE,
        "provider": getattr(llm, "provider", "fake"),
        "model": getattr(llm, "model", "FakeLLM"),
        "chunks_path": str(getattr(cfg.paths, "chunks", "")),
        "chroma_path": str(getattr(cfg.paths, "chroma", "")),
        "bm25_path": str(getattr(cfg.paths, "bm25", "")),
    }


def _get_reranker(cfg, system_id: str):
    """Load the configured transformer reranker for UI runs that require it."""
    global _UI_RERANKER
    if system_id not in _RERANKER_SYSTEMS:
        return None
    if _UI_RERANKER is None:
        rerank_cfg = getattr(cfg, "reranker", None)
        _UI_RERANKER = Reranker(
            getattr(rerank_cfg, "model", "BAAI/bge-reranker-base"),
            batch_size=int(getattr(rerank_cfg, "batch_size", 16)),
            max_length=int(getattr(rerank_cfg, "max_length", 384)),
        )
    return _UI_RERANKER


def _summarize_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for item in trace:
        result_text = item.get("result")
        if isinstance(result_text, (dict, list)):
            summary = json.dumps(result_text, ensure_ascii=False)[:500]
        else:
            summary = str(result_text or "")[:500]
        steps.append(
            {
                "tag": item.get("tag", "step"),
                "system": item.get("system", "system"),
                "summary": summary,
            }
        )
    return steps


def _parse_trace_payloads(trace: list[dict[str, Any]]) -> dict[str, Any]:
    pre_retrieval: list[str] = []
    post_retrieval: list[str] = []
    thought: list[str] = []

    for item in trace:
        tag = str(item.get("tag", "")).lower()
        result = item.get("result")
        if isinstance(result, str):
            content = result
        elif isinstance(result, (dict, list)):
            content = json.dumps(result, ensure_ascii=False)
        else:
            content = str(result or "")

        if tag in {"s1_rewrite", "s4_gate", "s5_agent", "s6a_decision", "s6b_eval", "s6c_plan", "s6c_decide"}:
            pre_retrieval.append(content)
        if tag in {"s2", "s4_notes", "s4_final", "s5_final", "s6a_final", "s6b_recon", "s6c_final"}:
            post_retrieval.append(content)
        if any(k in tag for k in ("thought", "plan", "decision", "judge", "notes", "agent")):
            thought.append(content)

    return {
        "pre_retrieval": pre_retrieval,
        "post_retrieval": post_retrieval,
        "thought": thought,
    }


def build_ui_response(system_id: str, question: str) -> dict[str, Any]:
    """Run the selected system and return a UI-friendly payload."""
    cfg = load_config()
    factory = get_system_factory(system_id)
    system = factory(cfg=cfg, llm=_get_llm(cfg), retriever=_get_retriever(cfg), reranker=_get_reranker(cfg, system_id))
    result = system.run("ui-demo", question)

    side_summary = _parse_trace_payloads(result.trace)
    
    # Extract LLM calls count from usage (correctly tracks budget.used)
    llm_calls = int(result.usage.get("llm_calls", 0)) if result.usage else 0
    
    ui_payload = {
        "system": result.system,
        "question": result.question,
        "answer": result.answer,
        "runtime": get_ui_runtime_status(),
        "abstained": result.abstained,
        "cited_ids": result.cited_ids,
        "candidates": result.candidate_ids,  # For UI visualization
        "trace": result.trace,
        "steps": _summarize_trace(result.trace),
        "model_thinking_state": side_summary["thought"],
        "chain_of_thought": side_summary["thought"],
        "pre_retrieval": side_summary["pre_retrieval"],
        "post_retrieval": side_summary["post_retrieval"],
        "evidence": [
            {
                "chunk_id": item.chunk_id,
                "header": item.header,
                "text": item.text,
                "note": item.note,
                "source_type": item.source_type,
            }
            for item in result.final_evidence
        ],
        "usage": result.usage,
        "errors": result.errors,
        "llm_calls": llm_calls,
    }
    return ui_payload
