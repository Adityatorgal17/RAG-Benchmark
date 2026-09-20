"""Small backend adapter for the single-page RAG explorer UI.

This module exposes a single function used by the UI server to run any of the
project's benchmark systems through the existing system registry and return a
JSON-serializable payload that can be displayed in a lightweight dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragc.config import load_config
from ragc.llm.fake import FakeLLM
from ragc.retrieval.retriever import Retriever
from ragc.systems.registry import get_system_factory


def _demo_chunks() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "demo-001",
            "doc_id": "doc-1",
            "source_type": "attack",
            "title": "CVE advisory",
            "section": "Overview",
            "url": "https://example.org/demo-1",
            "date": "2024-01-01",
            "cve_ids": ["CVE-2024-0001"],
            "header": "Remote code execution in edge appliance",
            "text": "Attackers can exploit an unauthenticated remote code execution flaw in the edge appliance when inbound traffic reaches the vulnerable service.",
            "n_words": 32,
        },
        {
            "chunk_id": "demo-002",
            "doc_id": "doc-2",
            "source_type": "mitigation",
            "title": "Mitigation guidance",
            "section": "Remediation",
            "url": "https://example.org/demo-2",
            "date": "2024-01-02",
            "cve_ids": [],
            "header": "Apply vendor patch and restrict exposure",
            "text": "The recommended mitigation is to apply the vendor patch immediately and restrict the service to trusted networks or internal management interfaces.",
            "n_words": 30,
        },
        {
            "chunk_id": "demo-003",
            "doc_id": "doc-3",
            "source_type": "news",
            "title": "Threat context",
            "section": "Background",
            "url": "https://example.org/demo-3",
            "date": "2024-01-03",
            "cve_ids": ["CVE-2024-0002"],
            "header": "Exploit trend in public-facing services",
            "text": "Public-facing telephony and edge services have become a common target for exploitation chains that combine reconnaissance with weak credential hygiene.",
            "n_words": 27,
        },
    ]


def _get_retriever(cfg):
    chunks_path = Path(getattr(getattr(cfg, "paths", None), "chunks", "data/processed/chunks.jsonl"))
    if chunks_path.exists():
        return Retriever(cfg, chunks_path=str(chunks_path))
    return Retriever(cfg, chunks=_demo_chunks())


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
    system = factory(cfg=cfg, llm=FakeLLM(), retriever=_get_retriever(cfg), reranker=None)
    result = system.run("ui-demo", question)

    side_summary = _parse_trace_payloads(result.trace)
    ui_payload = {
        "system": result.system,
        "question": result.question,
        "answer": result.answer,
        "abstained": result.abstained,
        "cited_ids": result.cited_ids,
        "candidate_ids": result.candidate_ids,
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
    }
    return ui_payload
