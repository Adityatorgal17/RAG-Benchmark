"""Registry utilities for locating system implementations by their benchmark ID."""

from __future__ import annotations

from typing import Callable

from ragc.systems.s_1_none import S1NoRetrievalSystem
from ragc.systems.s0_naive import S0NaiveSystem
from ragc.systems.s1_pre import S1PreSystem
from ragc.systems.s2_post import S2PostSystem, A2aSystem, A2bSystem
from ragc.systems.s3_decomp import S3DecompSystem
from ragc.systems.s4_llm_centric import S4LLMCentricSystem
from ragc.systems.s5_single_agent import S5SingleAgentSystem
from ragc.systems.s6a_acquisition import S6aAcquisitionSystem
from ragc.systems.s6b_utilization import S6bUtilizationSystem
from ragc.systems.s6c_hybrid import S6cHybridSystem

SYSTEM_IDS = [
    "S-1",
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6a",
    "S6b",
    "S6c",
    "A1",
    "A2a",
    "A2b",
]


def get_system_factory(system_id: str) -> Callable:
    """Return the constructor for a named system."""
    normalized = system_id.strip()
    if normalized not in SYSTEM_IDS:
        raise ValueError(f"Unsupported system id: {system_id!r}")

    factories = {
        "S-1": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S1NoRetrievalSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "S0": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S0NaiveSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "S1": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S1PreSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "S2": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S2PostSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "S3": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S3DecompSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "S4": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S4LLMCentricSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "S5": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S5SingleAgentSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "S6a": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S6aAcquisitionSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "S6b": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S6bUtilizationSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "S6c": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S6cHybridSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "A1": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: S6cHybridSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, use_routing=False, **kwargs),
        "A2a": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: A2aSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
        "A2b": lambda cfg=None, llm=None, retriever=None, reranker=None, **kwargs: A2bSystem(cfg=cfg, llm=llm, retriever=retriever, reranker=reranker, **kwargs),
    }

    return factories[normalized]


def list_system_ids() -> list[str]:
    return list(SYSTEM_IDS)
