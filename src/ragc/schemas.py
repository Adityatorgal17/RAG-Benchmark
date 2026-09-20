"""Shared data models for chunks, retrieval hits, and system output.

These schemas define the contract between corpus processing, retrieval, generation,
and evaluation. Keeping the structure explicit makes it easier to trace data flow and
add new fields without breaking the rest of the codebase.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    section: str
    url: str
    date: str | None = None
    cve_ids: list[str] = Field(default_factory=list)
    header: str
    text: str
    n_words: int


class Hit(BaseModel):
    chunk_id: str
    score: float
    rank: int
    via: str


class EvidenceItem(BaseModel):
    chunk_id: str
    header: str
    text: str
    note: str | None = None
    source_type: str = ""


class LLMResult(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


class SystemResult(BaseModel):
    system: str
    qid: str
    question: str
    answer: str
    abstained: bool
    cited_ids: list[str]
    candidate_ids: list[str]
    final_evidence: list[EvidenceItem]
    trace: list[dict] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
