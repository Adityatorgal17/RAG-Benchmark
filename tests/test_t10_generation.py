import re

from ragc.generation.evidence import build_evidence
from ragc.generation.generator import generate
from ragc.llm.fake import FakeLLM
from ragc.schemas import EvidenceItem


def test_build_evidence_truncates_and_handles_empty():
    items = [
        EvidenceItem(
            chunk_id="nvd_CVE-2024-3400#000",
            header="CVE-2024-3400 - Overview",
            text="PAN-OS is affected by a command injection issue in the web interface.",
            note="The issue is in the web interface.",
            source_type="nvd",
        ),
        EvidenceItem(
            chunk_id="vendor_panos#000",
            header="PAN-OS advisory - Summary",
            text="Affected versions include 10.2.9-h1 and customers should upgrade.",
            source_type="vendor",
        ),
    ]

    block, kept = build_evidence(items, max_tokens=40)
    assert kept and "[nvd_CVE-2024-3400#000]" in block
    assert "[vendor_panos#000]" not in block or len(kept) < 2

    empty_block, empty_kept = build_evidence([], max_tokens=2000)
    assert empty_block == "(none)"
    assert empty_kept == []


def test_generate_parses_citations_and_abstention():
    llm = FakeLLM()

    answer, used = generate(
        llm,
        "Which PAN-OS versions are affected?",
        [
            EvidenceItem(
                chunk_id="vendor_panos#000",
                header="PAN-OS advisory - Summary",
                text="Affected versions include 10.2.9-h1 and later.",
                source_type="vendor",
            )
        ],
        extra="",
    )
    assert answer
    assert any(match in answer for match in ["10.2.9-h1", "vendor_panos#000"])
    assert used[0].chunk_id == "vendor_panos#000"

    abstain_answer, _ = generate(
        llm,
        "What is the fix for this issue?",
        [],
        extra="",
        no_evidence_mode=False,
    )
    assert abstain_answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
