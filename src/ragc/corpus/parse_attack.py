"""ATT&CK technique parser for threat-pattern sources.

This module extracts the relevant MITRE ATT&CK entries and reshapes them into the same
corpus schema used by the rest of the project. It allows the benchmark to connect CVE
findings to the corresponding adversary techniques.
"""

from __future__ import annotations

import re
from typing import Any


def _strip_citations(text: str) -> str:
    return re.sub(r"\(Citation:.*?\)", "", text, flags=re.I | re.S)


def parse_attack_bundle(bundle: dict[str, Any], *, wanted: list[str]) -> list[dict[str, Any]]:
    objects = bundle.get("objects", [])
    by_id = {obj.get("id"): obj for obj in objects if obj.get("id")}
    relationships = bundle.get("relationships", [])
    mitigations: dict[str, list[str]] = {}
    for rel in relationships:
        if rel.get("relationship_type") != "mitigates":
            continue
        src = rel.get("source_ref")
        tgt = rel.get("target_ref")
        if src in mitigations:
            mitigations[src].append(tgt)
        else:
            mitigations[src] = [tgt]

    docs: list[dict[str, Any]] = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        external_ids = [ref.get("external_id") for ref in obj.get("external_references", []) if ref.get("external_id")]
        match = set(external_ids) & set(wanted)
        if not match:
            continue

        text_bits = [
            "## Description",
            _strip_citations(obj.get("description", "")).strip(),
            "## Tactics and platforms",
            f"- Kill-chain phases: {', '.join(p.get('phase_name', '') for p in obj.get('kill_chain_phases', []) if p.get('phase_name')) or ['n/a']}",
            f"- Platforms: {', '.join(obj.get('x_mitre_platforms', []) or ['n/a'])}",
            "## Detection",
            _strip_citations(obj.get("x_mitre_detection", "")).strip() or "n/a",
            "## Mitigations",
        ]

        mitigation_names: list[str] = []
        for target_id in mitigations.get(obj.get("id"), []):
            target = by_id.get(target_id)
            if target:
                name = target.get("name", "")
                description = _strip_citations(target.get("description", "")).strip()
                mitigation_names.append(f"- {name}: {description or 'n/a'}")
        text_bits.extend(mitigation_names or ["- none reported"])

        docs.append({
            "doc_id": f"attack_{next(iter(match))}",
            "source_type": "attack",
            "title": obj.get("name", next(iter(match))),
            "url": "https://attack.mitre.org/techniques/" + next(iter(match)),
            "date": None,
            "cve_ids": [],
            "text": "\n".join(text_bits),
        })

    return docs
