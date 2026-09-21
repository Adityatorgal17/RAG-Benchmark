"""NVD record parser for security advisories.

This module converts raw NVD JSON into the project’s standard document format so CVE
metadata and affected product information can be chunked and retrieved consistently.
"""

from __future__ import annotations

from typing import Any


def _flatten_cwe(weaknesses):
    cwes: list[str] = []
    for item in weaknesses or []:
        for desc in item.get("description", []) or []:
            value = desc.get("value")
            if value:
                cwes.append(value)
    return cwes


def parse_nvd_record(payload: dict[str, Any], *, doc_id: str, url: str) -> dict[str, Any]:
    if "vulnerabilities" in payload:
        vulnerabilities = payload.get("vulnerabilities") or []
        cve = (vulnerabilities[0] or {}).get("cve", {}) if vulnerabilities else {}
    else:
        cve = payload.get("cve", {})
    desc = cve.get("descriptions") or []
    english = next((item.get("value") for item in desc if item.get("lang") == "en"), "")

    metrics = cve.get("metrics", {})
    scores: list[str] = []
    for metric_list in metrics.values():
        for metric in metric_list or []:
            cvss = metric.get("cvssData", {})
            base_score = cvss.get("baseScore")
            severity = cvss.get("baseSeverity")
            vector = cvss.get("vectorString")
            if base_score is not None:
                scores.append(f"- Base score: {base_score} ({severity or 'unknown'}), vector: {vector or 'n/a'}")

    cwes = _flatten_cwe(cve.get("weaknesses"))
    entries: list[str] = []
    for node in (cve.get("configurations") or []):
        for match in (node.get("nodes") or []):
            for cpe in (match.get("cpeMatch") or []):
                if not cpe.get("vulnerable"):
                    continue
                criteria = cpe.get("criteria", "")
                start = cpe.get("versionStartIncluding")
                end = cpe.get("versionEndIncluding")
                start_ex = cpe.get("versionStartExcluding")
                end_ex = cpe.get("versionEndExcluding")
                details = [criteria]
                if start:
                    details.append(f"versionStartIncluding={start}")
                if start_ex:
                    details.append(f"versionStartExcluding={start_ex}")
                if end:
                    details.append(f"versionEndIncluding={end}")
                if end_ex:
                    details.append(f"versionEndExcluding={end_ex}")
                entries.append("- " + "; ".join(details))

    refs = []
    for ref in cve.get("references") or []:
        href = ref.get("url")
        tags = ", ".join(ref.get("tags") or [])
        if href:
            refs.append(f"- {href} [{tags}]" if tags else f"- {href}")

    weakness_lines = [f"- {cwe}" for cwe in cwes] if cwes else ["- none reported"]
    affected_lines = entries if entries else ["- none reported"]
    reference_lines = refs if refs else ["- none reported"]

    lines = [
        "## Overview",
        f"- CVE id: {cve.get('id', doc_id)}",
        f"- Published: {cve.get('published', 'unknown')}",
        f"- Last modified: {cve.get('lastModified', 'unknown')}",
        f"- Description: {english}",
        "## Severity",
        *scores,
        "## Weakness",
        *weakness_lines,
        "## Affected products",
        *affected_lines,
        "## References",
        *reference_lines,
    ]

    return {
        "doc_id": doc_id,
        "source_type": "nvd",
        "title": cve.get("id", doc_id),
        "url": url,
        "date": cve.get("published"),
        "cve_ids": [cve.get("id", doc_id)],
        "text": "\n".join(lines),
    }
