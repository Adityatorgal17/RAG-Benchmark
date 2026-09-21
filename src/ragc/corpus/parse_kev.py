"""CISA KEV catalog parser."""

from __future__ import annotations

from typing import Any


def parse_kev_catalog(payload: dict[str, Any], *, cve_id: str, doc_id: str, url: str) -> dict[str, Any] | None:
    """Return one document for ``cve_id`` if present in the KEV catalog."""
    rows = payload.get("vulnerabilities") or payload.get("items") or []
    wanted = cve_id.upper()
    entry = None
    for row in rows:
        if str(row.get("cveID") or row.get("cveId") or row.get("cve") or "").upper() == wanted:
            entry = row
            break
    if entry is None:
        return None

    def field(name: str, default: str = "") -> str:
        return str(entry.get(name, default) or default)

    title = field("vulnerabilityName", f"CISA KEV {wanted}")
    lines = [
        "## KEV entry",
        f"- cveID: {field('cveID', wanted)}",
        f"- vendorProject: {field('vendorProject')}",
        f"- product: {field('product')}",
        f"- vulnerabilityName: {title}",
        f"- dateAdded: {field('dateAdded')}",
        f"- shortDescription: {field('shortDescription')}",
        f"- requiredAction: {field('requiredAction')}",
        f"- dueDate: {field('dueDate')}",
        f"- knownRansomwareCampaignUse: {field('knownRansomwareCampaignUse')}",
        f"- notes: {field('notes')}",
    ]
    return {
        "doc_id": doc_id,
        "source_type": "kev",
        "title": title,
        "url": url,
        "date": field("dateAdded") or None,
        "cve_ids": [wanted],
        "text": "\n".join(lines),
    }
