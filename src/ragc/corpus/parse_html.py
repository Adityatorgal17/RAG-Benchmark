"""HTML-to-document parser for plain web content.

This file normalizes raw HTML pages into a simple document structure with cleaned text,
metadata, and a standard section layout. It allows vendor and advisory pages to be fed
into the same retrieval pipeline as other corpus sources.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any


def parse_html_document(html: str, *, doc_id: str, source_type: str, url: str) -> dict[str, Any]:
    try:
        import trafilatura

        extracted = trafilatura.extract(html, output_format="markdown", include_tables=True, include_comments=False)
    except Exception:
        extracted = None

    text = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = unescape(text)
    text = re.sub(r"\n+", "\n", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
    title = title_match.group(1).strip() if title_match else doc_id
    title = re.sub(r"\s+", " ", title)

    if extracted and len(extracted.split()) >= 20:
        body = extracted.strip()
    else:
        paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip() and len(p.strip().split()) >= 3]
        body = "\n".join(paragraphs)
    body = "\n".join(line.strip() for line in body.splitlines() if len(line.strip().split()) >= 3 or line.strip().startswith("#"))

    doc = {
        "doc_id": doc_id,
        "source_type": source_type,
        "title": title,
        "url": url,
        "date": None,
        "cve_ids": [],
        "text": f"## Overview\n{body}",
    }
    return doc
