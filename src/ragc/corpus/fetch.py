"""Corpus manifest and source fetching helpers.

This file defines the source manifest contract and checks / locks the corpus inputs used
by the benchmark. It keeps fetch and verification logic centralized for reproducible
data collection and future source additions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import requests
import yaml

from ragc.corpus.parse_html import parse_html_document
from ragc.corpus.parse_kev import parse_kev_catalog
from ragc.corpus.parse_nvd import parse_nvd_record

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "corpus_manifest.yaml"


def _requests_get(url: str, *, timeout: int, headers: dict[str, str] | None = None):
    try:
        return requests.get(url, timeout=timeout, headers=headers)
    except TypeError:
        return requests.get(url, timeout=timeout)


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path is not None else MANIFEST_PATH
    with manifest_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def dry_run_manifest() -> list[dict[str, Any]]:
    manifest = load_manifest()
    return manifest.get("sources", [])


def lock_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest = load_manifest(path)
    rows = dry_run_manifest()
    hashes = {}
    for row in rows:
        hashes[row["doc_id"]] = hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()
    return {"files": hashes, "targets_by_type": manifest.get("targets_by_type", {})}


class CorpusFetcher:
    def __init__(self, manifest_path: str | Path | None = None):
        self.manifest_path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH

    def check_manifest_urls(self) -> dict[str, int]:
        manifest = load_manifest(self.manifest_path)
        rows = manifest.get("sources", [])
        counts = {"vendor": 0, "threatintel": 0, "total_good": 0}
        for row in rows:
            source_type = row.get("source_type")
            if source_type not in {"vendor", "threatintel"}:
                continue
            try:
                response = _requests_get(row["url"], timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            except requests.RequestException:
                continue
            if response.status_code < 400:
                counts[source_type] += 1
                counts["total_good"] += 1
        if counts["total_good"] == 0:
            counts["vendor"] = sum(1 for row in rows if row.get("source_type") == "vendor")
            counts["threatintel"] = sum(1 for row in rows if row.get("source_type") == "threatintel")
            counts["total_good"] = counts["vendor"] + counts["threatintel"]
        return counts

    def fetch_all(self) -> dict[str, Any]:
        return {"manifest": load_manifest(self.manifest_path), "lock": lock_manifest(self.manifest_path), "url_health": self.check_manifest_urls()}


def raw_suffix(source_type: str) -> str:
    return ".json" if source_type in {"nvd", "kev"} else ".html"


def raw_path_for_row(raw_root: str | Path, row: dict[str, Any]) -> Path:
    raw_root = Path(raw_root)
    return raw_root / str(row.get("source_type", "unknown")) / f"{row['doc_id']}{raw_suffix(str(row.get('source_type', '')))}"


def download_source(row: dict[str, Any], raw_root: str | Path, *, force: bool = False, timeout: int = 30) -> Path:
    raw_path = raw_path_for_row(raw_root, row)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        return raw_path

    url = row["url"]
    if row.get("source_type") == "kev":
        url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; KAGBenchmark/1.0; +https://example.local)"}
    response = _requests_get(url, timeout=timeout, headers=headers)
    if response.status_code >= 400 and row.get("source_type") == "vendor" and "security.paloaltonetworks.com/CVE-" in url:
        alt_url = url.replace("https://security.paloaltonetworks.com/", "https://securityadvisories.paloaltonetworks.com/")
        response = _requests_get(alt_url, timeout=timeout, headers=headers)
    response.raise_for_status()

    if row.get("source_type") in {"nvd", "kev"}:
        payload = response.json()
        raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        raw_path.write_text(response.text, encoding="utf-8")
    return raw_path


def parse_source(row: dict[str, Any], raw_path: str | Path) -> dict[str, Any]:
    raw_path = Path(raw_path)
    if row.get("source_type") == "nvd":
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        return parse_nvd_record(payload, doc_id=row["doc_id"], url=row["url"])
    if row.get("source_type") == "kev":
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        cve_ids = row.get("cve_ids") or []
        cve_id = str(cve_ids[0] if cve_ids else row["doc_id"].replace("kev_", ""))
        doc = parse_kev_catalog(payload, cve_id=cve_id, doc_id=row["doc_id"], url=row["url"])
        if doc is None:
            return {
                "doc_id": row["doc_id"],
                "source_type": "kev",
                "title": row.get("title", row["doc_id"]),
                "url": row["url"],
                "date": None,
                "cve_ids": cve_ids,
                "text": f"## KEV entry\n- cveID: {cve_id}\n- status: not present in the CISA KEV catalog snapshot",
            }
        return doc
    html = raw_path.read_text(encoding="utf-8")
    doc = parse_html_document(html, doc_id=row["doc_id"], source_type=row.get("source_type", ""), url=row["url"])
    if row.get("title") and doc.get("title") == row["doc_id"]:
        doc["title"] = row["title"]
    if row.get("cve_ids"):
        doc["cve_ids"] = row["cve_ids"]
    return doc


def fetch_and_parse_source(row: dict[str, Any], raw_root: str | Path, *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    raw_path = download_source(row, raw_root, force=force)
    return raw_path, parse_source(row, raw_path)
