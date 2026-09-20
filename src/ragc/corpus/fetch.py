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

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "corpus_manifest.yaml"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def dry_run_manifest() -> list[dict[str, Any]]:
    manifest = load_manifest()
    return manifest.get("sources", [])


def lock_manifest() -> dict[str, Any]:
    manifest = load_manifest()
    rows = dry_run_manifest()
    hashes = {}
    for row in rows:
        hashes[row["doc_id"]] = hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()
    return {"files": hashes, "targets_by_type": manifest.get("targets_by_type", {})}


class CorpusFetcher:
    def __init__(self, manifest_path: str | Path | None = None):
        self.manifest_path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH

    def check_manifest_urls(self) -> dict[str, int]:
        manifest = load_manifest()
        rows = manifest.get("sources", [])
        counts = {"vendor": 0, "threatintel": 0, "total_good": 0}
        for row in rows:
            source_type = row.get("source_type")
            if source_type not in {"vendor", "threatintel"}:
                continue
            try:
                response = requests.get(row["url"], timeout=10)
            except requests.RequestException:
                continue
            if response.status_code < 400:
                counts[source_type] += 1
                counts["total_good"] += 1
        return counts

    def fetch_all(self) -> dict[str, Any]:
        return {"manifest": load_manifest(), "lock": lock_manifest(), "url_health": self.check_manifest_urls()}
