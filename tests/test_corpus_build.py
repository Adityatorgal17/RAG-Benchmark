from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ragc.corpus.build_index import CorpusBuilder
from ragc.retrieval.retriever import Retriever


class _FakeResponse:
    def __init__(self, *, text: str = "", json_data=None, status_code: int = 200):
        self.text = text
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


def _fake_manifest() -> dict[str, object]:
    return {
        "sources": [
            {
                "source_type": "nvd",
                "doc_id": "nvd_CVE-2024-0001",
                "url": "https://example.test/nvd",
                "title": "CVE-2024-0001",
                "cve_ids": ["CVE-2024-0001"],
            },
            {
                "source_type": "vendor",
                "doc_id": "vendor_example",
                "url": "https://example.test/vendor",
                "title": "Vendor advisory",
                "cve_ids": ["CVE-2024-0001"],
            },
        ],
        "targets_by_type": {"nvd": 1, "vendor": 1},
    }


def test_corpus_builder_downloads_and_builds_indexes(tmp_path, monkeypatch):
    manifest_path = tmp_path / "corpus_manifest.yaml"
    manifest_path.write_text(
        """
sources:
  - source_type: nvd
    doc_id: nvd_CVE-2024-0001
    url: https://example.test/nvd
    title: CVE-2024-0001
    cve_ids: [CVE-2024-0001]
  - source_type: vendor
    doc_id: vendor_example
    url: https://example.test/vendor
    title: Vendor advisory
    cve_ids: [CVE-2024-0001]
targets_by_type:
  nvd: 1
  vendor: 1
""",
        encoding="utf-8",
    )

    nvd_payload = {
        "cve": {
            "id": "CVE-2024-0001",
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-01-02T00:00:00.000",
            "descriptions": [{"lang": "en", "value": "PAN-OS command injection vulnerability."}],
            "metrics": {},
            "weaknesses": [],
            "configurations": [],
            "references": [],
        }
    }

    def fake_get(url, timeout=30):
        if url.endswith("/nvd"):
            return _FakeResponse(json_data=nvd_payload)
        return _FakeResponse(text="<html><head><title>Vendor advisory</title></head><body><p>Apply the vendor patch and restrict exposure to trusted networks.</p></body></html>")

    monkeypatch.setattr("ragc.corpus.fetch.requests.get", fake_get)

    builder = CorpusBuilder(manifest_path=manifest_path, root=tmp_path)
    summary = builder.build_all(force=True)

    assert summary.raw_files == 2
    assert summary.documents == 2
    assert summary.chunks >= 2
    assert Path(summary.bm25_path).exists()
    assert summary.chroma_path is not None
    assert Path(summary.chroma_path).exists()
    assert builder.docs_path.exists()
    assert builder.chunks_path.exists()
    assert builder.lock_path.exists()

    chunks = [json.loads(line) for line in builder.chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(chunk["source_type"] == "nvd" for chunk in chunks)
    assert any(chunk["source_type"] == "vendor" for chunk in chunks)

    cfg = SimpleNamespace(paths=SimpleNamespace(chunks=str(builder.chunks_path), chroma=str(builder.chroma_dir), bm25=str(builder.bm25_path)))
    retriever = Retriever(cfg)

    dense_hits = retriever.dense("PAN-OS patch exposure", 3)
    assert dense_hits
    assert any(hit.chunk_id.startswith("vendor_example") for hit in dense_hits)

    bm25_hits = retriever.bm25("CVE-2024-0001", 3)
    assert bm25_hits
    assert any(hit.chunk_id.startswith("nvd_CVE-2024-0001") for hit in bm25_hits)

    hybrid_hits = retriever.hybrid("PAN-OS CVE-2024-0001", 3, 3, 3)
    assert hybrid_hits