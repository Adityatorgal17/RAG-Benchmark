"""Corpus download and index build pipeline.

This module turns the manifest-defined sources into three persisted artifacts:
raw downloads under `data/raw`, processed documents/chunks under `data/processed`,
and retrieval indexes under `data/index`.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError as exc:  # pragma: no cover - dependency should exist in project env
    raise RuntimeError("chromadb is required to build the dense index. Install requirements.txt.") from exc

from ragc.config import load_config
from ragc.corpus.chunker import chunk_document
from ragc.corpus.fetch import MANIFEST_PATH, fetch_and_parse_source, load_manifest
from ragc.retrieval.bm25 import BM25Index
from ragc.retrieval.embeddings import TextEmbedder


ROOT = Path(__file__).resolve().parents[3]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CorpusBuildSummary:
    raw_files: int
    documents: int
    chunks: int
    bm25_path: str
    chroma_path: str | None
    lock_path: str


class CorpusBuilder:
    def __init__(self, *, manifest_path: str | Path | None = None, root: str | Path | None = None):
        self.root = Path(root) if root is not None else ROOT
        self.manifest_path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
        self.raw_dir = self.root / "data" / "raw"
        self.processed_dir = self.root / "data" / "processed"
        self.index_dir = self.root / "data" / "index"
        self.chroma_dir = self.index_dir / "chroma"
        self.bm25_path = self.index_dir / "bm25.pkl"
        self.docs_path = self.processed_dir / "docs.jsonl"
        self.chunks_path = self.processed_dir / "chunks.jsonl"
        self.lock_path = self.root / "data" / "corpus.lock"
        cfg = load_config()
        embedding_cfg = getattr(cfg, "embedding", None)
        self.embedder = TextEmbedder(getattr(embedding_cfg, "model", "BAAI/bge-small-en-v1.5"))

    def _chunk_config(self) -> dict[str, int]:
        cfg = load_config()
        chunking = getattr(cfg, "chunking", None)
        return {
            "target_words": int(getattr(chunking, "target_words", 180)),
            "overlap_words": int(getattr(chunking, "overlap_words", 30)),
            "min_words": int(getattr(chunking, "min_words", 25)),
        }

    def load_manifest(self) -> dict[str, Any]:
        return load_manifest(self.manifest_path)

    def fetch_raw(self, *, force: bool = False) -> list[tuple[Path, dict[str, Any]]]:
        manifest = self.load_manifest()
        rows = manifest.get("sources", [])
        results: list[tuple[Path, dict[str, Any]]] = []
        for row in rows:
            raw_path, doc = fetch_and_parse_source(row, self.raw_dir, force=force)
            results.append((raw_path, doc))
        return results

    def build_documents(self, *, force: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
        items = self.fetch_raw(force=force)
        docs = [doc for _, doc in items]
        chunks: list[dict[str, Any]] = []
        for doc in docs:
            chunks.extend(chunk_document(doc, **self._chunk_config()))
        _write_jsonl(self.docs_path, docs)
        _write_jsonl(self.chunks_path, chunks)
        return docs, chunks, [path for path, _ in items]

    def build_bm25(self, chunks: list[dict[str, Any]]) -> Path:
        _ensure_parent(self.bm25_path)
        index = BM25Index(load_config(), chunks)
        index.save(self.bm25_path)
        return self.bm25_path

    def build_chroma(self, chunks: list[dict[str, Any]], *, force: bool = False) -> Path | None:
        if force and self.chroma_dir.exists():
            shutil.rmtree(self.chroma_dir)

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            client.delete_collection("cve_corpus")
        except Exception:
            pass

        collection = client.get_or_create_collection(name="cve_corpus", metadata={"hnsw:space": "cosine"})
        if collection.count() and not force:
            return self.chroma_dir

        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            if not batch:
                continue
            texts = [f"{item.get('header', '')}\n\n{item.get('text', '')}".strip() for item in batch]
            embeddings = self.embedder.embed(texts)
            collection.add(
                ids=[str(item["chunk_id"]) for item in batch],
                documents=texts,
                metadatas=[
                    {
                        "chunk_id": item["chunk_id"],
                        "doc_id": item.get("doc_id", ""),
                        "source_type": item.get("source_type", ""),
                        "title": item.get("title", ""),
                        "section": item.get("section", ""),
                        "url": item.get("url", ""),
                        "date": item.get("date") or "",
                        "n_words": int(item.get("n_words", 0)),
                    }
                    for item in batch
                ],
                embeddings=embeddings,
            )
        return self.chroma_dir

    def build_lock(self, raw_paths: list[Path]) -> Path:
        manifest = self.load_manifest()
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": {path.stem: _file_sha256(path) for path in raw_paths},
            "targets_by_type": manifest.get("targets_by_type", {}),
        }
        _ensure_parent(self.lock_path)
        self.lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return self.lock_path

    def build_all(self, *, force: bool = False) -> CorpusBuildSummary:
        docs, chunks, raw_paths = self.build_documents(force=force)
        bm25_path = self.build_bm25(chunks)
        chroma_path = self.build_chroma(chunks, force=force)
        lock_path = self.build_lock(raw_paths)
        return CorpusBuildSummary(
            raw_files=len(raw_paths),
            documents=len(docs),
            chunks=len(chunks),
            bm25_path=str(bm25_path),
            chroma_path=str(chroma_path) if chroma_path is not None else None,
            lock_path=str(lock_path),
        )


def build_all(*, force: bool = False) -> CorpusBuildSummary:
    return CorpusBuilder().build_all(force=force)
