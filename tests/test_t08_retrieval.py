from ragc.config import load_config
from ragc.retrieval.retriever import Retriever


def _chunks():
    return [
        {
            "chunk_id": "nvd_CVE-2024-3400#000",
            "doc_id": "nvd_CVE-2024-3400",
            "source_type": "nvd",
            "title": "CVE-2024-3400",
            "section": "Overview",
            "url": "https://example.com/a",
            "date": None,
            "cve_ids": ["CVE-2024-3400"],
            "header": "CVE-2024-3400 - Overview",
            "text": "PAN-OS is affected by a command injection issue in the web interface. The vulnerability impacts PAN-OS versions 10.2.9-h1 and later.",
            "n_words": 32,
        },
        {
            "chunk_id": "attack_T1190#000",
            "doc_id": "attack_T1190",
            "source_type": "attack",
            "title": "T1190",
            "section": "Description",
            "url": "https://example.com/b",
            "date": None,
            "cve_ids": [],
            "header": "T1190 - Description",
            "text": "Exploit public-facing applications. This technique covers internet-facing application exploitation and remote attack paths.",
            "n_words": 25,
        },
        {
            "chunk_id": "vendor_panos#000",
            "doc_id": "vendor_panos",
            "source_type": "vendor",
            "title": "PAN-OS advisory",
            "section": "Summary",
            "url": "https://example.com/c",
            "date": None,
            "cve_ids": ["CVE-2024-3400"],
            "header": "PAN-OS advisory - Summary",
            "text": "Affected versions include 10.2.9-h1. Customers should upgrade to a fixed release and review firewall exposures.",
            "n_words": 28,
        },
    ]


def test_retriever_supports_dense_bm25_and_hybrid_queries():
    cfg = load_config()
    retriever = Retriever(cfg, chunks=_chunks())

    dense_hits = retriever.dense("Which ATT&CK technique covers exploiting internet-facing applications?", 5)
    assert dense_hits
    assert any(hit.chunk_id == "attack_T1190#000" for hit in dense_hits)

    bm25_hits = retriever.bm25("10.2.9-h1", 5)
    assert bm25_hits
    assert any("10.2.9-h1" in retriever.get_chunk(hit.chunk_id)["text"] for hit in bm25_hits)

    hybrid_hits = retriever.hybrid("CVE-2024-3400 affected PAN-OS versions", 20, 20, 5)
    assert hybrid_hits
    assert any(hit.chunk_id in {"nvd_CVE-2024-3400#000", "vendor_panos#000"} for hit in hybrid_hits)
