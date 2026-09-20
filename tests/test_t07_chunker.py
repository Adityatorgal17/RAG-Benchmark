from ragc.corpus.chunker import chunk_document


def test_chunk_document_respects_window_and_overlap_rules():
    text = (
        "## Overview\n"
        "This is sentence one. This is sentence two. This is sentence three. "
        "This is sentence four. This is sentence five. This is sentence six. "
        "This is sentence seven. This is sentence eight. This is sentence nine. "
        "This is sentence ten. This is sentence eleven. This is sentence twelve. "
        "This is sentence thirteen. This is sentence fourteen. This is sentence fifteen. "
        "This is sentence sixteen. This is sentence seventeen. This is sentence eighteen. "
        "This is sentence nineteen. This is sentence twenty. This is sentence twenty-one. "
        "This is sentence twenty-two. This is sentence twenty-three. This is sentence twenty-four. "
        "This is sentence twenty-five. This is sentence twenty-six. This is sentence twenty-seven. "
        "This is sentence twenty-eight. This is sentence twenty-nine. This is sentence thirty. "
        "This is sentence thirty-one. This is sentence thirty-two. This is sentence thirty-three. "
        "This is sentence thirty-four. This is sentence thirty-five. This is sentence thirty-six. "
        "This is sentence thirty-seven. This is sentence thirty-eight. This is sentence thirty-nine. "
        "This is sentence forty. This is sentence forty-one. This is sentence forty-two. "
        "This is sentence forty-three. This is sentence forty-four. This is sentence forty-five. "
        "This is sentence forty-six. This is sentence forty-seven. This is sentence forty-eight. "
        "This is sentence forty-nine. This is sentence fifty. This is sentence fifty-one."
    )

    chunks = chunk_document({
        "doc_id": "nvd_CVE-2024-3400",
        "source_type": "nvd",
        "title": "CVE-2024-3400",
        "url": "https://example.com",
        "date": None,
        "cve_ids": ["CVE-2024-3400"],
        "text": text,
    })

    assert chunks
    assert all(len(chunk["text"].split()) <= 180 for chunk in chunks)
    assert all(len(chunk["text"].split()) >= 25 for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert any("CVE-2024-3400" in chunk["cve_ids"] for chunk in chunks)

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    assert len(set(first_words[-30:]) & set(second_words[:30])) >= 1
