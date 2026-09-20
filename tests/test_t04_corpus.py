from ragc.corpus.fetch import dry_run_manifest, load_manifest


def test_load_manifest_counts_targets():
    manifest = load_manifest()
    by_type = manifest["targets_by_type"]
    assert by_type["nvd"] == 13
    assert by_type["kev"] == 13
    assert by_type["vendor"] == 6
    assert by_type["threatintel"] == 7
    assert by_type["attack"] == 9


def test_dry_run_manifest():
    rows = dry_run_manifest()
    assert rows
    assert any(item["doc_id"].startswith("nvd_") for item in rows)
    assert any(item["doc_id"].startswith("attack_") for item in rows)
    assert sum(1 for item in rows if item["source_type"] == "nvd") == 13
    assert sum(1 for item in rows if item["source_type"] == "attack") == 9
