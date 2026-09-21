from ragc.ui_backend import build_ui_response


def test_build_ui_response_uses_backend_system_result():
    payload = build_ui_response("S0", "What is the main threat in this corpus?")

    assert payload["system"] == "S0"
    assert payload["answer"]
    assert isinstance(payload["trace"], list)
    assert isinstance(payload["evidence"], list)
    assert isinstance(payload["steps"], list)


def test_build_ui_response_supports_post_retrieval_systems():
    for system_id in ["S2", "S3"]:
        payload = build_ui_response(system_id, "What is the main threat in this corpus?")

        assert payload["system"] == system_id
        assert payload["answer"]
        assert isinstance(payload["evidence"], list)
