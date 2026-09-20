from ragc.config import load_config


def test_load_config():
    cfg = load_config()
    assert cfg.main_llm.model == "gemini-2.5-flash-lite"
    assert cfg.seed == 42


def test_pytest_runs_with_zero_tests():
    assert True
