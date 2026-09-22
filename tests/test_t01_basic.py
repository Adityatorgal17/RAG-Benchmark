from ragc.config import load_config


def test_load_config():
    cfg = load_config()
    assert cfg.main_llm.provider == "groq"
    assert cfg.main_llm.model == "openai/gpt-oss-20b"
    assert cfg.judge_llm.provider == "gemini"
    assert cfg.seed == 42


def test_pytest_runs_with_zero_tests():
    assert True
