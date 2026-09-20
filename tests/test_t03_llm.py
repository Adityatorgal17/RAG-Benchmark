from ragc.llm.cache import LLMCache
from ragc.llm.fake import FakeLLM
from ragc.llm.keypool import KeyPool, QuotaExhausted


def test_fake_llm_and_cache_roundtrip(tmp_path):
    cache = LLMCache(path=tmp_path / "cache.sqlite")
    llm = FakeLLM(cache=cache)

    first = llm.generate("hello", json_mode=False)
    assert first.text == "OK"
    assert first.cached is False

    second = llm.generate("hello", json_mode=False)
    assert second.text == "OK"
    assert second.cached is True
    assert first.prompt_tokens == second.prompt_tokens


def test_key_pool_moves_to_next_key_on_quota():
    keys = ["k1", "k2", "k3"]
    pool = KeyPool(keys)

    def fail_once(key):
        if key == "k1":
            raise QuotaExhausted("daily quota exhausted")
        return {"ok": True}

    result = pool.run_with_fallback(fail_once)
    assert result == {"ok": True}
    assert pool.active_key == "k2"


def test_key_pool_raises_when_all_keys_are_exhausted():
    pool = KeyPool(["k1", "k2"])

    def always_fail(key):
        raise QuotaExhausted(f"quota on {key}")

    try:
        pool.run_with_fallback(always_fail)
        assert False, "Expected QuotaExhausted"
    except QuotaExhausted:
        pass
