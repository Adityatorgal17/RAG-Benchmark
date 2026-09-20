from ragc.util import normalize_whitespace, rrf_rank, split_sentences, token_estimate


def test_token_estimate_and_whitespace_normalization():
    text = "  Hello   world\nagain  "
    assert token_estimate(text) == 5
    assert normalize_whitespace(text) == "Hello world again"


def test_split_sentences_and_rrf_rank():
    text = "Alpha. Beta! Gamma? Delta."
    assert split_sentences(text) == ["Alpha.", "Beta!", "Gamma?", "Delta."]
    assert split_sentences("Version 10.2.9-h1 is patched.") == ["Version 10.2.9-h1 is patched."]

    ranks = [
        [10, 3, 5],
        [5, 10, 1],
    ]
    result = rrf_rank(ranks, k=60)
    assert result[:3] == [10, 5, 3]
    assert result[-1] == 1
