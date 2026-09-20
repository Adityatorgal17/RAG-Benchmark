from ragc.eval.nli_faith import NLIFaith
from ragc.retrieval.reranker import Reranker


def test_reranker_and_nli_faithfulness():
    reranker = Reranker(model_name="dummy", batch_size=2, max_length=128, verbose=False)
    scores = reranker.score([
        (
            "PAN-OS command injection issue in the web interface",
            "CVE-2024-3400 affects PAN-OS through a command injection flaw in the web interface",
        ),
        (
            "PAN-OS command injection issue in the web interface",
            "The issue is unrelated to PAN-OS and only affects Linux kernel logs",
        ),
    ])
    assert scores[0] > scores[1]

    nli = NLIFaith(model_name="dummy", max_length=128, verbose=False)
    entail = nli.entail_prob(
        "CVE-2024-3400 is a command injection issue in PAN-OS.",
        "The vulnerability is a command injection flaw affecting PAN-OS.",
    )
    contradiction = nli.entail_prob(
        "CVE-2024-3400 is a command injection issue in PAN-OS.",
        "The vulnerability is unrelated to PAN-OS and affects only Windows services.",
    )
    assert entail > 0.5
    assert contradiction < 0.5
