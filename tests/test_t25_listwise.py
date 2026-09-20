import json

from ragc.eval.listwise import judge_listwise, parse_listwise_payload
from ragc.llm.fake import FakeLLM


class JSONFakeLLM(FakeLLM):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        return type("Resp", (), {"text": json.dumps(self.payload)})()


def test_parse_listwise_payload_accepts_valid_json():
    payload = {"ranking": ["C", "A", "B"]}
    parsed = parse_listwise_payload(json.dumps(payload))
    assert parsed["ranking"] == ["C", "A", "B"]


def test_judge_listwise_recovers_original_labels_after_shuffle():
    llm = JSONFakeLLM({"ranking": ["B", "C", "A"]})
    candidates = {
        "s1": "Wrong answer.",
        "s2": "Best answer.",
        "s3": "Okay answer.",
    }

    result = judge_listwise(
        llm,
        question="Which version is affected?",
        gold_answer="Only 10.2.9-h1 is affected.",
        gold_claims=[{"id": "c1", "text": "Affected version is 10.2.9-h1."}],
        candidates=candidates,
        seed=42,
    )

    assert result["ranking"] == ["s2", "s3", "s1"]
    assert "Gold answer:" in llm.calls[0]["prompt"]
    assert "Candidates:" in llm.calls[0]["prompt"]
