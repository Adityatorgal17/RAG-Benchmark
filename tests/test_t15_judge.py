import json

from ragc.eval.judge import judge_answer, parse_judge_payload
from ragc.llm.fake import FakeLLM


class JSONFakeLLM(FakeLLM):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def generate(self, prompt: str, *, json_mode: bool = False, system: str | None = None):
        return type("Resp", (), {"text": json.dumps(self.payload)})()


def test_parse_judge_payload_accepts_valid_json():
    payload = {"correctness": 2, "claims": {"c1": True, "c2": False}, "abstained": False, "contradicts_gold": []}
    parsed = parse_judge_payload(json.dumps(payload))
    assert parsed["correctness"] == 2
    assert parsed["claims"]["c1"] is True


def test_judge_answer_uses_gold_prompt_and_returns_structured_output():
    llm = JSONFakeLLM({"correctness": 1, "claims": {"c1": True}, "abstained": False, "contradicts_gold": ["version"]})

    result = judge_answer(
        llm,
        question="Which version is affected?",
        gold_answer="Only version 10.2.9-h1 is affected.",
        gold_claims=[{"id": "c1", "text": "Affected version is 10.2.9-h1."}],
        candidate_answer="The affected version is 10.2.9-h1. [nvd#123]",
    )

    assert result["correctness"] == 1
    assert result["contradicts_gold"] == ["version"]
    assert "Gold answer:" in llm.calls[0]["prompt"]
    assert "Candidate answer:" in llm.calls[0]["prompt"]
