from ragc.systems.base import CallBudget, System
from ragc.llm.fake import FakeLLM
from ragc.schemas import SystemResult


class DemoSystem(System):
    name = "demo"

    def run(self, qid: str, question: str) -> SystemResult:
        self.budget.spend(1)
        return SystemResult(
            system=self.name,
            qid=qid,
            question=question,
            answer="ok",
            abstained=False,
            cited_ids=[],
            candidate_ids=[],
            final_evidence=[],
            usage={"llm_calls": 1},
        )


def test_call_budget_prevents_overrun():
    budget = CallBudget(max_calls=4, reserve_final=1)
    assert budget.can_spend(2) is True
    budget.spend(2)
    assert budget.can_spend(1) is True
    budget.spend(1)
    assert budget.can_spend(1) is False


def test_system_uses_budget_and_returns_result():
    system = DemoSystem(cfg={}, llm=FakeLLM(), retriever=None, reranker=None)
    result = system.run("q1", "What is the CVE?")
    assert result.system == "demo"
    assert result.answer == "ok"
    assert system.budget.used == 1
