"""Anthropic adapter against a stub client: request shape and response handling, no network."""
from types import SimpleNamespace

import pytest

from investigator.llm.providers import AnthropicInvestigatorLLM, LLMRefusal
from investigator.llm.schemas import InvestigationSummary, SupervisorDecision


# Mimics client.messages: records each request and returns canned responses in order.
class StubMessages:
    def __init__(self, responses):
        self.calls, self._responses = [], list(responses)

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_summarize_uses_structured_output():
    summary = InvestigationSummary(risk_level="high", suspicious_indicators=[], policy_citations=[],
                                   recommended_action="Hold.", confidence=0.8)
    client = SimpleNamespace(messages=StubMessages([SimpleNamespace(stop_reason="end_turn", parsed_output=summary)]))
    llm = AnthropicInvestigatorLLM(model="claude-opus-5-5", client=client)
    assert llm.summarize({"case_id": "CASE-1", "risk": {}}) is summary
    call = client.messages.calls[0]
    assert call["output_format"] is InvestigationSummary and call["model"] == "claude-opus-5-5"
    # system prompt must keep the token-format instruction so the model doesn't guess names
    assert "[PATIENT_1a2b]" in call["system"] and "CASE-1" in call["messages"][0]["content"]


def test_plan_includes_default_and_handles_refusal():
    decision = SupervisorDecision(next_step="summarize", reason="ok")
    client = SimpleNamespace(messages=StubMessages([
        SimpleNamespace(stop_reason="end_turn", parsed_output=decision),
        SimpleNamespace(stop_reason="refusal", parsed_output=None, stop_details=SimpleNamespace(explanation="declined")),
    ]))
    llm = AnthropicInvestigatorLLM(client=client)
    assert llm.plan({"evidence_gathered": True, "risk": True, "retrieval": True}) is decision
    assert "Default plan" in client.messages.calls[0]["messages"][0]["content"]
    with pytest.raises(LLMRefusal, match="declined"):
        llm.plan({})
