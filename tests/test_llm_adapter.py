from types import SimpleNamespace

import pytest

from investigator.llm.providers import AnthropicInvestigatorLLM, LLMRefusal
from investigator.llm.schemas import InvestigationSummary, SupervisorDecision


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


from investigator.llm.providers import GeminiInvestigatorLLM, make_llm  # noqa: E402


class StubModels:
    def __init__(self, responses):
        self.calls, self._responses = [], list(responses)

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_gemini_summarize_validates_json_against_schema():
    payload = {"risk_level": "high", "suspicious_indicators": [], "policy_citations": [],
               "recommended_action": "Hold.", "confidence": 0.8}
    import json
    client = SimpleNamespace(models=StubModels([SimpleNamespace(text=json.dumps(payload))]))
    out = GeminiInvestigatorLLM(model="gemini-test", client=client).summarize({"case_id": "CASE-1"})
    assert isinstance(out, InvestigationSummary) and out.risk_level == "high"
    cfg = client.models.calls[0]["config"]
    assert cfg.response_mime_type == "application/json" and "risk_level" in cfg.response_json_schema["properties"]
    assert "[PATIENT_1a2b]" in cfg.system_instruction


def test_gemini_blocked_reply_raises_refusal():
    blocked = SimpleNamespace(text=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"), candidates=[])
    client = SimpleNamespace(models=StubModels([blocked]))
    with pytest.raises(LLMRefusal, match="SAFETY"):
        GeminiInvestigatorLLM(client=client).plan({})


def test_provider_selection_prefers_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert isinstance(make_llm(), GeminiInvestigatorLLM)
