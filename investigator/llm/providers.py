"""Model boundary: one InvestigatorLLM interface, Gemini and Anthropic adapters, and a deterministic fake.

The fake is what tests and offline runs use; make_llm() picks based on env.
"""
from __future__ import annotations

import os
from typing import Any, Protocol

from .schemas import Indicator, InvestigationSummary, SupervisorDecision

DEFAULT_MODEL = os.environ.get("INVESTIGATOR_MODEL", "claude-opus-5-5")


class LLMRefusal(Exception):
    """The model declined. Nodes catch this and fall back or fail the case; it never crashes the graph."""


class InvestigatorLLM(Protocol):
    def plan(self, state_view: dict[str, Any]) -> SupervisorDecision: ...
    def summarize(self, packet: dict[str, Any]) -> InvestigationSummary: ...


def default_plan(state_view: dict[str, Any]) -> SupervisorDecision:
    """Planner-executor default: the next missing piece of the investigation.
    Deterministic so routing is auditable; the LLM may override with a reason."""
    # order matters: retrieval builds its query from risk signals, and risk needs the claim
    if not state_view.get("evidence_gathered"):
        return SupervisorDecision(next_step="gather_evidence", reason="no evidence collected yet")
    if not state_view.get("risk"):
        return SupervisorDecision(next_step="assess_risk", reason="risk signals not assessed")
    if not state_view.get("retrieval"):
        return SupervisorDecision(next_step="retrieve_policy", reason="no governing policy retrieved")
    return SupervisorDecision(next_step="summarize", reason="evidence, risk and policy available")


SUMMARY_SYSTEM = """You are a payment-integrity investigation assistant. You write the case summary an
investigator reads first. Rules:
- Use ONLY the evidence records and policy text in the packet. Every indicator must list the ids of the
  records that support it. Cite policy by chunk_id. Never invent ids.
- Member and provider identities appear as tokens like [PATIENT_1a2b]. Use the tokens; never guess names.
- Risk level follows the cited policy: apply prior-case adjustments exactly as the policy states.
- If a tool failed and evidence is missing, say so in missing_evidence and lower confidence.
- The recommended action must be one the cited policy allows. You are advisory; a human decides."""

PLAN_SYSTEM = """You are the supervisor of a payment-integrity investigation. Choose the next step from the
allowed values. Prefer the default plan unless the state shows a reason to deviate (for example, evidence
was gathered but a critical tool failed and should be retried before summarizing)."""


class AnthropicInvestigatorLLM:
    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        import anthropic   # lazy so the fake path (tests, CI) doesn't need the SDK installed
        self.client = client or anthropic.Anthropic()
        self.model = model

    def _parse(self, system: str, user: str, schema):
        # structured output parsed straight into the pydantic schema, so no JSON scraping from free text
        response = self.client.messages.parse(
            model=self.model, max_tokens=6000, system=system,
            messages=[{"role": "user", "content": user}], output_format=schema,
        )
        # a refusal has no parsed output; raise something typed instead of returning None
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)   # getattr: stop_details may be absent on the response
            raise LLMRefusal(getattr(detail, "explanation", "model declined"))
        return response.parsed_output

    def plan(self, state_view: dict[str, Any]) -> SupervisorDecision:
        import json
        # hand the model the deterministic plan so deviating from it is an explicit, explained choice
        user = f"Default plan: {default_plan(state_view).model_dump()}\n\nState:\n{json.dumps(state_view, indent=1)}"
        return self._parse(PLAN_SYSTEM, user, SupervisorDecision)

    def summarize(self, packet: dict[str, Any]) -> InvestigationSummary:
        import json
        return self._parse(SUMMARY_SYSTEM, json.dumps(packet, indent=1), InvestigationSummary)


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def gemini_api_key() -> str | None:
    # the SDK accepts either name; GOOGLE_API_KEY wins if both are set, same as the SDK
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


class GeminiInvestigatorLLM(AnthropicInvestigatorLLM):
    """Gemini via google-genai. Reuses the prompts above; only the client and parsing differ."""

    def __init__(self, model: str = GEMINI_MODEL, client=None):
        from google import genai   # lazy, same reason as the anthropic import
        from google.genai import types
        self._types = types
        self.client = client or genai.Client(api_key=gemini_api_key())
        self.model = model
        self.label = f"Gemini · {model}"

    def _parse(self, system: str, user: str, schema):
        response = self.client.models.generate_content(
            model=self.model, contents=user,
            config=self._types.GenerateContentConfig(
                system_instruction=system, response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(), temperature=0),
        )
        if not response.text:
            # blocked prompt or safety stop: nothing to parse, so treat it like a refusal
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None) or getattr((response.candidates or [None])[0], "finish_reason", None)
            raise LLMRefusal(f"Gemini returned no text ({reason})")
        return schema.model_validate_json(response.text)


ACTION_BY_SIGNAL = {
    "duplicate_payment": "Confirm no reversal exists, then raise a recovery request for the second payment.",
    "amount_outlier": "Request medical records for the visit and compare documented complexity to the billed level.",
    "unbundling": "Reprice to the bundled panel code and issue a coding education letter.",
    "prior_confirmed_case": "Review prior case findings for the same pattern before outreach.",
    "unverified_bank_change": "Hold payment, perform call-back to the number on file, and notify the SIU lead.",
}
_LEVELS = ["low", "medium", "high"]


def _shift(level: str, delta: int) -> str:
    return _LEVELS[max(0, min(2, _LEVELS.index(level) + delta))]


class FakeInvestigatorLLM:
    """Deterministic summarizer for tests and offline runs.

    Only references ids present in the packet, the same property verify checks on the real model."""

    def plan(self, state_view: dict[str, Any]) -> SupervisorDecision:
        return default_plan(state_view)

    def summarize(self, packet: dict[str, Any]) -> InvestigationSummary:
        risk = packet.get("risk", {})
        signals = risk.get("signals", [])
        priors = packet.get("prior_cases", [])
        level = risk.get("tier", "low")
        # cite the signal id along with its evidence so each indicator traces back to the rule that raised it
        indicators = [Indicator(description=s["description"], evidence_ids=list(s["evidence_ids"]) + [s["id"]],
                                severity=s["severity"]) for s in signals]
        # PI-004 §2 prior-case adjustment; a confirmed prior outweighs any false positive on the same case
        if any(p.get("outcome") == "false_positive" for p in priors) and not any(p.get("outcome", "").startswith("confirmed") for p in priors):
            level = _shift(level, -1)
        elif any(p.get("outcome", "").startswith("confirmed") for p in priors):
            level = _shift(level, +1)
        citations = [c["chunk_id"] for c in packet.get("policy", [])[:3]]
        failures = packet.get("tool_failures", [])
        # Arbitrary but monotonic: each failed tool and a missing policy citation each cost confidence.
        confidence = 0.9 if signals else 0.75
        confidence -= 0.2 * len(failures)
        if not citations:
            confidence -= 0.2
        action = ACTION_BY_SIGNAL.get(signals[0]["type"], "Close with logged rationale.") if signals else "Close with logged rationale."
        reason = ""
        if level == "high":
            reason = "High-tier finding requires investigator approval before outbound action (PI-004 §3)."
        elif failures:
            reason = "Evidence incomplete due to tool failure."
        return InvestigationSummary(
            risk_level=level, suspicious_indicators=indicators, policy_citations=citations,
            recommended_action=action, escalation_reason=reason, confidence=round(max(0.0, confidence), 2),
            missing_evidence=[f"{f} unavailable" for f in failures],
        )


def make_llm(provider: str | None = None) -> InvestigatorLLM:
    # "auto" picks Gemini, then Claude, based on which key is set; no key means offline
    provider = (provider or os.environ.get("LLM_PROVIDER", "auto")).lower()
    if provider == "fake":
        return FakeInvestigatorLLM()
    if provider == "gemini" or (provider == "auto" and gemini_api_key()):
        return GeminiInvestigatorLLM()
    if provider == "anthropic" or (provider == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicInvestigatorLLM()
    return FakeInvestigatorLLM()
