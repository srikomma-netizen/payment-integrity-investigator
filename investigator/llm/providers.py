"""Model boundary. Two providers behind one interface.

The interface is
provider-agnostic and this build ships a Claude adapter (official SDK,
structured outputs) plus a deterministic fake for tests and offline demos.
"""
from __future__ import annotations

import os
from typing import Any, Protocol

from .schemas import Indicator, InvestigationSummary, SupervisorDecision

DEFAULT_MODEL = os.environ.get("INVESTIGATOR_MODEL", "claude-opus-5-5")


class LLMRefusal(Exception):
    pass


class InvestigatorLLM(Protocol):
    def plan(self, state_view: dict[str, Any]) -> SupervisorDecision: ...
    def summarize(self, packet: dict[str, Any]) -> InvestigationSummary: ...


def default_plan(state_view: dict[str, Any]) -> SupervisorDecision:
    """Planner-executor default: the next missing piece of the investigation.
    Deterministic so routing is auditable; the LLM may override with a reason."""
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
        import anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model

    def _parse(self, system: str, user: str, schema):
        response = self.client.messages.parse(
            model=self.model, max_tokens=6000, system=system,
            messages=[{"role": "user", "content": user}], output_format=schema,
        )
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            raise LLMRefusal(getattr(detail, "explanation", "model declined"))
        return response.parsed_output

    def plan(self, state_view: dict[str, Any]) -> SupervisorDecision:
        import json
        user = f"Default plan: {default_plan(state_view).model_dump()}\n\nState:\n{json.dumps(state_view, indent=1)}"
        return self._parse(PLAN_SYSTEM, user, SupervisorDecision)

    def summarize(self, packet: dict[str, Any]) -> InvestigationSummary:
        import json
        return self._parse(SUMMARY_SYSTEM, json.dumps(packet, indent=1), InvestigationSummary)


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
    """Grounded, deterministic composer. It can only reference ids that are
    in the packet, which is exactly the property the real model is held to
    by the verify node."""

    def plan(self, state_view: dict[str, Any]) -> SupervisorDecision:
        return default_plan(state_view)

    def summarize(self, packet: dict[str, Any]) -> InvestigationSummary:
        risk = packet.get("risk", {})
        signals = risk.get("signals", [])
        priors = packet.get("prior_cases", [])
        level = risk.get("tier", "low")
        indicators = [Indicator(description=s["description"], evidence_ids=list(s["evidence_ids"]) + [s["id"]],
                                severity=s["severity"]) for s in signals]
        # PI-004 §2 prior-case adjustment
        if any(p.get("outcome") == "false_positive" for p in priors) and not any(p.get("outcome", "").startswith("confirmed") for p in priors):
            level = _shift(level, -1)
        elif any(p.get("outcome", "").startswith("confirmed") for p in priors):
            level = _shift(level, +1)
        citations = [c["chunk_id"] for c in packet.get("policy", [])[:3]]
        failures = packet.get("tool_failures", [])
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
    provider = (provider or os.environ.get("LLM_PROVIDER", "auto")).lower()
    if provider == "anthropic" or (provider == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicInvestigatorLLM()
    return FakeInvestigatorLLM()
