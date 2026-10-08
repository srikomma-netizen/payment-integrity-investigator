"""Structured outputs the model must produce.

Anything a decision depends on is an enum or an id the verify node can check; free text stays in descriptions.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]
# must match the supervisor's conditional edges in graph/build.py; "finalize" is deliberately not choosable
NextStep = Literal["gather_evidence", "assess_risk", "retrieve_policy", "summarize"]


class Indicator(BaseModel):
    description: str = Field(description="What looks suspicious, in one sentence, using tokens not names.")
    evidence_ids: list[str] = Field(description="Ids of records that support it: claim, payment, signal, event, or prior-case ids.")
    severity: RiskLevel


class InvestigationSummary(BaseModel):
    # Field descriptions end up in the JSON schema the model sees, so they double as instructions.
    risk_level: RiskLevel
    suspicious_indicators: list[Indicator]
    policy_citations: list[str] = Field(description="chunk_ids of the retrieved policy text relied on.")
    recommended_action: str = Field(description="One concrete next action per the cited policy.")
    escalation_reason: str = Field(default="", description="Why a human must review, or empty.")
    confidence: float = Field(ge=0.0, le=1.0)
    missing_evidence: list[str] = Field(default_factory=list, description="Evidence that could not be obtained.")


class SupervisorDecision(BaseModel):
    next_step: NextStep
    reason: str
