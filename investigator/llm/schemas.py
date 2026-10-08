"""Structured output schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]
# keep in sync with supervisor edges in build.py
NextStep = Literal["gather_evidence", "assess_risk", "retrieve_policy", "summarize"]


class Indicator(BaseModel):
    description: str = Field(description="What looks suspicious, in one sentence, using tokens not names.")
    evidence_ids: list[str] = Field(description="Ids of records that support it: claim, payment, signal, event, or prior-case ids.")
    severity: RiskLevel


class InvestigationSummary(BaseModel):
    # descriptions go into the schema the model sees
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
