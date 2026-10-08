"""Graph state."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class CaseState(TypedDict, total=False):
    case_id: str
    claim_id: str
    role: str
    as_of: str
    # evidence, masked
    claim: dict
    history: list[dict]
    payments: list[dict]
    provider: dict
    vendor: dict | None
    prior_cases: list[dict]
    tool_failures: list[str]   # last write wins, copy before appending
    evidence_gathered: bool
    # analysis
    risk: dict
    retrieval: dict
    summary: dict
    verification: dict
    # control
    next_step: str
    steps: int
    route: str
    human_decision: dict | None
    status: str
    audit: Annotated[list[str], operator.add]
