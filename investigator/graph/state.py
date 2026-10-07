"""Graph state. Every node reads this and returns a partial update.

`audit` uses an additive reducer so nodes append without clobbering, which
is also what makes the trail survive a checkpoint/resume.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class CaseState(TypedDict, total=False):
    case_id: str
    claim_id: str
    role: str
    as_of: str
    # evidence (all masked at the tool boundary)
    claim: dict
    history: list[dict]
    payments: list[dict]
    provider: dict
    vendor: dict | None
    prior_cases: list[dict]
    tool_failures: list[str]
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
