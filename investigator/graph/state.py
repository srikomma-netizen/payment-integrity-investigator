"""Graph state. Every node reads this and returns a partial update."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


# total=False: nodes return partial dicts and most keys don't exist until their node has run
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
    tool_failures: list[str]   # last-write-wins, so nodes copy the existing list before appending
    evidence_gathered: bool
    # analysis
    risk: dict
    retrieval: dict
    summary: dict
    verification: dict
    # control
    next_step: str
    steps: int   # supervisor loop counter, bounded by MAX_STEPS
    route: str
    human_decision: dict | None
    status: str
    # additive reducer: nodes append audit lines without clobbering, and the trail survives checkpoint/resume
    audit: Annotated[list[str], operator.add]
