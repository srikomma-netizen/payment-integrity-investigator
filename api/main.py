"""HTTP API over the Investigator: start cases, role-scoped views, reviewer decisions, feedback summary.

Run:  uvicorn api.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from investigator.feedback.store import FeedbackStore
from investigator.graph.build import Investigator

# TODO: role is caller-supplied here; take it from the authenticated user once auth is in front of this
ROLE_PATTERN = "^(analyst|investigator|siu_lead|auditor)$"


class StartRequest(BaseModel):
    claim_id: str = Field(pattern=r"^CLM-\d{4}$")
    role: str = Field(default="investigator", pattern=ROLE_PATTERN)


class DecisionRequest(BaseModel):
    decision: str = Field(pattern="^(approve|reject|escalate)$")
    reviewer: str = Field(min_length=1, max_length=80)
    rationale: str = Field(default="", max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # single shared instance: cases live in its in-memory checkpointer, so run one worker
    app.state.investigator = Investigator()
    app.state.feedback = FeedbackStore()
    yield


app = FastAPI(title="Payment Integrity Investigator", version="0.1.0", lifespan=lifespan)


def _public(state: dict) -> dict:
    # Still masked (no rehydrate); GET /cases/{id} is the only role-scoped re-identified view.
    # Raw evidence is left out entirely, and retrieval is cut down to citations.
    keys = ("case_id", "claim_id", "role", "status", "route", "summary", "verification", "risk",
            "retrieval", "tool_failures", "human_decision", "pending_review", "audit")
    out = {k: state.get(k) for k in keys}
    if out.get("retrieval"):
        out["retrieval"] = {"query": out["retrieval"]["query"],
                            "citations": [c["citation"] for c in out["retrieval"]["chunks"]],
                            "rejected": out["retrieval"]["rejected"]}
    return out


@app.get("/health")
def health():
    return {"ok": True, "llm": type(app.state.investigator.llm).__name__}


@app.get("/tools")
def tools():
    return app.state.investigator.registry.specs()


@app.post("/cases")
def start_case(req: StartRequest):
    inv: Investigator = app.state.investigator
    if req.claim_id not in inv.ds.claims:
        raise HTTPException(404, "claim not found")
    return _public(inv.start(req.claim_id, role=req.role))


@app.get("/cases/{case_id}")
def get_case(case_id: str, role: str = "analyst"):   # least-privileged default; unknown roles see only tokens
    inv: Investigator = app.state.investigator
    try:
        view = inv.view(case_id, role)
    except Exception:
        raise HTTPException(404, "case not found")
    # an unknown thread id gives back empty state rather than raising
    if not view.get("case_id"):
        raise HTTPException(404, "case not found")
    # action names only, not the tokens, so the response doesn't echo what was resolved
    view["phi_resolutions"] = [a.action for a in inv.deid.vault.audit[-10:]]
    return view


@app.post("/cases/{case_id}/decision")
def decide(case_id: str, req: DecisionRequest):
    inv: Investigator = app.state.investigator
    try:
        state = inv.resume(case_id, decision=req.decision, reviewer=req.reviewer, rationale=req.rationale)
    except ValueError as e:
        raise HTTPException(409, str(e))   # not paused: unknown case or already decided
    # recorded after resume so the feedback row carries the final state the decision produced
    app.state.feedback.record(state, decision=req.decision, reviewer=req.reviewer, rationale=req.rationale)
    return _public(state)


@app.get("/feedback/summary")
def feedback_summary():
    fb: FeedbackStore = app.state.feedback
    return {"records": len(fb.records), "false_positive_by_rule": fb.false_positive_rate_by_rule(),
            "attribution": fb.attribution_summary(), "golden_candidates": fb.golden_candidates()}
