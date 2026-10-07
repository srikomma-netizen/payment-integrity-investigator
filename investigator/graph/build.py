"""Assemble the LangGraph StateGraph and expose a small facade.

    START ─▶ intake ─▶ supervisor ─┬─▶ gather_evidence ─┐
                          ▲        ├─▶ assess_risk ─────┤
                          │        ├─▶ retrieve_policy ─┘ (back to supervisor)
                          └────────┤
                                   └─▶ summarize ─▶ verify ─▶ route ─┬─▶ human_review ─▶ finalize ─▶ END
                                                                     └─▶ finalize ─▶ END

The supervisor is the planner; specialized nodes are executors. A
checkpointer makes `human_review`'s interrupt resumable by thread id.
"""
from __future__ import annotations

import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ..llm.providers import InvestigatorLLM, make_llm
from ..rag.index import HybridIndex
from ..security.phi import Deidentifier, Vault
from ..synthetic import Dataset, build_dataset
from ..tools.registry import ToolRegistry
from .nodes import Services
from .state import CaseState


def build_graph(services: Services, checkpointer=None):
    g = StateGraph(CaseState)
    g.add_node("intake", services.intake)
    g.add_node("supervisor", services.supervisor)
    g.add_node("gather_evidence", services.gather_evidence)
    g.add_node("assess_risk", services.assess_risk)
    g.add_node("retrieve_policy", services.retrieve_policy)
    g.add_node("summarize", services.summarize)
    g.add_node("verify", services.verify)
    g.add_node("route", services.route)
    g.add_node("human_review", services.human_review)
    g.add_node("finalize", services.finalize)

    g.add_edge(START, "intake")
    g.add_edge("intake", "supervisor")
    g.add_conditional_edges("supervisor", lambda s: "finalize" if s.get("status") == "failed" else s["next_step"],
                            ["gather_evidence", "assess_risk", "retrieve_policy", "summarize", "finalize"])
    for executor in ("gather_evidence", "assess_risk", "retrieve_policy"):
        g.add_conditional_edges(executor, lambda s: "finalize" if s.get("status") == "failed" else "supervisor",
                                ["supervisor", "finalize"])
    g.add_edge("summarize", "verify")
    g.add_edge("verify", "route")
    g.add_conditional_edges("route", lambda s: "human_review" if s["route"] == "human_review" else "finalize",
                            ["human_review", "finalize"])
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


class Investigator:
    """Facade used by the API, the evals, and the demo."""

    def __init__(self, ds: Dataset | None = None, llm: InvestigatorLLM | None = None, as_of: str = "2025-12-31"):
        self.ds = ds or build_dataset()
        self.deid = Deidentifier(Vault(), known_names=[m["name"] for m in self.ds.members.values()])
        self.registry = ToolRegistry(self.ds, self.deid)
        self.index = HybridIndex()
        self.llm = llm or make_llm()
        self.services = Services(self.registry, self.index, self.llm, self.deid)
        self.graph = build_graph(self.services)
        self.as_of = as_of

    @staticmethod
    def _cfg(case_id: str) -> dict:
        return {"configurable": {"thread_id": case_id}}

    def start(self, claim_id: str, role: str = "investigator") -> dict[str, Any]:
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        self.graph.invoke({"case_id": case_id, "claim_id": claim_id, "role": role, "as_of": self.as_of, "audit": []},
                          self._cfg(case_id))
        return self.get(case_id)

    def resume(self, case_id: str, *, decision: str, reviewer: str, rationale: str = "") -> dict[str, Any]:
        snap = self.graph.get_state(self._cfg(case_id))
        if not snap.tasks or not snap.tasks[0].interrupts:
            raise ValueError(f"{case_id} is not awaiting human review")
        self.graph.invoke(Command(resume={"decision": decision, "reviewer": reviewer, "rationale": rationale}), self._cfg(case_id))
        return self.get(case_id)

    def get(self, case_id: str) -> dict[str, Any]:
        snap = self.graph.get_state(self._cfg(case_id))
        state = dict(snap.values)
        pending = snap.tasks[0].interrupts[0].value if snap.tasks and snap.tasks[0].interrupts else None
        state["pending_review"] = pending
        if pending:
            state["status"] = "awaiting_human_review"
        return state

    def view(self, case_id: str, role: str) -> dict[str, Any]:
        """Role-scoped, re-identified view. Re-association happens here, never in the model."""
        state = self.get(case_id)
        keys = ("case_id", "claim_id", "status", "route", "summary", "verification", "human_decision", "claim", "pending_review")
        return self.deid.vault.rehydrate({k: state.get(k) for k in keys}, role)
