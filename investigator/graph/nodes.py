"""Investigation graph nodes: plain methods over CaseState, testable without LangGraph."""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from ..llm.providers import InvestigatorLLM, LLMRefusal
from ..llm.schemas import InvestigationSummary
from ..rag.index import HybridIndex, RetrievalFilter
from ..security.phi import Deidentifier
from ..tools.registry import ToolError, ToolRegistry
from .state import CaseState

MAX_STEPS = 12      # guard against a planner that keeps choosing the same step
TOOL_RETRIES = 1    # one immediate retry covers blips; anything longer should degrade, not stall the case


class Services:
    def __init__(self, registry: ToolRegistry, index: HybridIndex, llm: InvestigatorLLM, deid: Deidentifier):
        self.registry, self.index, self.llm, self.deid = registry, index, llm, deid

    # ---- helpers ----------------------------------------------------------
    def _call(self, s: CaseState, tool: str, failures: list[str], **args) -> Any:
        """Call a tool with retry. On final failure, record it in `failures` and return None."""
        last: Exception | None = None
        # NOTE: permission denials are retried too; harmless but wasteful. No backoff between attempts.
        for attempt in range(TOOL_RETRIES + 1):
            try:
                return self.registry.call(tool, s["role"], **args)
            except ToolError as e:
                last = e
        # mutates the caller's list on purpose; nodes then return it as the new tool_failures
        failures.append(f"{tool}: {last}")
        return None

    # ---- nodes ------------------------------------------------------------
    def intake(self, s: CaseState) -> dict:
        return {"steps": 0, "tool_failures": [], "status": "running",
                "audit": [f"intake claim={s['claim_id']} role={s['role']}"]}

    def supervisor(self, s: CaseState) -> dict:
        steps = s.get("steps", 0) + 1
        if steps > MAX_STEPS:
            # summarize rather than fail: a partial summary still routes to a human via missing evidence/confidence
            return {"steps": steps, "next_step": "summarize", "audit": [f"supervisor: step limit {MAX_STEPS} hit, forcing summarize"]}
        # The planner gets progress flags, not the evidence itself: it decides what to do next, not what it found.
        view = {"evidence_gathered": s.get("evidence_gathered", False), "risk": bool(s.get("risk")),
                "retrieval": bool(s.get("retrieval")), "tool_failures": s.get("tool_failures", []), "steps": steps}
        try:
            decision = self.llm.plan(view)
        except LLMRefusal:
            # a planner refusal shouldn't stop the case; the deterministic plan is always valid
            from ..llm.providers import default_plan
            decision = default_plan(view)
        return {"steps": steps, "next_step": decision.next_step, "audit": [f"supervisor -> {decision.next_step} ({decision.reason})"]}

    def gather_evidence(self, s: CaseState) -> dict:
        failures: list[str] = list(s.get("tool_failures", []))
        cid = s["claim_id"]
        claim = self._call(s, "get_claim", failures, claim_id=cid)
        # the claim is the one hard dependency; everything else below degrades to empty and is noted as missing
        if claim is None:
            return {"tool_failures": failures, "evidence_gathered": True, "status": "failed",
                    "audit": ["gather_evidence: claim unavailable, cannot proceed"]}
        out: dict = {"claim": claim, "tool_failures": failures, "evidence_gathered": True}
        # failures is shared by reference, so later calls' failures land in out["tool_failures"] too
        out["history"] = self._call(s, "get_claim_history", failures, claim_id=cid) or []
        out["payments"] = self._call(s, "get_payment_records", failures, claim_id=cid) or []
        out["provider"] = self._call(s, "get_provider_profile", failures, provider_id=claim["provider_id"]) or {}
        out["vendor"] = self._call(s, "get_vendor_profile", failures, vendor_id=claim["vendor_id"]) if claim.get("vendor_id") else None
        # analysts and auditors aren't allowed this tool, so for them it shows up as a failure and lowers confidence
        out["prior_cases"] = self._call(s, "get_prior_cases", failures, claim_id=cid) or []
        out["audit"] = [f"gather_evidence: history={len(out['history'])} payments={len(out['payments'])} "
                        f"priors={len(out['prior_cases'])} failures={len(failures)}"]
        return out

    def assess_risk(self, s: CaseState) -> dict:
        failures = list(s.get("tool_failures", []))
        # empty low-tier fallback keeps downstream nodes simple; the failure is still recorded for the summary
        risk = self._call(s, "get_risk_signals", failures, claim_id=s["claim_id"]) or {"score": 0, "tier": "low", "signals": []}
        return {"risk": risk, "tool_failures": failures,
                "audit": [f"assess_risk: tier={risk['tier']} score={risk['score']} rules={[x['rule_id'] for x in risk['signals']]}"]}

    def retrieve_policy(self, s: CaseState) -> dict:
        claim, risk = s["claim"], s.get("risk", {})
        claim_type = "vendor" if claim.get("vendor_id") else "professional"
        # Query is built from signal types, not the claim text: no PHI in the query, and it targets the
        # policy section that governs the pattern. No signals -> fall back to the generic review-path section.
        terms = [x["type"].replace("_", " ") for x in risk.get("signals", [])] or ["investigation review path"]
        terms.append(claim.get("procedure_code", ""))   # lets BM25 hit code tables like the panel-pair appendix
        if s.get("prior_cases"):
            terms.append("prior case")
        query = " ".join(terms)
        res = self.index.retrieve(query, RetrievalFilter(role=s["role"], as_of=s.get("as_of", "2025-12-31"), claim_type=claim_type), k=4)
        return {"retrieval": {"query": query, "chunks": [r.to_dict() for r in res.chunks],
                              "rejected": res.rejected, "trace": res.trace},
                "audit": [f"retrieve_policy: q='{query}' -> {res.ids()[:4]} rejected={len(res.rejected)}"]}

    def summarize(self, s: CaseState) -> dict:
        packet = {
            "case_id": s["case_id"], "claim": s.get("claim"), "history": s.get("history", []),
            "payments": s.get("payments", []), "provider": s.get("provider"), "vendor": s.get("vendor"),
            "prior_cases": s.get("prior_cases", []), "risk": s.get("risk", {}),
            "policy": s.get("retrieval", {}).get("chunks", []), "tool_failures": s.get("tool_failures", []),
        }
        # everything in the packet came through the tool boundary or retrieval, so it is already masked
        try:
            summary = self.llm.summarize(packet)
        except LLMRefusal as e:
            # failed status sends the case to human_review in route(), so a refusal never auto-closes
            return {"status": "failed", "audit": [f"summarize: model refusal {e}"]}
        return {"summary": summary.model_dump(), "audit": [f"summarize: level={summary.risk_level} conf={summary.confidence} indicators={len(summary.suspicious_indicators)}"]}

    def verify(self, s: CaseState) -> dict:
        """Faithfulness and leak checks, independent of the model."""
        summary = s.get("summary") or {}
        # Build the set of ids the model could legitimately cite: only what was actually fetched into state.
        known: set[str] = {s["claim_id"]}
        known |= {h["claim_id"] for h in s.get("history", [])}
        known |= {p["payment_id"] for p in s.get("payments", [])}
        known |= {p["case_id"] for p in s.get("prior_cases", [])}
        for sig in s.get("risk", {}).get("signals", []):
            known.add(sig["id"]); known |= set(sig["evidence_ids"])
        if s.get("vendor"):
            # event ids are synthesized the same way rule R5 builds them in risk/rules.py
            known |= {f"EVT-{e['vendor_id']}-{e['date']}" for e in s["vendor"].get("events", [])}
        # citations must point at chunks retrieved for this case, not just any chunk in the index
        chunk_ids = {c["chunk_id"] for c in s.get("retrieval", {}).get("chunks", [])}

        unknown = sorted({e for ind in summary.get("suspicious_indicators", []) for e in ind["evidence_ids"]} - known)
        bad_cites = sorted(set(summary.get("policy_citations", [])) - chunk_ids)
        uncited = [i["description"] for i in summary.get("suspicious_indicators", []) if not i["evidence_ids"]]
        # only the free-text fields; enums and ids can't carry PHI
        text = " ".join([summary.get("recommended_action", ""), summary.get("escalation_reason", "")]
                        + [i["description"] for i in summary.get("suspicious_indicators", [])])
        # check against the raw member roster, not just regexes, to catch a name the model reconstructed
        known_phi = [v for m in self.registry.ds.members.values() for v in (m["name"], m["mrn"], m["phone"], m["email"])]
        leaks = self.deid.vault.contains_raw_phi(text, known_phi)
        # any single failure makes the summary unfaithful; route() then forces human review
        faithful = not unknown and not bad_cites and not uncited and not leaks
        return {"verification": {"faithful": faithful, "unknown_evidence_ids": unknown, "invalid_citations": bad_cites,
                                 "uncited_indicators": uncited, "phi_leaks": leaks},
                "audit": [f"verify: faithful={faithful} unknown={unknown} bad_cites={bad_cites} leaks={leaks}"]}

    def route(self, s: CaseState) -> dict:
        """PI-004 §1 review paths, plus verification outcome."""
        summ, ver = s.get("summary", {}), s.get("verification", {})
        level, conf = summ.get("risk_level", "low"), summ.get("confidence", 0.0)
        # Fail closed: defaults (no summary, conf 0.0, faithful missing -> False) all land in human_review.
        if s.get("status") == "failed":
            route = "human_review"
        elif level == "high" or conf < 0.6 or summ.get("missing_evidence") or not ver.get("faithful", False):
            route = "human_review"
        elif level == "medium":
            route = "analyst_queue"
        else:
            route = "auto_close"
        return {"route": route, "audit": [f"route: {route} (level={level} conf={conf} faithful={ver.get('faithful')})"]}

    def human_review(self, s: CaseState) -> dict:
        packet = {"case_id": s["case_id"], "claim_id": s["claim_id"], "summary": s.get("summary"),
                  "verification": s.get("verification"), "reason": s.get("summary", {}).get("escalation_reason")}
        # Graph pauses here; resumed with the reviewer's decision. On resume LangGraph re-runs this node
        # from the top and interrupt() returns the resume value, so nothing above it may have side effects.
        decision = interrupt(packet)
        return {"human_decision": decision, "audit": [f"human_review: {decision.get('decision')} by {decision.get('reviewer')}"]}

    def finalize(self, s: CaseState) -> dict:
        d = s.get("human_decision")
        # a failed run stays failed even if a reviewer approved it; the record should show what broke
        if s.get("status") == "failed":
            status = "failed"
        elif d is None:
            status = {"auto_close": "closed_auto", "analyst_queue": "queued_analyst"}.get(s.get("route", ""), "done")
        else:
            # reject = reviewer says the flag was wrong; FeedbackStore counts it as a false positive
            status = {"approve": "closed_confirmed", "reject": "closed_false_positive", "escalate": "escalated_siu"}.get(d.get("decision"), "done")
        return {"status": status, "audit": [f"finalize: {status}"]}
