"""MCP-shaped tool registry: the only way agents reach case data.

Properties:
  * agents never touch the database; they call named tools with JSON schemas
  * every tool masks PHI at the boundary, so the model only sees tokens
  * per-role permissions on each tool
  * every call is audited (who, what, args, outcome)
  * failures are typed so the graph can retry or degrade, not crash

`mcp_server.py` exposes this same registry over the Model Context Protocol
for out-of-process agents; the graph calls it in-process for tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..security.phi import Deidentifier
from ..synthetic import Dataset
from ..risk.rules import assess


class ToolError(Exception):
    """Transient or permission failure surfaced to the caller as data."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    roles: tuple[str, ...]
    fn: Callable[..., Any]


@dataclass
class ToolAudit:
    role: str
    tool: str
    args: dict
    ok: bool
    detail: str = ""


class ToolRegistry:
    def __init__(self, ds: Dataset, deid: Deidentifier):
        self.ds, self.deid = ds, deid
        self.audit: list[ToolAudit] = []
        self._fail_next: dict[str, int] = {}     # test hook: simulate outages
        self._tools: dict[str, ToolSpec] = {}
        self._register_all()

    # ---- framework -------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def specs(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in self._tools.values()]

    def fail_next(self, tool: str, times: int = 1) -> None:
        self._fail_next[tool] = times

    def call(self, name: str, role: str, **args) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"unknown tool {name}")
        if role not in spec.roles:
            self.audit.append(ToolAudit(role, name, args, False, "permission denied"))
            raise ToolError(f"role '{role}' may not call {name}")
        if self._fail_next.get(name, 0) > 0:
            self._fail_next[name] -= 1
            self.audit.append(ToolAudit(role, name, args, False, "simulated backend outage"))
            raise ToolError(f"{name}: backend unavailable")
        result = spec.fn(**args)
        self.audit.append(ToolAudit(role, name, args, True))
        return result

    # ---- tools -----------------------------------------------------------
    def _register_all(self) -> None:
        ds, deid = self.ds, self.deid
        all_roles = ("analyst", "investigator", "siu_lead", "auditor")

        def get_claim(claim_id: str) -> dict:
            c = ds.claims.get(claim_id)
            if not c:
                raise ToolError(f"claim {claim_id} not found")
            m = ds.members[c["member_id"]]
            masked = deid.mask_record({**c, "member": {k: m[k] for k in ("name", "dob", "mrn", "plan")}})
            masked["member_token"] = masked["member"]["name"]
            return masked

        def get_claim_history(claim_id: str, limit: int = 10) -> list[dict]:
            c = ds.claims[claim_id]
            hist = ds.claims_for_member(c["member_id"])[-limit:]
            return [deid.mask_record({k: x[k] for k in ("claim_id", "provider_id", "service_date", "procedure_code", "billed_amount", "status")}) for x in hist]

        def get_payment_records(claim_id: str) -> list[dict]:
            return [p for p in ds.payments if p["claim_id"] == claim_id]

        def get_provider_profile(provider_id: str) -> dict:
            p = ds.providers.get(provider_id)
            if not p:
                raise ToolError(f"provider {provider_id} not found")
            return deid.mask_record({**p, "claim_count": len(ds.claims_for_provider(provider_id))})

        def get_vendor_profile(vendor_id: str) -> dict:
            v = ds.vendors.get(vendor_id)
            if not v:
                raise ToolError(f"vendor {vendor_id} not found")
            safe = {k: v[k] for k in v if k != "bank_account_last4"}   # restricted, never returned
            safe["events"] = [e for e in ds.vendor_events if e["vendor_id"] == vendor_id]
            return safe

        def get_prior_cases(claim_id: str) -> list[dict]:
            c = ds.claims[claim_id]
            return [deid.mask_record(p) for p in ds.prior_cases
                    if p.get("provider_id") == c["provider_id"] or p.get("member_id") == c["member_id"]
                    or (c.get("vendor_id") and p.get("vendor_id") == c.get("vendor_id"))]

        def get_risk_signals(claim_id: str) -> dict:
            a = assess(ds, claim_id)
            return {"score": a.score, "tier": a.tier, "flagged": a.flagged,
                    "signals": [{"id": s.id, "rule_id": s.rule_id, "type": s.signal_type, "severity": s.severity,
                                 "description": deid.mask_text(s.description), "evidence_ids": s.evidence_ids} for s in a.signals]}

        cid_schema = {"type": "object", "properties": {"claim_id": {"type": "string"}}, "required": ["claim_id"], "additionalProperties": False}
        self.register(ToolSpec("get_claim", "Claim header with masked member demographics.", cid_schema, all_roles, get_claim))
        self.register(ToolSpec("get_claim_history", "Recent claims for the same member (masked).",
                               {"type": "object", "properties": {"claim_id": {"type": "string"}, "limit": {"type": "integer"}},
                                "required": ["claim_id"], "additionalProperties": False}, all_roles, get_claim_history))
        self.register(ToolSpec("get_payment_records", "Payments issued against a claim.", cid_schema, all_roles, get_payment_records))
        self.register(ToolSpec("get_provider_profile", "Provider master record (NPI masked).",
                               {"type": "object", "properties": {"provider_id": {"type": "string"}}, "required": ["provider_id"], "additionalProperties": False},
                               all_roles, get_provider_profile))
        self.register(ToolSpec("get_vendor_profile", "Vendor master record and change events. Bank details are never returned.",
                               {"type": "object", "properties": {"vendor_id": {"type": "string"}}, "required": ["vendor_id"], "additionalProperties": False},
                               ("investigator", "siu_lead", "auditor"), get_vendor_profile))
        self.register(ToolSpec("get_prior_cases", "Prior investigation cases linked to the member, provider, or vendor.", cid_schema,
                               ("investigator", "siu_lead"), get_prior_cases))
        self.register(ToolSpec("get_risk_signals", "Upstream risk signals and score for a claim.", cid_schema, all_roles, get_risk_signals))
