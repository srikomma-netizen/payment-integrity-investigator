"""MCP server for the tool registry.

Run: python -m investigator.tools.mcp_server
"""
from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from ..security.phi import Deidentifier, Vault
from ..synthetic import build_dataset
from .registry import ToolError, ToolRegistry


def build_server() -> MCPServer:
    ds = build_dataset()
    registry = ToolRegistry(ds, Deidentifier(Vault(), known_names=[m["name"] for m in ds.members.values()]))
    server = MCPServer("payment-integrity-tools", instructions="Masked, role-scoped access to claims, payments, providers, vendors, prior cases and risk signals.")

    def _wrap(tool_name: str):
        def call(role: str, **kwargs) -> str:
            try:
                return json.dumps(registry.call(tool_name, role, **kwargs))
            except ToolError as e:
                # error as a result, not a protocol error
                return json.dumps({"error": str(e)})
        return call

    # schema comes from the signature, so no loop over specs()

    @server.tool(name="get_claim", description="Claim header with masked member demographics.")
    def get_claim(claim_id: str, role: str = "analyst") -> str:
        return _wrap("get_claim")(role, claim_id=claim_id)

    @server.tool(name="get_claim_history", description="Recent claims for the same member (masked).")
    def get_claim_history(claim_id: str, limit: int = 10, role: str = "analyst") -> str:
        return _wrap("get_claim_history")(role, claim_id=claim_id, limit=limit)

    @server.tool(name="get_payment_records", description="Payments issued against a claim.")
    def get_payment_records(claim_id: str, role: str = "analyst") -> str:
        return _wrap("get_payment_records")(role, claim_id=claim_id)

    @server.tool(name="get_provider_profile", description="Provider master record (NPI masked).")
    def get_provider_profile(provider_id: str, role: str = "analyst") -> str:
        return _wrap("get_provider_profile")(role, provider_id=provider_id)

    @server.tool(name="get_vendor_profile", description="Vendor record and change events; bank details never returned.")
    def get_vendor_profile(vendor_id: str, role: str = "investigator") -> str:
        return _wrap("get_vendor_profile")(role, vendor_id=vendor_id)

    @server.tool(name="get_prior_cases", description="Prior investigation cases linked to the claim.")
    def get_prior_cases(claim_id: str, role: str = "investigator") -> str:
        return _wrap("get_prior_cases")(role, claim_id=claim_id)

    @server.tool(name="get_risk_signals", description="Upstream risk signals and score for a claim.")
    def get_risk_signals(claim_id: str, role: str = "analyst") -> str:
        return _wrap("get_risk_signals")(role, claim_id=claim_id)

    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
