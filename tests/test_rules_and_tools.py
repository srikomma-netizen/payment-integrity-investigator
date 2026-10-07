import pytest

from investigator.risk.rules import assess, flagged_claims
from investigator.security.phi import Deidentifier, Vault
from investigator.synthetic import build_dataset
from investigator.tools.registry import ToolError, ToolRegistry


@pytest.fixture(scope="module")
def ds():
    return build_dataset()


@pytest.fixture
def registry(ds):
    return ToolRegistry(ds, Deidentifier(Vault(), known_names=[m["name"] for m in ds.members.values()]))


def test_planted_patterns_fire_expected_rules(ds):
    assert {s.rule_id for s in assess(ds, "CLM-1007").signals} >= {"R1"}
    assert {s.rule_id for s in assess(ds, "CLM-1012").signals} == {"R2", "R4"}
    assert {s.rule_id for s in assess(ds, "CLM-1015").signals} == {"R3"}
    assert {s.rule_id for s in assess(ds, "CLM-1019").signals} == {"R5"}
    assert assess(ds, "CLM-1003").signals == [] and not assess(ds, "CLM-1003").flagged


def test_signals_carry_evidence_ids(ds):
    dup = next(s for s in assess(ds, "CLM-1007").signals if s.rule_id == "R1")
    assert set(dup.evidence_ids) == {"PAY-1007A", "PAY-1007B"}
    assert "CLM-1019" in flagged_claims(ds)[-1].evidence_ids() or any(a.claim_id == "CLM-1019" for a in flagged_claims(ds))


def test_tools_mask_phi_at_boundary(registry, ds):
    claim = registry.call("get_claim", "analyst", claim_id="CLM-1012")
    member = ds.members[ds.claims["CLM-1012"]["member_id"]]
    assert member["name"] not in str(claim) and member["mrn"] not in str(claim)
    assert claim["member"]["name"].startswith("[PATIENT_")
    assert "Maya Okafor" not in claim["notes"]


def test_tool_permissions_and_restricted_fields(registry):
    with pytest.raises(ToolError, match="may not call"):
        registry.call("get_prior_cases", "analyst", claim_id="CLM-1012")
    vendor = registry.call("get_vendor_profile", "investigator", vendor_id="VND-702")
    assert "bank_account_last4" not in vendor and vendor["events"]


def test_tool_outage_is_typed_and_audited(registry):
    registry.fail_next("get_payment_records", 1)
    with pytest.raises(ToolError, match="unavailable"):
        registry.call("get_payment_records", "analyst", claim_id="CLM-1007")
    assert registry.call("get_payment_records", "analyst", claim_id="CLM-1007")
    assert [a.ok for a in registry.audit] == [False, True]


def test_specs_are_mcp_shaped(registry):
    specs = registry.specs()
    assert {s["name"] for s in specs} >= {"get_claim", "get_risk_signals", "get_vendor_profile"}
    assert all(s["input_schema"]["type"] == "object" for s in specs)
