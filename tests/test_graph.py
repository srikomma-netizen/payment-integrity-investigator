import pytest

from investigator.graph.build import Investigator
from investigator.llm.providers import FakeInvestigatorLLM
from investigator.llm.schemas import Indicator, InvestigationSummary


@pytest.fixture
def inv():
    return Investigator(llm=FakeInvestigatorLLM())


def test_clean_claim_auto_closes(inv):
    st = inv.start("CLM-1003")
    assert st["status"] == "closed_auto" and st["route"] == "auto_close"
    assert st["verification"]["faithful"] and st["pending_review"] is None


def test_high_risk_pauses_for_human_then_resumes(inv):
    st = inv.start("CLM-1012")
    assert st["status"] == "awaiting_human_review" and st["pending_review"]["summary"]["risk_level"] == "high"
    done = inv.resume(st["case_id"], decision="reject", reviewer="inv_9", rationale="documentation supports level 5")
    assert done["status"] == "closed_false_positive" and done["human_decision"]["reviewer"] == "inv_9"
    with pytest.raises(ValueError):
        inv.resume(st["case_id"], decision="approve", reviewer="x")


def test_prior_false_positive_lowers_level(inv):
    st = inv.start("CLM-1007")
    assert st["risk"]["tier"] == "high" and st["summary"]["risk_level"] == "medium" and st["route"] == "analyst_queue"


def test_tool_outage_degrades_to_human_review(inv):
    inv.registry.fail_next("get_payment_records", 2)
    st = inv.start("CLM-1007")
    assert st["tool_failures"] and st["summary"]["missing_evidence"]
    assert st["route"] == "human_review" and st["status"] == "awaiting_human_review"


def test_analyst_role_cannot_see_identity_but_investigator_can(inv):
    st = inv.start("CLM-1019")
    a = inv.view(st["case_id"], "analyst")["claim"]["member"]["name"]
    i = inv.view(st["case_id"], "investigator")["claim"]["member"]["name"]
    assert a.startswith("[PATIENT_") and not i.startswith("[")
    assert "bank_account_last4" not in str(st.get("vendor"))


def test_verify_catches_hallucinated_evidence_and_citations():
    class Hallucinating(FakeInvestigatorLLM):
        def summarize(self, packet):
            return InvestigationSummary(
                risk_level="low", confidence=0.95, recommended_action="Close.",
                suspicious_indicators=[Indicator(description="made up", evidence_ids=["PAY-9999Z"], severity="low")],
                policy_citations=["PI-009:v1:1:0"])
    inv = Investigator(llm=Hallucinating())
    st = inv.start("CLM-1003")
    v = st["verification"]
    assert not v["faithful"] and v["unknown_evidence_ids"] == ["PAY-9999Z"] and v["invalid_citations"] == ["PI-009:v1:1:0"]
    assert st["route"] == "human_review"


def test_verify_catches_phi_leak():
    class Leaky(FakeInvestigatorLLM):
        def summarize(self, packet):
            s = super().summarize(packet)
            s.recommended_action = "Contact Maya Okafor at 312-555-0199."
            return s
    inv = Investigator(llm=Leaky())
    st = inv.start("CLM-1015")
    assert "pattern:PHONE" in st["verification"]["phi_leaks"] and st["route"] == "human_review"


def test_audit_trail_survives_checkpoint_resume(inv):
    st = inv.start("CLM-1019")
    before = len(st["audit"])
    done = inv.resume(st["case_id"], decision="escalate", reviewer="lead")
    assert len(done["audit"]) == before + 2 and done["status"] == "escalated_siu"
