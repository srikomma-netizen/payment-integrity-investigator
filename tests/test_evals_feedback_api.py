import pytest
from fastapi.testclient import TestClient

from api.main import app
from investigator.evals.run_evals import load_golden, run_suite, summarize
from investigator.feedback.store import FeedbackStore
from investigator.graph.build import Investigator
from investigator.llm.providers import FakeInvestigatorLLM


def test_golden_suite_passes_offline():
    results = run_suite(provider="fake")
    failures = [(r.id, r.notes) for r in results if not r.passed]
    assert not failures, failures
    s = summarize(results)
    assert s["passed"] == len(load_golden()) and s["retrieval"]["authorized_rate"].startswith(str(len(results)))


def test_feedback_store_attribution_and_golden_candidates(tmp_path):
    inv = Investigator(llm=FakeInvestigatorLLM())
    fb = FeedbackStore(tmp_path / "fb.json")
    st = inv.start("CLM-1012")
    done = inv.resume(st["case_id"], decision="reject", reviewer="inv_1", rationale="records support level 5")
    rec = fb.record(done, decision="reject", reviewer="inv_1", rationale="records support level 5")
    assert rec.disposition == "false_positive" and set(rec.rules_fired) == {"R2", "R4"}
    assert fb.false_positive_rate_by_rule()["R2"]["rate"] == 1.0
    assert fb.attribution_summary() == {"upstream_rule": 1}
    cand = fb.golden_candidates()[0]
    assert cand["expect"]["risk_level"] == "low" and "PI-002:2" in cand["expect"]["policy_sections"]
    assert FeedbackStore(tmp_path / "fb.json").records[0].case_id == st["case_id"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_round_trip(client):
    assert client.get("/health").json()["ok"]
    assert len(client.get("/tools").json()) == 7
    started = client.post("/cases", json={"claim_id": "CLM-1019", "role": "investigator"}).json()
    assert started["status"] == "awaiting_human_review" and started["pending_review"]
    cid = started["case_id"]
    analyst = client.get(f"/cases/{cid}", params={"role": "analyst"}).json()
    lead = client.get(f"/cases/{cid}", params={"role": "siu_lead"}).json()
    assert analyst["claim"]["member"]["name"].startswith("[PATIENT_") and not lead["claim"]["member"]["name"].startswith("[")
    done = client.post(f"/cases/{cid}/decision", json={"decision": "escalate", "reviewer": "lead_2", "rationale": "wire already sent"}).json()
    assert done["status"] == "escalated_siu"
    assert client.post(f"/cases/{cid}/decision", json={"decision": "approve", "reviewer": "x"}).status_code == 409
    fb = client.get("/feedback/summary").json()
    assert fb["records"] == 1 and "R5" in fb["false_positive_by_rule"]


def test_api_validation(client):
    assert client.post("/cases", json={"claim_id": "nope"}).status_code == 422
    assert client.post("/cases", json={"claim_id": "CLM-9999"}).status_code == 404
    assert client.get("/cases/CASE-NOPE").status_code == 404
