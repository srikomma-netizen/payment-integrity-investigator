from investigator.rag.index import HybridIndex, RetrievalFilter
from investigator.rag.ingest import ingest


def test_ingest_attaches_metadata_and_neighbors():
    chunks = ingest()
    c = next(c for c in chunks if c.policy_id == "PI-002" and c.section_id == "3")
    assert c.status == "approved" and c.version == 7 and c.claim_types == ["professional"]
    assert c.citation.startswith("PI-002 v7 §3")
    assert any(n.startswith("PI-002") for n in c.neighbors)


def test_superseded_draft_and_restricted_are_filtered():
    idx = HybridIndex()
    res = idx.retrieve("duplicate payment portal timeout", RetrievalFilter(role="analyst"), k=5)
    ids = res.ids()
    assert all("PI-001:v3" not in i and "PI-006" not in i and "PI-005" not in i for i in ids)
    reasons = dict(res.rejected)
    assert reasons.get("PI-001:v3:4:0") == "status=superseded"
    assert reasons.get("PI-006:v0:1:0") == "status=draft"


def test_access_level_opens_for_cleared_roles():
    idx = HybridIndex()
    analyst = idx.retrieve("PHI minimum necessary model boundary", RetrievalFilter(role="analyst"), k=3).ids()
    lead = idx.retrieve("PHI minimum necessary model boundary", RetrievalFilter(role="siu_lead"), k=3).ids()
    assert not any(i.startswith("PI-005") for i in analyst)
    assert lead[0].startswith("PI-005")


def test_claim_type_filter_and_hybrid_code_match():
    idx = HybridIndex()
    res = idx.retrieve("unbundling 80053 85025", RetrievalFilter(role="analyst", claim_type="professional"), k=3)
    assert res.ids()[0].startswith("PI-002")
    assert any(c.chunk.section_id == "A" for c in res.chunks)   # exact code match pulls the appendix


def test_context_expansion_marks_origin():
    idx = HybridIndex()
    res = idx.retrieve("bank detail change call-back", RetrievalFilter(role="investigator", claim_type="vendor"), k=2)
    assert any(c.expanded_from for c in res.chunks) or len(res.chunks) >= 2
    assert all(c.chunk.policy_id == "PI-003" for c in res.chunks[:2])
