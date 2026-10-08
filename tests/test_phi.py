from investigator.security.phi import Deidentifier, Vault


def test_structured_fields_tokenized_and_deterministic():
    d = Deidentifier(Vault())
    a = d.mask_record({"name": "Maya Okafor", "mrn": "MRN1234567", "plan": "PPO"})
    b = d.mask_record({"name": "Maya Okafor", "mrn": "MRN1234567", "plan": "PPO"})
    assert a == b and a["name"].startswith("[PATIENT_") and a["mrn"].startswith("[MRN_") and a["plan"] == "PPO"


def test_free_text_patterns_masked():
    d = Deidentifier(Vault())
    out = d.mask_text("Patient Maya Okafor DOB 1972-03-09, ssn 123-45-6789, call 312-555-0199, maya@example.net")
    assert "Maya Okafor" not in out and "123-45-6789" not in out and "312-555-0199" not in out and "@example.net" not in out
    assert "[PATIENT_" in out and "[SSN_" in out and "[PHONE_" in out and "[EMAIL_" in out and "[DOB_" in out


def test_rehydrate_respects_role_clearance_and_audits():
    v = Vault()
    d = Deidentifier(v)
    rec = d.mask_record({"name": "Luis Costa", "ssn_last4": "4321"})
    assert v.rehydrate(rec, "analyst") == rec
    inv = v.rehydrate(rec, "investigator")
    # investigators get identity but not SSN, so that token must come back unchanged
    assert inv["name"] == "Luis Costa" and inv["ssn_last4"] == rec["ssn_last4"]
    lead = v.rehydrate(rec, "siu_lead")
    assert lead["ssn_last4"] == "4321"
    actions = [a.action for a in v.audit]
    assert "resolve_denied" in actions and "resolve" in actions


def test_leak_check():
    v = Vault()
    # known values are reported before pattern hits
    assert v.contains_raw_phi("call 312-555-0101 for Maya", ["Maya"]) == ["Maya", "pattern:PHONE"]
    assert v.contains_raw_phi("token [PATIENT_abcdef] only", ["Maya"]) == []
