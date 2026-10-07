"""Synthetic healthcare-payments dataset.

Every name, identifier and amount is generated. The shapes are realistic
(claims, payments, vendors, prior cases, member demographics) so the PHI
controls, upstream risk rules, and investigation workflow have something
real to chew on. Several cases are planted with known patterns so evals
have ground truth:

  CLM-1007  duplicate payment of the same claim
  CLM-1012  procedure amount far above the provider's baseline (upcoding)
  CLM-1015  unbundled procedure pair billed separately
  CLM-1019  vendor changed bank details days before a large invoice
  CLM-1003  clean claim (control)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

SEED = 11

PROCEDURES = {
    "99213": ("Office visit, established patient", 110.0),
    "99214": ("Office visit, moderate complexity", 165.0),
    "99215": ("Office visit, high complexity", 230.0),
    "80053": ("Comprehensive metabolic panel", 48.0),
    "85025": ("Complete blood count", 32.0),
    "71046": ("Chest X-ray, two views", 95.0),
    "93000": ("Electrocardiogram with interpretation", 70.0),
    "36415": ("Venipuncture", 12.0),
}
UNBUNDLE_PAIRS = {("80053", "85025"): "80050"}  # components of a general health panel

FIRST = ["Maya", "Jordan", "Priya", "Elena", "Noah", "Amara", "Luis", "Hana", "Omar", "Ivy", "Theo", "Sana"]
LAST = ["Okafor", "Lindqvist", "Ramirez", "Nakamura", "Haddad", "Fischer", "Mensah", "Costa", "Byrne", "Soto"]
STREETS = ["Maple Ave", "Oak St", "Cedar Ln", "Birch Rd", "Elm Ct"]


@dataclass
class Dataset:
    members: dict[str, dict] = field(default_factory=dict)
    providers: dict[str, dict] = field(default_factory=dict)
    vendors: dict[str, dict] = field(default_factory=dict)
    claims: dict[str, dict] = field(default_factory=dict)
    payments: list[dict] = field(default_factory=list)
    prior_cases: list[dict] = field(default_factory=list)
    vendor_events: list[dict] = field(default_factory=list)

    def claims_for_member(self, member_id: str) -> list[dict]:
        return sorted((c for c in self.claims.values() if c["member_id"] == member_id), key=lambda c: c["service_date"])

    def claims_for_provider(self, provider_id: str) -> list[dict]:
        return [c for c in self.claims.values() if c["provider_id"] == provider_id]


def _member(rng: random.Random, i: int) -> dict:
    fn, ln = rng.choice(FIRST), rng.choice(LAST)
    return {
        "member_id": f"MBR-{2000 + i}",
        "name": f"{fn} {ln}",
        "dob": f"19{rng.randint(45, 99):02d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "mrn": f"MRN{rng.randint(1_000_000, 9_999_999)}",
        "ssn_last4": f"{rng.randint(1000, 9999)}",
        "phone": f"312-555-{rng.randint(1000, 9999)}",
        "email": f"{fn.lower()}.{ln.lower()}@example.net",
        "address": f"{rng.randint(10, 999)} {rng.choice(STREETS)}, Naperville, IL",
        "plan": rng.choice(["PPO-Gold", "HMO-Silver", "PPO-Bronze"]),
    }


def build_dataset(seed: int = SEED) -> Dataset:
    rng = random.Random(seed)
    ds = Dataset()
    for i in range(12):
        m = _member(rng, i)
        ds.members[m["member_id"]] = m
    for i, (name, spec) in enumerate([
        ("Lakeside Family Clinic", "Primary Care"), ("Northshore Diagnostics", "Laboratory"),
        ("Prairie Imaging Center", "Radiology"), ("Riverbend Cardiology", "Cardiology"),
        ("Oakline Medical Group", "Primary Care"),
    ]):
        ds.providers[f"PRV-{500 + i}"] = {"provider_id": f"PRV-{500 + i}", "name": name, "specialty": spec,
                                           "npi": f"{rng.randint(10**9, 10**10 - 1)}", "state": "IL"}
    for i, (name, cat) in enumerate([
        ("MedSupply Partners", "Medical Supplies"), ("ClearPath Billing Services", "Billing"),
        ("Summit Facilities", "Facilities"), ("Aster Pharma Distribution", "Pharmaceuticals"),
    ]):
        vid = f"VND-{700 + i}"
        ds.vendors[vid] = {"vendor_id": vid, "name": name, "category": cat, "country": "US",
                           "bank_account_last4": f"{rng.randint(1000, 9999)}", "status": "ACTIVE",
                           "onboarded": "2023-0%d-15" % rng.randint(1, 9)}

    # ---- claims: a baseline of normal activity -------------------------
    claim_no = 1000
    members = list(ds.members)
    providers = list(ds.providers)
    for _ in range(26):
        claim_no += 1
        code = rng.choice(list(PROCEDURES))
        desc, base = PROCEDURES[code]
        pid = rng.choice(providers)
        cid = f"CLM-{claim_no}"
        ds.claims[cid] = {
            "claim_id": cid, "member_id": rng.choice(members), "provider_id": pid,
            "service_date": f"2025-{rng.randint(1, 8):02d}-{rng.randint(1, 28):02d}",
            "procedure_code": code, "procedure_desc": desc,
            "billed_amount": round(base * rng.uniform(0.9, 1.15), 2),
            "diagnosis": rng.choice(["E11.9", "I10", "J06.9", "M54.5", "Z00.00"]),
            "status": "paid", "notes": "",
        }
    # ---- planted patterns ------------------------------------------------
    ds.claims["CLM-1003"].update({"notes": "Routine follow-up, no issues."})
    # duplicate payment
    ds.claims["CLM-1007"].update({"notes": "Resubmitted after portal timeout."})
    # upcoding: 99215 billed at 3x baseline by a primary-care provider
    ds.claims["CLM-1012"].update({"procedure_code": "99215", "procedure_desc": PROCEDURES["99215"][0],
                                  "billed_amount": 690.0, "provider_id": "PRV-500",
                                  "notes": "Patient Maya Okafor DOB 1972-03-09 seen for extended consult."})
    # unbundling: 80053 and 85025 same member same day same provider
    ds.claims["CLM-1015"].update({"procedure_code": "80053", "procedure_desc": PROCEDURES["80053"][0],
                                  "billed_amount": 48.0, "provider_id": "PRV-501", "service_date": "2025-06-11"})
    claim_no += 1
    ds.claims["CLM-1027"] = {**ds.claims["CLM-1015"], "claim_id": "CLM-1027", "procedure_code": "85025",
                             "procedure_desc": PROCEDURES["85025"][0], "billed_amount": 32.0}
    # vendor bank change then spike (vendor-side case tied to a facilities claim line)
    ds.claims["CLM-1019"].update({"vendor_id": "VND-702", "procedure_code": "FAC-INV", "procedure_desc": "Facilities invoice",
                                  "billed_amount": 48_900.0, "service_date": "2025-07-20",
                                  "notes": "Invoice from Summit Facilities, contact ssn 123-45-6789 on file? call 312-555-0199"})
    ds.vendor_events.append({"vendor_id": "VND-702", "event": "bank_details_changed", "date": "2025-07-14",
                             "channel": "email", "verified_callback": False})
    ds.vendor_events.append({"vendor_id": "VND-700", "event": "bank_details_changed", "date": "2024-02-02",
                             "channel": "portal", "verified_callback": True})

    # ---- payments (one per claim, duplicate for CLM-1007) -----------------
    for cid, c in ds.claims.items():
        if c.get("vendor_id"):
            continue
        ds.payments.append({"payment_id": f"PAY-{cid[-4:]}A", "claim_id": cid, "amount": c["billed_amount"],
                            "paid_date": c["service_date"], "payee": c["provider_id"], "method": "ACH"})
    dup = ds.claims["CLM-1007"]
    ds.payments.append({"payment_id": "PAY-1007B", "claim_id": "CLM-1007", "amount": dup["billed_amount"],
                        "paid_date": dup["service_date"], "payee": dup["provider_id"], "method": "ACH"})
    ds.payments.append({"payment_id": "PAY-1019A", "claim_id": "CLM-1019", "amount": 48_900.0,
                        "paid_date": "2025-07-22", "payee": "VND-702", "method": "WIRE"})

    # ---- prior cases ------------------------------------------------------
    ds.prior_cases = [
        {"case_id": "PC-0441", "provider_id": "PRV-500", "member_id": None, "opened": "2024-11-02",
         "outcome": "confirmed_upcoding", "summary": "Provider billed 99215 for routine visits; recovery of 4,120 USD."},
        {"case_id": "PC-0467", "provider_id": None, "member_id": ds.claims["CLM-1007"]["member_id"], "opened": "2025-02-10",
         "outcome": "false_positive", "summary": "Duplicate flag caused by portal resubmission; single payment confirmed."},
        {"case_id": "PC-0470", "provider_id": None, "member_id": None, "vendor_id": "VND-702", "opened": "2025-03-30",
         "outcome": "closed_no_action", "summary": "Invoice amount query resolved with PO."},
    ]
    return ds
