"""Rule-based risk flags for claims."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..synthetic import UNBUNDLE_PAIRS, Dataset


@dataclass
class RiskSignal:
    rule_id: str
    signal_type: str
    severity: str            # low | medium | high
    description: str
    evidence_ids: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def id(self) -> str:
        # not unique if R3/R5 fire twice on one claim
        return f"SIG-{self.rule_id}"


@dataclass
class RiskAssessment:
    claim_id: str
    score: float
    tier: str                # low | medium | high
    signals: list[RiskSignal]
    flagged: bool

    def evidence_ids(self) -> list[str]:
        out: list[str] = []
        for s in self.signals:
            out.extend(s.evidence_ids)
        return sorted(set(out))


# unused for now
SEVERITY_WEIGHT = {"low": 0.2, "medium": 0.45, "high": 0.8}


def assess(ds: Dataset, claim_id: str) -> RiskAssessment:
    """Run all rules on one claim and score it."""
    c = ds.claims[claim_id]
    signals: list[RiskSignal] = []

    # R1 duplicate payment
    pays = [p for p in ds.payments if p["claim_id"] == claim_id]
    if len(pays) > 1:
        signals.append(RiskSignal("R1", "duplicate_payment", "high",
                                  f"{len(pays)} payments recorded for one claim totalling {sum(p['amount'] for p in pays):.2f} USD",
                                  [p["payment_id"] for p in pays], 0.8))

    # R2 amount outlier vs peers, median/MAD z-score
    peers = [x["billed_amount"] for x in ds.claims.values()
             if x["procedure_code"] == c["procedure_code"] and x["claim_id"] != claim_id and not x.get("vendor_id")]
    if len(peers) >= 3:
        med = statistics.median(peers)
        mad = statistics.median(abs(p - med) for p in peers) or 1.0   # MAD=0 when peers identical
        z = (c["billed_amount"] - med) / (1.4826 * mad)   # 1.4826: MAD -> stdev
        # one-sided, underbilling doesn't matter here
        if z > 4:
            signals.append(RiskSignal("R2", "amount_outlier", "high" if z > 8 else "medium",
                                      f"Billed {c['billed_amount']:.2f} USD vs peer median {med:.2f} USD for {c['procedure_code']} (robust z={z:.1f})",
                                      [claim_id], min(0.9, 0.3 + z / 20)))

    # R3 unbundling, same member/provider/date
    same_day = [x for x in ds.claims.values() if x["member_id"] == c["member_id"] and x["provider_id"] == c["provider_id"]
                and x["service_date"] == c["service_date"] and x["claim_id"] != claim_id]
    for other in same_day:
        pair = tuple(sorted((c["procedure_code"], other["procedure_code"])))   # keys are sorted
        if pair in UNBUNDLE_PAIRS:
            signals.append(RiskSignal("R3", "unbundling", "medium",
                                      f"Codes {pair[0]} and {pair[1]} billed separately; bundled code {UNBUNDLE_PAIRS[pair]} expected",
                                      [claim_id, other["claim_id"]], 0.55))

    # R4 prior confirmed case for provider
    priors = [p for p in ds.prior_cases if p.get("provider_id") == c["provider_id"] and p["outcome"].startswith("confirmed")]
    if priors:
        signals.append(RiskSignal("R4", "prior_confirmed_case", "medium",
                                  f"Provider has {len(priors)} prior confirmed case(s)", [p["case_id"] for p in priors], 0.4))

    # R5 unverified vendor bank change, 30 day window
    vid = c.get("vendor_id")
    if vid:
        for ev in ds.vendor_events:
            if ev["vendor_id"] == vid and ev["event"] == "bank_details_changed" and not ev["verified_callback"] \
                    and ev["date"] <= c["service_date"] and _days_between(ev["date"], c["service_date"]) <= 30:
                signals.append(RiskSignal("R5", "unverified_bank_change", "high",
                                          f"Bank details changed via {ev['channel']} on {ev['date']} without call-back, {_days_between(ev['date'], c['service_date'])} days before a {c['billed_amount']:.2f} USD invoice",
                                          [f"EVT-{vid}-{ev['date']}", claim_id], 0.85))

    # noisy-OR
    score = 1 - _prod(1 - s.score for s in signals) if signals else 0.0
    tier = "high" if score >= 0.7 else "medium" if score >= 0.35 else "low"
    return RiskAssessment(claim_id, round(score, 3), tier, signals, flagged=score >= 0.35)


def flagged_claims(ds: Dataset) -> list[RiskAssessment]:
    return [a for a in (assess(ds, cid) for cid in ds.claims) if a.flagged]


def _days_between(a: str, b: str) -> int:
    from datetime import date
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _prod(xs) -> float:
    p = 1.0
    for x in xs:
        p *= x
    return p
