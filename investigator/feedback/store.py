"""Reviewer feedback store and offline analysis."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DISPOSITION_BY_DECISION = {"approve": "confirmed", "reject": "false_positive", "escalate": "escalated"}


@dataclass
class FeedbackRecord:
    case_id: str
    claim_id: str
    reviewer: str
    decision: str                      # approve | reject | escalate
    disposition: str                   # confirmed | false_positive | escalated
    rationale: str
    recommended_level: str
    recommended_action: str
    rules_fired: list[str]
    retrieved_chunks: list[str]
    verification: dict
    confidence: float
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class FeedbackStore:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.records: list[FeedbackRecord] = []
        if path and path.exists():
            self.records = [FeedbackRecord(**r) for r in json.loads(path.read_text(encoding="utf-8"))]

    def record(self, state: dict[str, Any], *, decision: str, reviewer: str, rationale: str = "") -> FeedbackRecord:
        summ = state.get("summary") or {}
        rec = FeedbackRecord(
            case_id=state["case_id"], claim_id=state["claim_id"], reviewer=reviewer, decision=decision,
            disposition=DISPOSITION_BY_DECISION.get(decision, decision), rationale=rationale,
            recommended_level=summ.get("risk_level", "n/a"), recommended_action=summ.get("recommended_action", ""),
            rules_fired=[s["rule_id"] for s in state.get("risk", {}).get("signals", [])],
            retrieved_chunks=[c["chunk_id"] for c in state.get("retrieval", {}).get("chunks", [])],
            verification=state.get("verification", {}), confidence=summ.get("confidence", 0.0),
        )
        self.records.append(rec)
        if self.path:
            # TODO: rewrites whole file each time, switch to jsonl
            self.path.write_text(json.dumps([asdict(r) for r in self.records], indent=1), encoding="utf-8")
        return rec

    # --- analysis
    def false_positive_rate_by_rule(self) -> dict[str, dict]:
        fired: Counter = Counter()
        fp: Counter = Counter()
        # multi-rule cases count against every rule
        for r in self.records:
            for rule in r.rules_fired:
                fired[rule] += 1
                if r.disposition == "false_positive":
                    fp[rule] += 1
        return {rule: {"fired": fired[rule], "false_positive": fp[rule], "rate": round(fp[rule] / fired[rule], 2)}
                for rule in sorted(fired)}

    def attribute(self, rec: FeedbackRecord) -> str:
        if rec.disposition != "false_positive":
            return "none"
        if not rec.retrieved_chunks:
            return "retrieval"
        # default True for old records with no verification
        if not rec.verification.get("faithful", True):
            return "generation"
        if rec.rules_fired:
            return "upstream_rule"
        return "routing"

    def attribution_summary(self) -> dict[str, int]:
        return dict(Counter(self.attribute(r) for r in self.records if r.disposition == "false_positive"))

    def golden_candidates(self) -> list[dict]:
        """Reviewed cases shaped like golden.json entries."""
        out = []
        for r in self.records:
            level = {"confirmed": r.recommended_level, "false_positive": "low", "escalated": "high"}[r.disposition]
            out.append({"id": f"reviewed_{r.case_id.lower()}", "claim_id": r.claim_id, "source": "feedback",
                        "reviewer": r.reviewer, "expect": {"risk_level": level, "rules": r.rules_fired,
                                                           # policy:vN:section:part -> policy:section
                                                           "policy_sections": sorted({":".join(c.split(":")[0::2][:2]) for c in r.retrieved_chunks})}})
        return out
