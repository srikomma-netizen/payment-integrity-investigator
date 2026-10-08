"""Reviewer feedback store and offline analysis (false-positive rate per rule, failure attribution,
golden-set candidates). Nothing here feeds the model online.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DISPOSITION_BY_DECISION = {"approve": "confirmed", "reject": "false_positive", "escalate": "escalated"}


# Stores the decision alongside what the system recommended and why (rules, chunks, verification),
# so an override can be analysed later rather than just recorded.
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
        # path=None keeps everything in memory (tests, demo)
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
            # TODO: rewrites the whole file on every record; move to append-only JSONL or a table if volume grows
            self.path.write_text(json.dumps([asdict(r) for r in self.records], indent=1), encoding="utf-8")
        return rec

    # ---- analysis ------------------------------------------------------
    def false_positive_rate_by_rule(self) -> dict[str, dict]:
        fired: Counter = Counter()
        fp: Counter = Counter()
        # a case with several rules counts against each of them; we can't tell which one misled the reviewer
        for r in self.records:
            for rule in r.rules_fired:
                fired[rule] += 1
                if r.disposition == "false_positive":
                    fp[rule] += 1
        return {rule: {"fired": fired[rule], "false_positive": fp[rule], "rate": round(fp[rule] / fired[rule], 2)}
                for rule in sorted(fired)}

    def attribute(self, rec: FeedbackRecord) -> str:
        """Where did a wrong recommendation most likely come from?"""
        if rec.disposition != "false_positive":
            return "none"
        # Checked in pipeline order and the first broken stage wins: a later stage can't be blamed
        # for bad input it was handed.
        if not rec.retrieved_chunks:
            return "retrieval"
        # default True: an old record without verification shouldn't be blamed on generation
        if not rec.verification.get("faithful", True):
            return "generation"
        # pipeline was clean, so the flag itself was wrong
        if rec.rules_fired:
            return "upstream_rule"
        # nothing fired and nothing broke, yet it still reached a reviewer
        return "routing"

    def attribution_summary(self) -> dict[str, int]:
        return dict(Counter(self.attribute(r) for r in self.records if r.disposition == "false_positive"))

    def golden_candidates(self) -> list[dict]:
        """Reviewed cases shaped like golden entries. Human-confirmed level
        replaces the model's; an SME still curates before merge."""
        out = []
        for r in self.records:
            # KeyError here means a decision outside DISPOSITION_BY_DECISION slipped through record()
            level = {"confirmed": r.recommended_level, "false_positive": "low", "escalated": "high"}[r.disposition]
            out.append({"id": f"reviewed_{r.case_id.lower()}", "claim_id": r.claim_id, "source": "feedback",
                        "reviewer": r.reviewer, "expect": {"risk_level": level, "rules": r.rules_fired,
                                                           # chunk ids are policy:vN:section:part; keep policy:section like golden.json
                                                           "policy_sections": sorted({":".join(c.split(":")[0::2][:2]) for c in r.retrieved_chunks})}})
        return out
