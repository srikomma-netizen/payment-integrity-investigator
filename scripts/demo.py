"""End-to-end demo.

Run:  python scripts/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# run without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from investigator.evals.run_evals import run_suite, summarize   # noqa: E402
from investigator.feedback.store import FeedbackStore           # noqa: E402
from investigator.graph.build import Investigator               # noqa: E402
from investigator.risk.rules import flagged_claims              # noqa: E402


def show(st: dict) -> None:
    print(f"\n=== {st['claim_id']} as {st['role']}: status={st['status']} route={st.get('route')}")
    for line in st["audit"]:
        print("   ", line)
    summ = st.get("summary") or {}
    if summ:
        print(f"   level={summ['risk_level']} conf={summ['confidence']} action={summ['recommended_action']}")
        for ind in summ["suspicious_indicators"]:
            print(f"     - [{ind['severity']}] {ind['description']}  evidence={ind['evidence_ids']}")
        print(f"   citations={summ['policy_citations']}")


def main() -> None:
    inv = Investigator()
    print(f"LLM provider: {type(inv.llm).__name__}")
    print("\nUpstream controls flagged:", [(a.claim_id, a.tier, [s.rule_id for s in a.signals]) for a in flagged_claims(inv.ds)])

    show(inv.start("CLM-1003"))                 # clean
    show(inv.start("CLM-1007"))                 # duplicate payment
    st = inv.start("CLM-1012")                  # upcoding -> human review
    show(st)
    print("\n   pending review packet keys:", list(st["pending_review"]))
    print("   analyst view of member:    ", inv.view(st["case_id"], "analyst")["claim"]["member"])
    print("   investigator view of member:", inv.view(st["case_id"], "investigator")["claim"]["member"])
    fb = FeedbackStore()
    done = inv.resume(st["case_id"], decision="approve", reviewer="inv_4", rationale="chart does not support level 5")
    fb.record(done, decision="approve", reviewer="inv_4")
    print("   after approval:", done["status"], "|", done["audit"][-2:])

    inv.registry.fail_next("get_payment_records", 2)   # attempt + retry
    show(inv.start("CLM-1007"))

    print("\n=== Evals (retrieval / generation / workflow scored separately) ===")
    results = run_suite(provider="fake")
    for r in results:
        print(f"   {'PASS' if r.passed else 'FAIL'} {r.id:<42} recall={r.retrieval['recall']} faithful={r.generation['faithful']} route={r.workflow['route_match']}")
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
