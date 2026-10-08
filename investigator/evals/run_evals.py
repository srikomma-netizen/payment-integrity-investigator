"""Golden-set evals, scored per stage.

Run:  python -m investigator.evals.run_evals            (offline)
      LLM_PROVIDER=anthropic python -m investigator.evals.run_evals
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..graph.build import Investigator
from ..llm.providers import make_llm

GOLDEN_PATH = Path(__file__).with_name("golden.json")


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class CaseResult:
    id: str
    tags: list[str]
    retrieval: dict
    generation: dict
    workflow: dict
    attribution: str | None
    passed: bool
    notes: list[str] = field(default_factory=list)


def _section_key(chunk: dict) -> str:
    return f"{chunk['policy_id']}:{chunk['section_id']}"


def score_retrieval(state: dict, expected_sections: list[str], index) -> dict:
    chunks = state.get("retrieval", {}).get("chunks", [])
    primary = [c for c in chunks if not c.get("expanded_from")]
    # recall includes expanded chunks, precision/mrr don't
    got_all = [_section_key(c) for c in chunks]
    got_primary = list(dict.fromkeys(_section_key(c) for c in primary))
    hits = [s for s in expected_sections if s in got_all]
    recall = len(hits) / len(expected_sections) if expected_sections else 1.0
    precision = (len([s for s in got_primary if s in expected_sections]) / len(got_primary)) if got_primary else 0.0
    mrr = 0.0
    for i, s in enumerate(got_primary):
        if s in expected_sections:
            mrr = 1 / (i + 1)
            break
    authorized = all(c["chunk_id"] in index.chunks and index.chunks[c["chunk_id"]].status == "approved"
                     and index.chunks[c["chunk_id"]].version == index.current_version.get(c["policy_id"]) for c in chunks)
    return {"recall": round(recall, 2), "precision": round(precision, 2), "mrr": round(mrr, 2),
            "authorized": authorized, "retrieved": got_primary}


def score_generation(state: dict, exp: dict) -> dict:
    summ, ver = state.get("summary"), state.get("verification", {})
    if not summ:
        return {"schema_valid": False, "faithful": False, "risk_agreement": False, "indicator_recall": 0.0}
    cited = {e for ind in summ["suspicious_indicators"] for e in ind["evidence_ids"]}
    expected_rules = exp.get("rules", [])
    # covered only if an indicator cites the signal id
    represented = [r for r in expected_rules if f"SIG-{r}" in cited]
    return {
        "schema_valid": True,
        "faithful": bool(ver.get("faithful")),
        "risk_agreement": summ["risk_level"] == exp["risk_level"],
        "indicator_recall": round(len(represented) / len(expected_rules), 2) if expected_rules else 1.0,
        "confidence": summ["confidence"],
    }


def score_workflow(state: dict, exp: dict) -> dict:
    failures = state.get("tool_failures", [])
    return {
        "tier_match": state.get("risk", {}).get("tier") == exp["tier"],
        "route_match": state.get("route") == exp["route"],
        "tool_failures": len(failures),
        "degraded_gracefully": state.get("status") != "failed" and (len(failures) >= exp.get("min_tool_failures", 0)),
    }


def attribute(r: dict, g: dict, w: dict) -> str | None:
    """Blame the earliest failing stage."""
    if not w["tier_match"]:
        return "upstream_risk"
    if r["recall"] < 1.0 or not r["authorized"]:
        return "retrieval"
    if not g["schema_valid"] or not g["faithful"] or not g["risk_agreement"] or g["indicator_recall"] < 1.0:
        return "generation"
    if not w["route_match"] or not w["degraded_gracefully"]:
        return "routing"
    return None


def evaluate_case(inv: Investigator, case: dict) -> CaseResult:
    exp = case["expect"]
    if case.get("fail_tool"):
        # times > TOOL_RETRIES or the retry hides it
        inv.registry.fail_next(case["fail_tool"]["name"], case["fail_tool"]["times"])
    state = inv.start(case["claim_id"], role=case.get("role", "investigator"))
    r = score_retrieval(state, exp.get("policy_sections", []), inv.index)
    g = score_generation(state, exp)
    w = score_workflow(state, exp)
    attr = attribute(r, g, w)
    notes = []
    if attr:
        notes.append(f"attributed to {attr}: retrieval={r} generation={g} workflow={w}")
    return CaseResult(case["id"], case.get("tags", []), r, g, w, attr, attr is None, notes)


def run_suite(cases: list[dict] | None = None, *, provider: str | None = None) -> list[CaseResult]:
    cases = cases or load_golden()
    inv = Investigator(llm=make_llm(provider))
    return [evaluate_case(inv, c) for c in cases]


def summarize(results: list[CaseResult]) -> dict:
    n = len(results) or 1
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for t in r.tags:
            by_tag[t].append(r.passed)
    return {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "retrieval": {"mean_recall": round(sum(r.retrieval["recall"] for r in results) / n, 2),
                      "mean_precision": round(sum(r.retrieval["precision"] for r in results) / n, 2),
                      "mean_mrr": round(sum(r.retrieval["mrr"] for r in results) / n, 2),
                      "authorized_rate": f"{sum(r.retrieval['authorized'] for r in results)}/{len(results)}"},
        "generation": {"faithful_rate": f"{sum(r.generation['faithful'] for r in results)}/{len(results)}",
                       "risk_agreement_rate": f"{sum(r.generation['risk_agreement'] for r in results)}/{len(results)}"},
        "workflow": {"route_match_rate": f"{sum(r.workflow['route_match'] for r in results)}/{len(results)}"},
        "failure_attribution": dict(Counter(r.attribution for r in results if r.attribution)),
        "by_tag": {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_tag.items())},
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    results = run_suite(provider=argv[0] if argv else None)
    width = max(len(r.id) for r in results) + 2
    print(f"{'case'.ljust(width)}pass  recall  prec  mrr   faithful  risk  route  tool_fail")
    for r in results:
        print(f"{r.id.ljust(width)}{'PASS' if r.passed else 'FAIL'}  {r.retrieval['recall']:<6}  {r.retrieval['precision']:<4}  "
              f"{r.retrieval['mrr']:<4}  {str(r.generation['faithful']):<8}  {str(r.generation['risk_agreement']):<4}  "
              f"{str(r.workflow['route_match']):<5}  {r.workflow['tool_failures']}")
        for n in r.notes:
            print(f"{''.ljust(width)}  - {n}")
    print()
    print(json.dumps(summarize(results), indent=2))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
