---
policy_id: PI-004
title: Investigation Workflow and Escalation
version: 5
status: approved
effective_date: 2025-01-01
access_level: standard
claim_types: [professional, facility, vendor]
jurisdictions: [US]
---

# PI-004 Investigation Workflow and Escalation

## 1. Review Paths

Flagged cases are routed by risk tier and confidence:

- Low tier with high confidence: automated handling, closed with a logged
  rationale.
- Medium tier: assigned to an analyst queue; the analyst may close or
  escalate.
- High tier, or any case where the automated assessment reports low
  confidence or missing evidence: held for investigator review before any
  outbound action.

## 2. Prior Case Context

Prior cases for the same member, provider, or vendor must be retrieved and
cited. A prior false-positive disposition for the same pattern lowers the
severity by one level unless new evidence contradicts it. A prior confirmed
case raises severity by one level.

## 3. Human Review Requirements

An investigator must approve before: issuing a recovery letter above 1,000
USD, suspending a provider or vendor, or releasing a held payment. The
approval, the reviewer, and the rationale are recorded in the case audit
trail. The automated summary is advisory and never the final decision.

## 4. Feedback

Every investigator disposition, including overrides of the automated
recommendation, is captured with the evidence and recommendation it
overrode, and feeds the quarterly review of risk rules and retrieval quality.
