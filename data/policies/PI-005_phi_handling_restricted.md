---
policy_id: PI-005
title: PHI Handling in Investigations
version: 2
status: approved
effective_date: 2025-01-01
access_level: restricted
claim_types: [professional, facility, vendor]
jurisdictions: [US]
---

# PI-005 PHI Handling in Investigations (restricted)

## 1. Minimum Necessary

Investigation tooling exposes member identity only to roles with a documented
need. Analysts work on pseudonymous tokens. Investigators may resolve name,
medical record number, and date of birth. Full identifiers are limited to the
SIU lead.

## 2. Model Boundary

No protected health information may be sent to a language model. All records
are de-identified before retrieval or summarization, and re-identification
happens only in the application layer against the requesting role's
clearance, with every resolution logged.
