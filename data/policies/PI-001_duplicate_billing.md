---
policy_id: PI-001
title: Duplicate Claims and Duplicate Payments
version: 4
status: approved
effective_date: 2025-01-01
access_level: standard
claim_types: [professional, facility, vendor]
jurisdictions: [US]
---

# PI-001 Duplicate Claims and Duplicate Payments

## 1. Definitions

A duplicate claim is a second submission for the same member, provider,
service date, and procedure code where the first submission was adjudicated.
A duplicate payment is more than one disbursement against a single
adjudicated claim, regardless of how the second disbursement was triggered.

## 2. Detection Standard

Payment integrity systems must flag any claim with more than one payment
record. Resubmissions caused by portal timeouts or system retries are a known
source of duplicate payments and are not exempt from review; see Section 4
for the disposition rules.

## 3. Investigation Steps

1. Retrieve all payment records for the claim and confirm the count and total.
2. Confirm whether a reversal or offset already exists.
3. Check prior cases for the same member and provider for a known
   resubmission pattern, as recorded under Policy PI-004 Section 2.
4. Document the evidence identifiers for each payment in the case summary.

## 4. Disposition

- If a reversal exists, close the case as no action with the reversal id.
- If no reversal exists and the second payment is under 1,000 USD, raise a
  recovery request and close; no human approval is required.
- If no reversal exists and the second payment is 1,000 USD or more, or the
  provider has a prior confirmed case, escalate to an investigator before any
  recovery letter is issued (Policy PI-004 Section 3).
