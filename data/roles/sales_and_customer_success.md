---
title: Sales and Customer Success - Team Charter and Role Guide
domain: role
doc_id: role-sales-cs
owner: Chief Revenue Officer
last_reviewed: 2026-07-04
audience: Sales, Customer Success
---

# Sales and Customer Success - Team Charter and Role Guide

**Sales lead:** Daniel Osei, Chief Revenue Officer (headcount 190)
**Customer Success lead:** Lena Hoffmann, VP Customer Success (headcount 165)

## Where the boundary sits

Sales owns everything up to signature. Customer Success owns everything after,
including renewal. Expansion is **jointly owned**: the CSM spots the opportunity,
the AE closes it. This is the most common source of confusion for new joiners, so
when in doubt: *has the contract been signed?*

## Sales roles

### Sales Development Representative (SDR)

Generates qualified pipeline through outbound prospecting and inbound follow-up.
Measured on qualified meetings accepted by an AE, not on raw activity.

### Account Executive (AE)

Owns the deal from qualified opportunity through signature. Segmented into
Growth, Mid-Market and Enterprise. Enterprise AEs typically run six to nine month
cycles with multiple stakeholders.

### Solutions Engineer (SE)

The technical voice in the deal. Runs discovery, demos, security questionnaires
and proof-of-concept engagements. SEs are not carrying quota individually; they
share the AE's number.

### Sales Manager / Regional Director

Owns a team's number, forecast accuracy and coaching.

### Revenue Operations

Owns the CRM, territory design, quota setting, commission calculation and
forecast hygiene.

## Customer Success roles

### Customer Success Manager (CSM)

Owns adoption, health and renewal for a book of accounts. Runs QBRs, tracks
success plans, and is the commercial escalation point post-sale.

### Technical Account Manager (TAM)

Assigned to Enterprise accounts. Owns technical health: connector reliability,
query performance, upgrade planning.

### Onboarding Consultant

Runs the first 60 days after signature: connecting sources, modelling the first
metrics, training the customer's team.

### Support Engineer

Handles inbound tickets against published response targets. Escalates to
Engineering through the on-call path when a defect is confirmed.

## The sales process

| Stage | Exit criteria |
| --- | --- |
| 1. Qualified | Pain, authority and a rough timeline confirmed |
| 2. Discovery | Current metric process mapped, success criteria agreed in writing |
| 3. Technical validation | POC or deep demo complete, security review passed |
| 4. Business case | Pricing proposed, ROI framed, procurement path known |
| 5. Negotiation | Redlines with Legal, discount approved if outside standard |
| 6. Closed won | Signature received, handover to CS scheduled |

- Standard POCs are time-boxed to **21 days** with written success criteria. POCs
  without success criteria are not approved.
- Discounts beyond the standard schedule need Pricing Committee approval.
- Any non-standard contract language goes to Legal via `#ask-legal`. Never agree
  to bespoke terms verbally.

## Support response targets

| Severity | Definition | First response |
| --- | --- | --- |
| Sev-1 | Production down or data materially wrong | 1 hour, 24x7 |
| Sev-2 | Major feature broken, workaround exists | 4 business hours |
| Sev-3 | Minor defect or question | 1 business day |
| Sev-4 | Enhancement request | 3 business days |

Enterprise contracts may carry tighter commitments; always check the account's
contract record rather than assuming the standard table applies.

## Compensation mechanics

- AEs and SDRs are on a base plus variable plan, with variable paid quarterly
  after the quarter closes and collections are confirmed.
- CSMs carry a variable component tied to gross retention and expansion.
- New joiners in quota-carrying roles receive a **ramp guarantee** for their
  first two full quarters: variable is paid at target regardless of attainment
  while territory and pipeline build.
- Commission plans are issued individually and are confidential. Discussing your
  own compensation is your right; sharing someone else's plan document is not.

## Tools

| Purpose | Tool |
| --- | --- |
| CRM | Salesforce |
| Sales engagement | Outreach |
| Conversation intelligence | Gong |
| Proposals and e-signature | DocuSign |
| Support ticketing | Zendesk |
| Customer health scoring | Manna's Core (customer zero) |
| Data enrichment | ZoomInfo |

## Your first 30 days in Sales or CS

1. **Week 1:** Systems access, product overview sessions, shadow three customer
   calls, complete security and data-handling training.
2. **Week 2:** Certification on the core demo. You must pass a demo certification
   with an SE before running a customer-facing demo alone.
3. **Week 3:** Territory or book review with your manager; build your first
   30-60-90 plan.
4. **Week 4:** Run a live discovery call or QBR with your manager observing.

## Rules that matter from day one

- **Never quote a roadmap date to a customer.** Direct roadmap questions to
  Product through your CSM or AE.
- **Never name a customer who is not on the public reference list.**
- **Never share another customer's data, dashboards or configuration**, including
  in screenshots during demos. Use the demo tenant.
- **Log it in the CRM.** If a commitment to a customer is not written down in
  Salesforce, it does not exist.
