---
title: Internal Tools and Access Guide
domain: admin
doc_id: admin-internal-tools
owner: IT and Developer Experience
policy_version: 3.3
effective_from: 2026-04-01
last_reviewed: 2026-07-06
audience: All employees
---

# Internal Tools and Access Guide

Everything below is behind **Okta SSO**. If a tool asks you for a separate
password, stop and check with `#ask-it` before entering credentials, because that
is also what a phishing page does.

## Tool directory

| Tool | Used for | Who gets it | Request via |
| --- | --- | --- | --- |
| **Okta** | Single sign-on, MFA | Everyone | Automatic, day 1 |
| **Google Workspace** | Email, calendar, docs | Everyone | Automatic, day 1 |
| **Slack** | Messaging | Everyone | Automatic, day 1 |
| **Workday** | HR records, leave, payslips, conflicts register | Everyone | Automatic, day 1 |
| **1Password** | Credential management | Everyone | Automatic, day 1 |
| **Expensify** | Expense claims | Everyone | Automatic, day 1 |
| **Workplace app** | Desk and room booking, visitor registration | Everyone | Automatic, day 1 |
| **Confluence** | Handbook, design docs, runbooks | Everyone | Automatic, day 1 |
| **Jira** | Work tracking | Everyone | Automatic, day 2 |
| **TravelPerk** | Travel booking | Everyone | On first travel request |
| **GitHub Enterprise** | Source control, code review | Engineering, Data & AI | Access Request + manager approval |
| **Argo CD** | Deployments | Engineering | Access Request + group lead |
| **PagerDuty** | On-call paging | Engineers in a rotation | Automatic when added to rotation |
| **Grafana / Loki** | Dashboards, logs, traces | Engineering, Data & AI | Access Request |
| **AWS console** | Infrastructure | Platform & Infra, break-glass for others | Access Request + group lead, time-bound |
| **Snowflake / warehouse** | Internal analytics | Data & AI, analysts | Access Request + data handling training |
| **dbt Cloud** | Transformation models | Analytics Engineering | Access Request |
| **MLflow** | Experiment tracking | Data & AI | Access Request |
| **Managed notebooks** | Analysis on customer-derived data | Data & AI | Access Request + data handling training |
| **Salesforce** | CRM | Sales, CS, Finance, Product (read) | Access Request |
| **Outreach** | Sales engagement | SDRs, AEs | Access Request |
| **Gong** | Call recording and coaching | Sales, CS, Product (read) | Access Request |
| **DocuSign** | Contracts and signature | Sales, Legal, Finance | Access Request |
| **Zendesk** | Support ticketing | Support, CS | Access Request |
| **Figma** | Design | Product & Design, Engineering (view) | Access Request |
| **Manna's Core (internal tenant)** | Our own metrics | Everyone | Automatic, day 2 |

## How to request access

1. Open the **Access Request** app from your Okta dashboard.
2. Pick the system and the role or permission level.
3. Give a **business justification**. "Need it for my job" is declined; name the
   task or system you must work on.
4. Approval flows to your manager, and to the system owner for anything privileged.
5. Standard turnaround is **1 working day**; privileged and production access takes
   up to 3.

Notes that save time:

- Access to production and to customer-derived data is **blocked until Security
  Essentials and Data Handling training are complete**. This is enforced in the
  provisioning pipeline, so chasing IT will not unblock it.
- Access grants **expire by default**. Renewal prompts arrive 7 days before expiry.
- Privileged access is time-bound and audited. Break-glass access is available for
  incidents and is reviewed afterwards.
- Access is reviewed on every role change under the Joiner-Mover-Leaver process.

## Slack conventions

| Channel | Purpose |
| --- | --- |
| `#announcements` | Company-wide, read-only |
| `#ask-it` | Laptops, accounts, access, software |
| `#ask-people-ops` | HR, payroll, benefits, leave |
| `#ask-finance` | Expenses, reimbursements, budgets |
| `#ask-procurement` | Buying software or services |
| `#ask-legal` | Contracts, privacy, conflicts, compliance |
| `#ask-security` | Security questions (non-urgent) |
| `#security-incidents` | Suspected incidents, **urgent** |
| `#incidents` | Production incidents |
| `#dev-experience` | Local environment and tooling problems |
| `#architecture` | Technical design discussion |
| `#genai` | GenAI, RAG and agent work |
| `#onboarding-buddies` | New joiners and their buddies |
| `#demo-friday` | Fortnightly demo signup |
| `#random` | Non-work chat |

Conventions:

- Ask in the public `#ask-*` channel rather than DMing an individual. Answers are
  then searchable for the next person, which is the point.
- Use threads for replies.
- `@here` is for time-sensitive matters affecting the whole channel; `@channel` needs
  a genuinely good reason.
- Anything Confidential or Restricted does not belong in a public channel.

## Getting new software

All software, including free tools and browser extensions, goes through
**Procurement** with a security review. This is a security and data-protection
requirement, not a cost-control exercise: an unreviewed tool can quietly become a
data processor for customer data.

1. Post in `#ask-procurement` with the tool, the problem it solves, and whether it
   will touch personal or customer data.
2. Procurement checks whether we already own something equivalent. Often we do.
3. Security review, plus a Data Protection Impact Assessment if personal data is
   involved.
4. Contract, SSO integration, then provisioning.

Personally purchased software is **not reimbursable** under the expense policy.

## Laptop and hardware support

| Need | Route | Notes |
| --- | --- | --- |
| New starter laptop | Automatic | Shipped or collected day 1 |
| Broken hardware | `#ask-it` | Loaner available same day in Bengaluru |
| Peripherals (monitor, keyboard, headset) | `#ask-it` | Ordered by IT, not expensed |
| Refresh cycle | Automatic | Every 3 years, or earlier if justified |
| Lost or stolen device | `#security-incidents` **and** the 24x7 InfoSec line | Remote wipe within 1 hour |
| Software install request | `#ask-it` | Self-service catalogue covers most needs |

Do not disable the MDM agent or endpoint protection. Doing so removes your access
automatically and generates a security ticket.

## Standard IT response targets

| Priority | Example | First response |
| --- | --- | --- |
| P1 | Cannot work at all, hardware failure | 1 hour |
| P2 | Degraded, workaround exists | 4 business hours |
| P3 | Access request, standard software | 1 business day |
| P4 | Enhancement, non-urgent | 3 business days |

## First-week tool checklist

- [ ] Okta MFA enrolled with an authenticator app or hardware key
- [ ] 1Password vault set up, work passwords moved in
- [ ] Slack profile completed: full name, role, pronouns if you wish, photo, location
- [ ] Google Calendar working hours and time zone set
- [ ] Workday personal details and bank account verified
- [ ] Expensify linked and cost centre confirmed
- [ ] Security Essentials and Data Handling training complete
- [ ] Function-specific tools requested through Access Request
- [ ] Joined your function, group and pod Slack channels
- [ ] Manna's Core internal tenant opened at least once
