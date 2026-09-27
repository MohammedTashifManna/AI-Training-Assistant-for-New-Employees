---
title: Compliance and Information Security Policy
domain: admin
doc_id: admin-compliance-infosec
owner: Head of InfoSec and Privacy Counsel
policy_version: 6.0
effective_from: 2026-04-01
last_reviewed: 2026-07-01
audience: All employees and contractors
---

# Compliance and Information Security Policy

**Policy version 6.0, effective 1 April 2026.**

Manna's Analytics holds **SOC 2 Type II** and **ISO/IEC 27001:2022** certification
and processes personal data under the **GDPR** and India's **Digital Personal Data
Protection Act, 2023**. Our customers audit us. Individual employee behaviour is
in scope of those audits, which is why these controls are mandatory rather than
advisory.

## 1. Mandatory training

| Module | When | Renewal |
| --- | --- | --- |
| Security Essentials | Before system access beyond email is granted | Annual |
| Data Handling & Privacy | Week 1 | Annual |
| Code of Conduct acknowledgement | Week 1 | Annual |
| Secure Development (Engineering, Data & AI) | Week 2 | Annual |
| Anti-Bribery (Sales, Procurement, Legal) | Week 2 | Annual |

Access to production systems and customer-derived data is **blocked** until
Security Essentials and Data Handling are complete. This is enforced technically,
not by reminder emails.

## 2. Accounts, access and devices

- **Okta SSO** is mandatory for every business system. If a tool cannot do SSO, it
  does not get approved.
- **MFA** is required on every account. Hardware keys or an authenticator app;
  SMS-based codes are not accepted for privileged access.
- Passwords: managed in **1Password**. Never reuse a work password elsewhere, never
  share credentials, never put a credential in Slack, a ticket, a document or
  source code.
- **Least privilege.** Access is requested through the Access Request workflow with
  a business justification and expires by default.
- **Joiner-Mover-Leaver (JML):** access is reviewed on every role change and
  revoked within 4 hours of exit. Quarterly access reviews are run per system.
- **Devices:** only company-managed laptops may access company data. Full-disk
  encryption, endpoint protection and automatic patching are enforced by MDM.
  Screen lock at 5 minutes. Do not disable MDM agents.
- **Personal phones** may be used for email, Slack and MFA provided the device is
  enrolled in mobile management, has a passcode, and is not rooted or jailbroken.
- **No personal cloud storage.** No Google Drive, Dropbox, iCloud or personal
  email for company or customer data.
- **Removable media** is blocked by policy. Exceptions require Head of InfoSec
  approval.

## 3. Data classification and handling

| Class | Examples | Handling |
| --- | --- | --- |
| **Public** | Marketing site, published docs | No restriction |
| **Internal** | Org charts, internal processes, most Slack traffic | Employees only |
| **Confidential** | Financials, roadmap, pricing, contracts, source code | Need to know; do not forward externally |
| **Restricted** | Customer data, personal data, credentials, security findings | Approved systems only; access logged; encryption mandatory |

Rules for **Restricted** data:

- Access only with a specific task requiring it.
- Never copy to a local machine, personal device or personal account.
- Never paste into an unapproved third-party tool, including AI assistants.
- Production customer data must not be used in development or test environments.
  Use the synthetic data generator maintained by Developer Experience.
- Analysis on customer-derived data happens in the managed notebook environment,
  not on your laptop.

## 4. Privacy and personal data

- Collect the minimum personal data needed, use it only for the stated purpose,
  keep it only as long as necessary.
- Every new processing activity or new vendor touching personal data needs a
  **Data Protection Impact Assessment** triaged by Privacy Counsel.
- **Data subject requests** (access, deletion, correction) must be forwarded to
  `privacy@mannasanalytics.example` within **24 hours** of receipt. Statutory
  deadlines are short and start when the request reaches Manna's, not when it
  reaches Legal.
- Cross-border transfers require an approved transfer mechanism. Do not move
  personal data between regions on your own initiative.
- Our **Data Protection Officer** is Meera Nair, General Counsel.

## 5. Approved AI tool usage

Generative AI is encouraged for productivity, within limits.

**Approved for company use:** the enterprise Gemini workspace tenant, GitHub
Copilot Enterprise, and Gemini via Vertex AI in our own Google Cloud project.
These have data processing agreements in place and do not train on our inputs.

**Rules:**

- Never paste **Restricted** data (customer data, personal data, credentials,
  security findings) into any AI tool, approved or not, unless it is one of the
  approved enterprise tenants above.
- Never paste source code into a consumer AI product.
- Treat AI output as a draft. You remain accountable for correctness, licensing
  and security of anything you ship.
- Customer-facing AI features require a model card and Risk & Compliance Board
  review.
- New AI vendors go through Procurement and a DPIA. "It has a free tier" is not an
  approval path.

## 6. Secure development

- Secrets in code are treated as an incident. Use the secrets manager; pre-commit
  and CI scanning are enabled.
- Dependencies: automated vulnerability scanning; criticals patched within 7 days,
  highs within 30.
- Threat modelling required for any new externally-reachable service.
- Peer review is mandatory. Production access is audited and time-bound.
- Annual third-party penetration test; findings tracked to closure by the Risk &
  Compliance Board.

## 7. Phishing and social engineering

- Report suspected phishing with the **Report Phish** button in your mail client.
  Do not forward it to colleagues.
- Manna's runs unannounced simulated phishing campaigns. Clicking a simulation
  results in a short training module, not disciplinary action.
- Verify any unusual request about payments, credentials or urgency **out of band**
  through a known channel. Executive-impersonation fraud is the most common attack
  we see. No genuine Manna's process ever asks you to bypass a control because
  something is urgent.

## 8. Incident reporting

**Report immediately, and always within 1 hour of noticing, even if you are not
sure it is real:**

- Post in `#security-incidents` **and** email `security@mannasanalytics.example`.
- For a lost or stolen device, also call the 24x7 InfoSec line listed in the IT
  Handbook so the device can be remotely wiped.

Report: lost or stolen devices, suspected malware, an email you clicked and then
doubted, data sent to the wrong recipient, an exposed credential, unexpected
access you seem to have, or a customer telling you they can see data that is not
theirs.

**You will never be penalised for reporting an incident you caused.** Delay causes
far more damage than the original mistake, and our regulatory notification clocks
start when Manna's becomes aware.

Severity and escalation, including the 72-hour regulatory notification assessment
under GDPR, are handled by the InfoSec on-call and Privacy Counsel.

## 9. Physical security

- Badge in individually. Do not tailgate and do not let others tailgate you.
- Visitors are registered in advance and escorted at all times.
- Clear desk for anything Confidential or Restricted; lock your screen when you
  step away.
- Whiteboards are wiped after meetings involving customer or financial detail.
- Challenge politely, or call security, if you see someone unbadged on the floor.

## 10. Consequences and questions

Deliberate circumvention of these controls is a disciplinary matter under the Code
of Conduct and may breach law or contract. Genuine mistakes reported promptly are
treated as learning opportunities.

Questions: `#ask-security` for security, `#ask-legal` for privacy and compliance.
When you are unsure whether something is allowed, ask **before** you do it. The
answer is usually yes with a small adjustment.
