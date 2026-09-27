---
title: Engineering - Team Charter and Role Guide
domain: role
doc_id: role-engineering
owner: Office of the CTO
last_reviewed: 2026-07-02
audience: Engineering
---

# Engineering - Team Charter and Role Guide

**Function lead:** Vikram Iyer, Co-founder & CTO
**Headcount:** 310

## What Engineering owns

Engineering builds and operates the Manna's platform: the semantic layer, the
query engine, connectors, the alerting pipeline, the embedding runtime, and the
infrastructure underneath all of it.

## Groups within Engineering

| Group | Owns | Lead |
| --- | --- | --- |
| Ingestion | Connectors, scheduling, source-system reliability | Kavya Srinivasan |
| Semantic Engine | Metric compilation, query planning, caching | Tobias Lang |
| Signals Platform | Anomaly detection pipeline, alert routing | Farah Qureshi |
| Embed Runtime | Multi-tenant rendering, row-level security | Marcus Bell |
| Platform & Infra | Kubernetes, CI/CD, observability, cost | Sneha Pillai |
| Developer Experience | Build tooling, test infrastructure, local dev | Joel Mathew |

## Engineering roles

### Software Engineer (L3-L4)

Delivers well-scoped features inside a pod with review support. Expected to
write tests, participate in code review, and take part in on-call after the
first 90 days.

### Senior Software Engineer (L5)

Owns a subsystem end to end. Breaks ambiguous problems into deliverable work,
mentors junior engineers, and represents the pod in design discussions.

### Staff Engineer (L6)

Operates across pods on problems that no single pod can solve. Drives designs
through Architecture Review. Accountable for technical outcomes without direct
reports.

### Principal Engineer (L7-L8)

Sets multi-year technical direction for a domain. Permanent member of
Architecture Review.

### Engineering Manager (M1-M3)

Owns team health, delivery and growth. Managers at Manna's are expected to stay
technically literate but are not expected to be the strongest coder on the team.
The IC and management tracks are compensated equivalently at the same grade.

## Technology stack

| Layer | Technology |
| --- | --- |
| Backend services | Python 3.12 (FastAPI), Go for latency-critical paths |
| Query and compute | Rust for the metric compiler, DuckDB for local evaluation |
| Frontend | TypeScript, React, Vite |
| Data warehouses supported | Snowflake, BigQuery, Databricks, Redshift, Postgres |
| Streaming | Kafka |
| Orchestration | Temporal for workflows, Airflow for legacy batch |
| Infrastructure | AWS (primary, `ap-south-1` and `us-east-1`), Kubernetes (EKS) |
| IaC | Terraform |
| CI/CD | GitHub Actions, Argo CD for deploys |
| Observability | OpenTelemetry, Prometheus, Grafana, Loki |
| Source control | GitHub Enterprise |

## Engineering practices

- **Trunk-based development.** Short-lived branches, merged behind feature flags.
- **Two approvals** required on any change to a service with a customer-facing
  SLA; one approval otherwise.
- **CI must be green** before merge. Flaky tests are quarantined, not ignored.
- **Design docs** are required for anything that changes a public interface,
  crosses a group boundary, or is expected to take more than two weeks. Template
  lives in the Engineering handbook space.
- **Architecture Review (AR)** meets Wednesdays. Submit the doc 48 hours ahead.
- **Deploys** happen continuously. Friday afternoon deploys of high-risk changes
  are discouraged but not blocked.

## On-call

- Every group runs a weekly rotation, one primary and one secondary.
- New engineers join the rotation **after** their first 90 days and only after
  shadowing two full rotations.
- Severity definitions and the RCA process are documented in the Platform
  runbooks. A Sev-1 or Sev-2 requires a written RCA within five working days.
- On-call compensation: a weekly on-call allowance plus time-off-in-lieu for any
  night-time page, claimed through the leave system as "On-call TOIL".

## Your first 30 days in Engineering

1. **Day 1-2:** Laptop, Okta, GitHub access, join your pod channels.
2. **Day 3:** Get the local development environment running. The target is a
   working local stack by end of day three. If it takes longer, that is a bug in
   our tooling, so raise it in `#dev-experience`.
3. **Day 4-5:** Ship a documentation fix or a small test improvement to
   experience the full pipeline from branch to production.
4. **Week 2:** First reviewed feature ticket, paired with your buddy.
5. **Week 3:** Attend Architecture Review as an observer.
6. **Week 4:** Shadow an on-call shift. Present what you learned at your pod
   retro.

## Key contacts

| Need | Where |
| --- | --- |
| Local environment problems | `#dev-experience` |
| Production incident | `#incidents`, page via PagerDuty |
| Access to a repository or system | `#ask-it` with your manager's approval |
| Architecture question | `#architecture`, or bring it to AR |
