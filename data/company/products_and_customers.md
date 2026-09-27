---
title: Products, Customers and Market
domain: company
doc_id: company-products-customers
owner: Product Marketing
last_reviewed: 2026-07-10
audience: All employees
---

# Products, Customers and Market

## The problem we solve

A mid-market company typically runs 40 to 120 SaaS tools. Each one reports its
own version of "revenue", "active customer" and "churn". Finance publishes one
number, Sales quotes another, the board sees a third. Weeks are lost
reconciling spreadsheets instead of acting.

Manna's Analytics puts a **governed semantic layer** between raw data and the
people making decisions, so a metric is defined once, versioned, owned by a
named person, and consumed identically everywhere.

## The three modules

### Manna's Core

The metrics catalogue and semantic layer.

- Connects to Snowflake, BigQuery, Databricks, Redshift and Postgres.
- Metrics are defined in a version-controlled specification, reviewed like code.
- Every metric carries an owner, a definition, a freshness guarantee and a
  lineage graph back to source tables.
- Consumed through the Manna's UI, a SQL interface, a REST API, and connectors
  for spreadsheet tools and BI tools.

**Typical buyer:** VP Data, Head of Analytics, or a CFO who has lost patience
with reconciliation.

### Manna's Signals

Anomaly detection layered on Core metrics.

- Learns the seasonality of each metric and alerts when behaviour deviates.
- Alerts route to Slack, Teams, email, PagerDuty or a webhook.
- Every alert links back to the metric definition and the contributing segments,
  so the first question ("is this real?") is answerable in one click.
- Includes root-cause hints that rank which dimensions moved most.

**Typical buyer:** operations leaders who need to know about a problem the day
it happens, not at the month-end review.

### Manna's Embed

White-labelled analytics our customers ship to *their* customers.

- Dashboards rendered inside the customer's own product, fully themed.
- Row-level security enforced by tenant, so each end user only sees their data.
- Usage-based commercial model tied to monthly active viewers.

**Typical buyer:** product leaders at software companies who would rather not
build an analytics team.

## Who buys from us

| Segment | Definition | Share of ARR |
| --- | --- | --- |
| Mid-market | 200-1,500 employees | 54% |
| Enterprise | 1,500+ employees | 38% |
| Growth | Under 200 employees | 8% |

Strongest verticals, in order: logistics and supply chain, financial services,
retail and e-commerce, healthcare operations, and B2B SaaS.

Geographic split of ARR: Americas 47%, EMEA 33%, APAC 20%.

## Reference customers

These accounts have signed public reference agreements and may be named in
external conversations:

- **Vantage Freight** - logistics, uses Core and Signals across 14 depots.
- **Meridian Retail Group** - 600-store retailer, replaced a nightly
  spreadsheet process with Core.
- **Halcyon Health** - healthcare operations, uses Signals for capacity alerts.
- **Brightloom Software** - ships Manna's Embed to 8,000 of its own customers.

Any other customer name is confidential. Do not use unapproved logos or names in
external decks; check with Product Marketing first.

## Competitive landscape

| Competitor type | Examples we hear | How we differ |
| --- | --- | --- |
| BI suites | Legacy dashboarding vendors | We govern definitions, not just visualise them |
| Metrics startups | Newer semantic-layer tools | We ship alerting and embedding, not just the layer |
| Build-it-yourself | Internal data platform teams | Time to value in weeks, plus lineage and ownership |

Our most common real competitor is **"do nothing and keep using spreadsheets"**.

## Commercial basics every new joiner should know

- Contracts are annual by default, multi-year with a discount.
- Pricing scales on **modelled metrics** and **monthly active viewers**, not seats.
- Standard payment terms are net 30.
- The fiscal year runs 1 April to 31 March, so Q1 is April to June.
- Net revenue retention for FY2025-26 was 118%.
