---
title: Data & AI - Team Charter and Role Guide
domain: role
doc_id: role-data-ai
owner: VP Data & AI
last_reviewed: 2026-07-08
audience: Data & AI
---

# Data & AI - Team Charter and Role Guide

**Function lead:** Rahul Deshpande, VP Data & AI
**Headcount:** 85

## What Data & AI owns

Two distinct mandates sit inside this function, and new joiners often conflate
them:

1. **Product intelligence** - the machine learning that powers Manna's Signals:
   forecasting, seasonality decomposition, anomaly scoring, root-cause ranking,
   and the natural-language metric search in Core.
2. **Internal analytics** - the data that Manna's itself runs on: product usage,
   funnel and retention analysis, and the metrics the exec team reviews weekly.

## Teams within Data & AI

| Team | Owns | Lead |
| --- | --- | --- |
| Applied ML | Anomaly models, forecasting, root-cause ranking | Ishita Bose |
| ML Platform | Feature store, training pipelines, model serving, evaluation harnesses | Daniel Cho |
| GenAI | Natural-language metric search, assistive copilots, RAG systems | Nikhil Varma |
| Analytics Engineering | Internal warehouse models, dbt project, metric definitions | Grace Okonkwo |
| Product Analytics | Funnel, retention and experiment analysis | Emily Sanders |

## Data & AI roles

### Data Scientist (L3-L5)

Frames business problems as measurable questions, runs analysis, designs
experiments. Strong SQL is non-negotiable; Python is expected.

### Machine Learning Engineer (L4-L6)

Takes models from notebook to production and owns them there: latency,
correctness, retraining, drift monitoring, and rollback.

### Analytics Engineer (L3-L5)

Owns the transformation layer. Models raw data into trustworthy, documented,
tested tables and metric definitions using dbt.

### Research Scientist (L6-L7)

Works on problems 12 to 24 months out, typically in time-series modelling and
causal inference. Publication is supported but not required.

### AI Engineer / GenAI (L4-L6)

Builds LLM-backed features: retrieval pipelines, prompt and context design,
tool-using agents, evaluation harnesses, and guardrails.

## Technology stack

| Layer | Technology |
| --- | --- |
| Languages | Python 3.12, SQL, some Scala for legacy Spark jobs |
| Notebooks | JupyterLab on managed compute; local notebooks not permitted for customer data |
| Transformation | dbt Core, orchestrated by Temporal |
| ML frameworks | PyTorch, scikit-learn, statsmodels, Prophet |
| Feature store | Feast |
| Experiment tracking | MLflow |
| Model serving | Seldon on EKS, with a Go inference gateway |
| Vector search | pgvector for smaller corpora, Qdrant for large ones |
| LLM access | Gemini via Vertex AI, plus a self-hosted open-weights model for cost-sensitive paths |
| Agent framework | Google Agent Development Kit (ADK) for multi-step assistive features |
| BI | Manna's Core itself; we are customer zero |

## Working principles

- **Customer zero.** Internal analytics runs on Manna's Core. If Core is painful
  for us, that is a product bug, and we file it as one.
- **Offline before online.** No model reaches production without an offline
  evaluation on a held-out period and a documented baseline it beats.
- **Every model has an owner and a rollback.** If it cannot be turned off
  quickly, it is not ready.
- **Evaluation is code.** Prompt and retrieval changes ship with an updated eval
  set and a scorecard, not a vibe check.
- **No customer data in prompts sent to third-party endpoints** unless the model
  is covered by an approved data processing agreement. Check the current approved
  list with Legal before wiring up a new provider.

## Model and data governance

- Training data lineage is recorded in the model card for every production model.
- Customer data used for model training requires an explicit contractual basis.
  Analytics Engineering maintains the allow-list of datasets cleared for training.
- Personal data is pseudonymised before it reaches any training pipeline.
- Model cards are mandatory and are reviewed by the Risk & Compliance Board for
  anything customer-facing.

## Your first 30 days in Data & AI

1. **Week 1:** Access to the warehouse (read-only to start), dbt repo, MLflow,
   and the managed notebook environment. Complete the data handling module of
   security training before any access to customer-derived tables is granted.
2. **Week 1:** Read the metric definitions for the ten exec-reviewed metrics.
   Reproduce two of them yourself in SQL and compare against Core.
3. **Week 2:** Ship one dbt model change or one analysis with a written summary.
4. **Week 3:** Sit in on a Metric Review and a Product Council.
5. **Week 4:** Present a small finding at your team meeting. The point is
   practising "show the working", not the size of the finding.

## Key contacts

| Need | Where |
| --- | --- |
| Warehouse access or a permissions issue | `#ask-data-platform` |
| dbt model or metric definition question | `#analytics-engineering` |
| Model serving or training infrastructure | `#ml-platform` |
| GenAI, RAG or agent questions | `#genai` |
| Whether a dataset may be used for training | Ask Legal via `#ask-legal` |
