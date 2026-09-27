"""AI Training Assistant for New Employees.

An agentic onboarding assistant built on the Google Agent Development Kit
(ADK). A root *router* agent classifies each employee question and delegates it
to one of four paths:

===============  ==========================================================
Route            Handling
===============  ==========================================================
``company``      RAG over company overview / mission / org / product docs
``role``         RAG over role, team and career-framework docs
``admin``        RAG over HR, expense, leave, compliance and tooling policy
``general``      Direct LLM answer, no retrieval (fallback path)
===============  ==========================================================

The public entry point is :class:`training_assistant.assistant.TrainingAssistant`.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
