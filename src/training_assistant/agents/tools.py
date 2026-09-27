"""ADK function tools that expose the knowledge base to the specialist agents.

Each specialist gets exactly one tool, hard-wired to its own domain. The domain
is not a parameter the model can choose, which is what makes the router's
decision binding: a mis-scoped question cannot quietly reach into another
collection.

The docstrings matter. ADK turns them into the function declarations the model
sees, so they are written for a model deciding whether and how to call the tool,
not for a developer reading the source.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from google.adk.tools import ToolContext

from ..config import get_settings
from ..knowledge.retriever import RetrievedChunk, get_retriever

logger = logging.getLogger(__name__)

#: Session-state keys. The UI and the CLI read these to show which sources
#: backed an answer, so they form a small contract with the presentation layer.
STATE_SOURCES = "retrieved_sources"
STATE_DOMAIN = "retrieved_domain"
STATE_QUERY = "retrieval_query"

#: Wall time spent inside the retriever itself, excluding ADK tool-dispatch
#: overhead. Reported separately from the event-stream retrieval span so the
#: latency benchmark can distinguish "search is slow" from "the framework round
#: trip is slow".
STATE_SEARCH_SECONDS = "retrieval_search_seconds"


def _record(
    tool_context: ToolContext | None,
    domain: str,
    query: str,
    results: list[RetrievedChunk],
    search_seconds: float = 0.0,
) -> None:
    """Publish retrieval provenance into session state."""
    if tool_context is None:
        return
    try:
        tool_context.state[STATE_DOMAIN] = domain
        tool_context.state[STATE_QUERY] = query
        tool_context.state[STATE_SOURCES] = [r.to_dict() for r in results]
        tool_context.state[STATE_SEARCH_SECONDS] = round(search_seconds, 6)
    except Exception:  # pragma: no cover - state is best-effort telemetry
        logger.debug("Could not write retrieval state", exc_info=True)


def _search(domain: str, query: str, tool_context: ToolContext | None) -> dict[str, Any]:
    """Shared retrieval body for all three domain tools."""
    query = (query or "").strip()
    if not query:
        return {
            "status": "invalid_query",
            "domain": domain,
            "message": "The search query was empty. Call the tool again with the employee's question.",
            "passages": "",
            "sources": [],
        }

    settings = get_settings()

    try:
        retriever = get_retriever()
    except FileNotFoundError as exc:
        logger.error("Knowledge index missing: %s", exc)
        return {
            "status": "index_missing",
            "domain": domain,
            "message": (
                "The knowledge index has not been built, so no company "
                "documents can be searched. Tell the employee the assistant is "
                "not fully set up and to contact their onboarding buddy."
            ),
            "passages": "",
            "sources": [],
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Retriever failed to load")
        return {
            "status": "error",
            "domain": domain,
            "message": f"The knowledge base could not be loaded: {exc}",
            "passages": "",
            "sources": [],
        }

    search_started = time.perf_counter()
    results = retriever.search(domain, query, top_k=settings.top_k)
    search_seconds = time.perf_counter() - search_started

    _record(tool_context, domain, query, results, search_seconds)

    if not results:
        logger.info("No match in domain=%s for query=%r", domain, query[:80])
        return {
            "status": "no_match",
            "domain": domain,
            "query": query,
            "message": (
                "No passage in this knowledge base was relevant enough to answer "
                "the question. Do not answer from general knowledge. Tell the "
                "employee the documents you have do not cover it and point them "
                "to a human."
            ),
            "passages": "",
            "sources": [],
        }

    return {
        "status": "ok",
        "domain": domain,
        "query": query,
        "passage_count": len(results),
        "passages": retriever.format_context(results),
        "sources": [
            {"number": r.rank, "citation": r.citation, "relevance": round(r.score, 3)}
            for r in results
        ],
    }


# --------------------------------------------------------------------------- #
# The three domain tools
# --------------------------------------------------------------------------- #


def search_company_knowledge(query: str, tool_context: ToolContext) -> dict[str, Any]:
    """Search the general company knowledge base.

    Covers what Manna's Analytics does, its history and founders, mission, vision
    and values, offices and the hybrid working model, the executive team and how
    the company is organised into functions, groups and pods, the product
    modules, customers and competitors, headline business figures, company
    rituals, and the internal glossary of terms and acronyms.

    Args:
        query: A focused search phrase built from the employee's question, with
            conversational filler removed. For example, given "hey, I keep
            hearing people say AR in meetings, what is that?", pass
            "AR acronym meaning Architecture Review".

    Returns:
        A dict with a ``status`` field. When ``status`` is ``"ok"``, ``passages``
        holds numbered source passages to answer from and cite. When ``status``
        is ``"no_match"``, nothing relevant was found and you must say so rather
        than answering from general knowledge.
    """
    return _search("company", query, tool_context)


def search_role_knowledge(query: str, tool_context: ToolContext) -> dict[str, Any]:
    """Search the role, team and career knowledge base.

    Covers team charters for every function and who leads them, role definitions
    and the expectations attached to each grade, technology stacks and ways of
    working, the career grade ladder from L2 to L8 and the parallel management
    track, promotion and performance review process, on-call rotations, quota and
    commission mechanics for quota-carrying roles, and function-specific
    first-30-day plans for new joiners.

    Args:
        query: A focused search phrase built from the employee's question, with
            conversational filler removed. For example, given "I'm a bit nervous,
            when will I have to start doing on-call?", pass "on-call rotation new
            engineer eligibility 90 days".

    Returns:
        A dict with a ``status`` field. When ``status`` is ``"ok"``, ``passages``
        holds numbered source passages to answer from and cite. When ``status``
        is ``"no_match"``, nothing relevant was found and you must say so rather
        than answering from general knowledge.
    """
    return _search("role", query, tool_context)


def search_policy_knowledge(query: str, tool_context: ToolContext) -> dict[str, Any]:
    """Search the policy and administration knowledge base.

    Covers leave and time off, expenses, reimbursement and travel including caps
    and per-diems, payroll, salary structure, tax declarations, benefits and
    insurance, notice periods, probation and the exit process, the code of
    conduct including conflicts of interest and harassment reporting routes,
    compliance and information security including data handling and approved AI
    tools, mandatory training prerequisites, the onboarding checklist, how to
    request system access, and internal tools such as Workday, Okta, Expensify
    and Slack channels.

    Args:
        query: A focused search phrase built from the employee's question, with
            conversational filler removed. For example, given "sorry if this is
            obvious but do the leave days I didn't use just disappear in April?",
            pass "privilege leave carry forward expiry cap".

    Returns:
        A dict with a ``status`` field. When ``status`` is ``"ok"``, ``passages``
        holds numbered source passages to answer from and cite, including any
        policy version and review date. When ``status`` is ``"no_match"``,
        nothing relevant was found and you must say so rather than answering from
        general knowledge.
    """
    return _search("admin", query, tool_context)


#: Route name -> the tool that serves it. ``general`` has no tool by design.
DOMAIN_TOOLS = {
    "company": search_company_knowledge,
    "role": search_role_knowledge,
    "admin": search_policy_knowledge,
}

#: Tool name -> route, used by the offline model to work out which domain a
#: function call belongs to.
TOOL_NAME_TO_DOMAIN = {fn.__name__: domain for domain, fn in DOMAIN_TOOLS.items()}
