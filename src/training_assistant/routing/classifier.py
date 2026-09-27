"""Deterministic, rule-based query classifier.

This is the *decision node* of the system in its explainable form. It serves
three purposes:

1. **It drives routing in offline mode**, where there is no LLM to reason with,
   so the whole agent graph stays exercisable without credentials.
2. **It is the baseline** the LLM router is measured against in
   ``scripts/evaluate_routing.py``. A semantic router that cannot beat a keyword
   baseline is not earning its latency.
3. **It is a safety net.** When the live LLM router returns an unrecognised
   destination, the orchestrator falls back to this classifier rather than
   failing the turn.

The scoring is intentionally simple and inspectable: weighted phrase and term
matches per route, normalised by query length, with a confidence floor below
which the query is treated as out-of-scope and sent down the ``general`` path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final

from ..knowledge.text_utils import tokenize

#: The four routes exposed by the decision node, in presentation order.
ROUTES: Final[tuple[str, ...]] = ("company", "role", "admin", "general")

#: Route name -> ADK agent name. The agent names are what ``transfer_to_agent``
#: uses, so these two vocabularies must stay in sync with ``agents/graph.py``.
ROUTE_TO_AGENT: Final[dict[str, str]] = {
    "company": "company_knowledge_agent",
    "role": "role_knowledge_agent",
    "admin": "policy_admin_agent",
    "general": "general_support_agent",
}

AGENT_TO_ROUTE: Final[dict[str, str]] = {v: k for k, v in ROUTE_TO_AGENT.items()}

# --------------------------------------------------------------------------- #
# Signals
#
# Multi-word phrases are matched on **word boundaries** against the lowercased
# question and carry more weight, because they are far less ambiguous than
# single terms. Boundaries matter: a plain substring test lets the phrase
# "what does a" fire inside "what does ARR stand for", which misroutes the
# question. Single terms are matched against the *stemmed* token stream, so
# "policies" matches "policy" and "claims" matches "claim".
#
# Terms that are genuinely ambiguous across domains are deliberately left out
# of the term lists and expressed only as phrases. "customer" and "office" are
# the two worst offenders: every function talks about customers, and "office"
# appears in "home office" (an expense matter) as often as in "our offices"
# (a company matter).
# --------------------------------------------------------------------------- #

PHRASE_WEIGHT: Final[float] = 3.0
TERM_WEIGHT: Final[float] = 1.0

#: Structural signal for a glossary lookup: "what does <ACRONYM> mean / stand
#: for". Acronyms live in the company glossary, and such questions frequently
#: contain no other routable keyword at all, so they need their own rule rather
#: than an ever-growing list of individual acronyms.
GLOSSARY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bwhat\s+(?:does|do|is|are)\b[^?]*\b[A-Z]{2,6}\b"),
    re.compile(r"\b[A-Z]{2,6}\b[^?]*\bstands?\s+for\b"),
)
GLOSSARY_WEIGHT: Final[float] = 3.0

#: When two routes tie on score, prefe
# r the one supported by more *phrase*
#: matches, since phrases are the less ambiguous evidence. If that still ties,
#: this order decides, so the classifier is never non-deterministic.
TIE_BREAK_ORDER: Final[tuple[str, ...]] = ("admin", "role", "company", "general")

#: Minimum normalised score required to commit to a knowledge-base route.
#: Below this the question is treated as out of scope for the corpus.
CONFIDENCE_FLOOR: Final[float] = 0.34

ROUTE_PHRASES: Final[dict[str, tuple[str, ...]]] = {
    "company": (
        "company overview", "about the company", "what does the company do",
        "our mission", "company mission", "company values", "core values",
        "vision", "founded", "founder", "co-founder",
        "head office", "headquarters", "our offices", "office locations",
        "where are our office", "which offices", "office location",
        "which office", "offices located", "other offices",
        "exec team", "executive team", "leadership team", "senior leadership",
        "org chart", "organisation structure", "organization structure",
        "reporting structure", "how are we organised", "how are we organized",
        "annual recurring revenue", "fiscal year", "financial year start",
        "net revenue retention", "funding", "series d", "how many employees",
        "our products", "product line", "modules we sell", "what do we sell",
        "reference customer", "public reference", "name a customer",
        "our customers", "competitor", "competition", "market",
        "all hands", "demo friday", "metric review", "onboarding circle",
        "founders' review", "retro roulette", "town hall",
        "glossary", "acronym", "abbreviation", "what does it stand for",
        "stand for", "stands for", "calendar invite", "internal shorthand",
        "which customers", "mention publicly", "publicly", "public reference",
        "collaboration hours", "hybrid working", "hybrid by default",
        "difference between a pod and a group", "what is a pod",
        "Manna's core", "Manna's signals", "Manna's embed",
    ),
    "role": (
        "my role", "my team", "job description", "role guide",
        "responsibilities of", "what does a", "what do they do",
        "team charter", "which team", "which teams", "team structure within",
        "teams sit", "teams inside", "teams within", "teams are inside",
        "teams in the", "sit inside",
        "career framework", "career ladder", "grade ladder", "career path",
        "career progression", "job level", "grade band",
        "get promoted", "promotion", "time in grade",
        "performance review", "performance rating", "rating scale",
        "review cycle", "self assessment", "peer feedback", "calibration",
        "individual contributor", "management track", "ic track",
        "functional manager", "pod lead", "reporting line",
        "on call", "on-call", "rotation", "shadow", "toil",
        "tech stack", "technology stack", "technologies we use",
        "programming language", "which framework",
        "engineering practice", "code review", "design doc",
        "architecture review", "trunk based",
        "software engineer", "staff engineer", "principal engineer",
        "engineering manager", "data scientist", "machine learning engineer",
        "analytics engineer", "research scientist", "ai engineer",
        "product manager", "product designer", "user researcher",
        "technical writer", "design system",
        "account executive", "solutions engineer", "sales development",
        "customer success manager", "technical account manager",
        "onboarding consultant", "support engineer", "revenue operations",
        "sales process", "deal stage", "quota", "commission plan",
        "ramp guarantee", "demo certification", "run a demo",
        "customer demo", "run a customer", "customer call",
        "who owns the customer", "customer relationship",
        "owns the relationship", "after the contract",
        "contract is signed", "post-sale", "post sale",
        "renewal", "expansion", "handover to", "book of accounts",
        "severity level", "response target", "support severity",
        "first 30 days", "first thirty days", "30-60-90",
        "learning budget for my role", "internal mobility",
        "what is expected of", "expectations at", "scope of the grade",
    ),
    "admin": (
        "annual leave", "privilege leave", "casual leave", "sick leave",
        "time off", "paid time off", "days off", "holiday entitlement",
        "public holiday", "floating holiday", "carry forward",
        "carry-forward", "encashment", "leave balance", "leave policy",
        "maternity", "paternity", "parental leave", "adoption leave",
        "bereavement", "sabbatical", "unpaid leave", "loss of pay",
        "medical certificate", "carer leave", "marriage leave",
        "expense claim", "expense report", "expense policy", "reimburse",
        "reimbursement", "claim back", "out of pocket",
        "per diem", "hotel cap", "accommodation cap", "nightly cap",
        "mileage", "travel policy", "book travel", "flight class",
        "travelperk", "expensify",
        "home office setup", "home office", "setting up my home",
        "internet allowance", "mobile allowance",
        "co-working", "wellness allowance", "learning budget",
        "customer entertainment", "gift to a customer",
        "payroll", "payslip", "pay date", "payday", "salary structure",
        "form 16", "tax declaration", "flexible benefit", "provident fund",
        "gratuity", "full and final", "final settlement",
        "health insurance", "medical insurance", "sum insured",
        "add a dependent", "add my parents", "dependent nomination",
        "enrolment window", "enrollment window", "top-up cover",
        "employee assistance", "health check", "creche", "referral bonus",
        "notice period", "resign", "resignation", "exit process",
        "probation", "confirmation of probation", "relieving letter",
        "employment verification", "salary certificate",
        "internal transfer", "working hours", "attendance",
        "code of conduct", "conflict of interest", "declare a conflict",
        "outside employment", "side project", "side consulting",
        "harassment", "discrimination", "bullying", "retaliation",
        "internal committee", "ethics line", "speak up", "whistleblow",
        "bribery", "improper payment", "facilitation payment",
        "insider", "non-public information", "competition law",
        "information security", "security policy", "data classification",
        "restricted data", "confidential data", "customer data",
        "phishing", "social engineering", "security incident",
        "report an incident", "lost laptop", "stolen device",
        "data subject request", "privacy", "personal data", "gdpr",
        "approved ai tool", "paste customer data", "use chatgpt",
        "mandatory training", "security training", "data handling training",
        "secure development training", "before production access",
        "onboarding checklist", "first day", "day one", "pre-boarding",
        "background verification", "documents to submit",
        "get access to", "request access", "access request",
        "provisioned", "okta", "single sign-on", "mfa", "1password",
        "slack channel", "which channel", "who do i contact",
        "who do i ask", "where do i ask",
        "book a desk", "book a meeting room", "book a room",
        "desk booking", "room booking", "workplace app", "badge",
        "visitor", "parking", "cafeteria",
        "new laptop", "broken laptop", "hardware", "peripheral",
        "buy software", "new software", "procurement", "vendor",
        "workday", "jira access", "github access",
    ),
    "general": (
        "what is retrieval augmented generation", "what is rag",
        "large language model", "what is an llm",
        "in python", "in sql", "in javascript",
        "list and a tuple", "sql join", "difference between let and const",
        "big o", "time complexity", "design pattern",
        "recommend a book", "book recommendation", "recommend a course",
        "suggest a book",
        "first impression", "imposter syndrome",
        "status update", "questions to ask", "one to one", "one-to-one",
        "prioritise when", "prioritize when", "time management",
        "stay motivated", "work life balance",
        "weather", "traffic", "restaurant", "joke", "recipe",
        "who are you", "what can you do", "what can you help",
        "how do you work", "are you an ai",
    ),
}

#: Conversational framing that suggests a general-knowledge request. These are
#: *weak* signals: "can you explain X" says nothing about whether X is a company
#: topic. They therefore carry a lower weight than domain phrases and, crucially,
#: are counted **at most once** however many of them fire.
#:
#: Both properties were added to fix a real misroute. With these at full phrase
#: weight and counted individually, "can you explain how promotions work here?"
#: scored 6 for general (three overlapping framings) against 3 for role, and a
#: clear role question was answered by the fallback path.
GENERAL_FRAMING_PHRASES: Final[tuple[str, ...]] = (
    "can you explain", "could you explain", "explain what",
    "explain the difference", "explain how", "explain in",
    "in simple terms", "in plain english", "layman",
    "high level overview of", "give me an overview of",
    "can you recommend", "could you recommend", "any recommendation",
    "tips for", "any tips", "any advice", "advice for", "how should i",
    "how do i get better at", "best way to learn", "what is the best way",
)

#: Weight for a conversational framing signal. Deliberately below
#: ``TERM_WEIGHT``, so even a single domain word outranks it: "give me an
#: overview of the company" must reach the company route, not the fallback.
#: A framing only decides the route when nothing else matched at all.
FRAMING_WEIGHT: Final[float] = 0.9

ROUTE_TERMS: Final[dict[str, tuple[str, ...]]] = {
    "company": (
        "company", "Manna's", "founded", "founder", "mission", "vision",
        "values", "culture", "ceo", "cto", "cpo", "cfo", "exec",
        "executive", "leadership", "headquarters", "office", "offices",
        "bengaluru", "austin", "berlin", "product", "products", "module",
        "headquarter",
        "competitor", "market", "revenue", "arr",
        "nrr", "fiscal", "glossary", "acronym", "org", "organisation",
        "organization", "function", "pod", "group", "ritual",
    ),
    "role": (
        "role", "roles", "team", "teams", "charter", "grade", "grades",
        "level", "ladder", "career", "promotion", "promoted",
        "performance", "rating", "review", "calibration", "competency",
        "engineer", "engineering", "developer", "scientist", "analyst",
        "designer", "researcher", "manager", "lead", "principal", "staff",
        "quota", "commission", "territory", "pipeline", "demo",
        "rotation", "oncall", "shadowing", "stack", "framework",
        "responsibility", "responsibilities", "expectation",
        "expectations", "scope", "mentor", "mentoring", "track",
        "severity", "escalate", "sdr", "csm", "tam",
    ),
    "admin": (
        "leave", "holiday", "vacation", "pto", "absence", "entitlement",
        "expense", "expenses", "reimbursement", "reimburse", "claim",
        "receipt", "invoice", "allowance", "cap", "diem", "mileage",
        "travel", "hotel", "flight", "payroll", "payslip", "salary",
        "pay", "tax", "insurance", "benefit", "benefits", "dependent",
        "pf", "gratuity", "bonus", "settlement", "notice", "resign",
        "resignation", "probation", "exit", "policy", "policies",
        "compliance", "conduct", "harassment", "discrimination",
        "bribery", "conflict", "confidential", "restricted",
        "security", "phishing", "incident", "breach", "privacy",
        "gdpr", "training", "mandatory", "access", "permission",
        "provisioning", "okta", "mfa", "password", "laptop", "device",
        # "booking" is deliberately absent: it stems to "book", which collides
        # with the noun ("recommend a good book"). The "book a room" / "desk
        # booking" phrases cover the administrative sense unambiguously.
        "badge", "desk", "room", "visitor", "procurement",
        "software", "tool", "tools", "workday", "expensify", "slack",
        "channel", "checklist", "onboarding", "document", "documents",
        "form", "submit", "deadline", "approve", "approval", "approver",
    ),
    "general": (
        # Note: interrogatives and filler verbs ("explain", "describe", "tell")
        # are stripped from queries by QUERY_NOISE, so they can never match as
        # terms. Those signals are expressed as phrases below instead.
        # "tip", "tips", "advice" and "explain" are intentionally absent: they
        # appear inside GENERAL_FRAMING_PHRASES, and listing them here as well
        # double-counted a single cue. "Any tips for claiming expenses?" then
        # scored 2.5 for general against 2 for admin.
        "python", "sql", "javascript", "typescript", "algorithm",
        "tutorial", "concept", "example", "generally",
        "motivation", "productivity",
        "weather", "joke", "recipe", "sport", "movie", "news",
        "rag", "llm", "embedding", "transformer", "prompt",
    ),
}


@dataclass(slots=True)
class RouteDecision:
    """The outcome of classifying one question."""

    route: str
    confidence: float
    scores: dict[str, float]
    matched: dict[str, list[str]] = field(default_factory=dict)
    reason: str = ""

    @property
    def agent_name(self) -> str:
        return ROUTE_TO_AGENT[self.route]

    @property
    def matched_signals(self) -> list[str]:
        """Signals that supported the chosen route."""
        return self.matched.get(self.route, [])

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "agent": self.agent_name,
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "matched_signals": self.matched_signals,
            "reason": self.reason,
        }


@lru_cache(maxsize=1)
def _compiled_phrases() -> dict[str, tuple[tuple[str, re.Pattern[str]], ...]]:
    """Compile each route's *multi-word* phrases into boundary-anchored patterns.

    Interior whitespace becomes ``\\s+`` so "on  call" and "on\\ncall" still
    match, and the whole phrase is wrapped in ``\\b`` so it cannot fire inside a
    longer word.
    """
    compiled: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {}
    for route, phrases in ROUTE_PHRASES.items():
        entries: list[tuple[str, re.Pattern[str]]] = []
        for phrase in phrases:
            if " " not in phrase:
                continue  # handled as a strong term, see _strong_terms
            body = r"\s+".join(re.escape(part) for part in phrase.split())
            prefix = r"\b" if phrase[:1].isalnum() else ""
            suffix = r"\b" if phrase[-1:].isalnum() else ""
            entries.append((phrase, re.compile(f"{prefix}{body}{suffix}")))
        compiled[route] = tuple(entries)
    return compiled


@lru_cache(maxsize=1)
def _compiled_framings() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile the conversational framing phrases."""
    entries: list[tuple[str, re.Pattern[str]]] = []
    for phrase in GENERAL_FRAMING_PHRASES:
        body = r"\s+".join(re.escape(part) for part in phrase.split())
        entries.append((phrase, re.compile(rf"\b{body}\b")))
    return tuple(entries)


@lru_cache(maxsize=1)
def _strong_terms() -> dict[str, dict[str, str]]:
    """Single-word entries from the phrase lists, matched by stem.

    A one-word "phrase" such as ``promotion`` must not be matched by regex:
    ``\\bpromotion\\b`` fails on "promotions", so "explain how promotions work"
    lost its only strong role signal and fell through to the general path.
    Matching these by stem instead gives them plural and tense handling while
    keeping the higher phrase weight they were listed for.

    Returns ``route -> {stem: original_word}``.
    """
    strong: dict[str, dict[str, str]] = {}
    for route, phrases in ROUTE_PHRASES.items():
        mapping: dict[str, str] = {}
        for phrase in phrases:
            if " " in phrase:
                continue
            for stem in tokenize(phrase):
                mapping.setdefault(stem, phrase)
        strong[route] = mapping
    return strong


@lru_cache(maxsize=1)
def _stemmed_terms() -> dict[str, frozenset[str]]:
    """Pre-stem every route's term list so lookups are a set intersection."""
    return {
        route: frozenset(t for term in terms for t in tokenize(term))
        for route, terms in ROUTE_TERMS.items()
    }


def _score_route(
    route: str, lowered: str, tokens: set[str]
) -> tuple[float, list[str], int]:
    """Score one route.

    Returns ``(score, matched_signals, strong_signal_count)``. The strong count
    is used only for tie-breaking, where evidence quality matters more than
    evidence quantity.
    """
    score = 0.0
    matched: list[str] = []
    strong_count = 0

    for phrase, pattern in _compiled_phrases()[route]:
        if pattern.search(lowered):
            score += PHRASE_WEIGHT
            matched.append(phrase)
            strong_count += 1

    strong = _strong_terms()[route]
    strong_hits = strong.keys() & tokens
    for stem in sorted(strong_hits):
        score += PHRASE_WEIGHT
        matched.append(strong[stem])
        strong_count += 1

    # Skip anything already counted at the higher strong-term weight.
    for term in sorted((_stemmed_terms()[route] & tokens) - strong_hits):
        score += TERM_WEIGHT
        matched.append(term)

    # Conversational framing counts once, and only towards the general route.
    if route == "general":
        for phrase, pattern in _compiled_framings():
            if pattern.search(lowered):
                score += FRAMING_WEIGHT
                matched.append(phrase)
                break

    return score, matched, strong_count


def classify(question: str) -> RouteDecision:
    """Classify ``question`` into one of :data:`ROUTES`.

    Returns a :class:`RouteDecision` carrying the winning route, a normalised
    confidence in ``[0, 1]``, the full score vector and the signals that fired,
    so the decision can be shown to the user rather than being a black box.
    """
    lowered = " ".join(question.lower().split())
    if not lowered:
        return RouteDecision(
            route="general",
            confidence=0.0,
            scores={r: 0.0 for r in ROUTES},
            reason="Empty question; defaulted to the general path.",
        )

    tokens = set(tokenize(question, drop_query_noise=True))

    raw: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    phrase_hits: dict[str, int] = {}
    for route in ROUTES:
        score, signals, strong_count = _score_route(route, lowered, tokens)
        raw[route] = score
        matched[route] = signals
        phrase_hits[route] = strong_count

    # Structural glossary signal, evaluated on the original casing.
    if any(pattern.search(question) for pattern in GLOSSARY_PATTERNS):
        raw["company"] += GLOSSARY_WEIGHT
        matched["company"].append("acronym lookup pattern")
        phrase_hits["company"] += 1

    total = sum(raw.values())
    scores = (
        {route: value / total for route, value in raw.items()}
        if total > 0
        else {route: 0.0 for route in ROUTES}
    )

    best = max(
        ROUTES,
        key=lambda r: (
            raw[r],
            phrase_hits[r],
            -TIE_BREAK_ORDER.index(r),
        ),
    )
    best_score = scores[best]

    # No signal at all, or an explicit generic-topic win: use the fallback path.
    if total == 0.0:
        return RouteDecision(
            route="general",
            confidence=0.0,
            scores=scores,
            matched=matched,
            reason=(
                "No company, role or policy signals matched, so the question "
                "was sent to the direct-LLM fallback path."
            ),
        )

    if best == "general":
        return RouteDecision(
            route="general",
            confidence=best_score,
            scores=scores,
            matched=matched,
            reason=(
                "The question looks like a general or technical query rather "
                f"than a Manna's-specific one (signals: "
                f"{', '.join(matched['general'][:3])})."
            ),
        )

    # A knowledge-base route won, but only weakly. Retrieval would most likely
    # return nothing useful, so prefer the honest fallback.
    knowledge_total = sum(raw[r] for r in ("company", "role", "admin"))
    knowledge_share = raw[best] / knowledge_total if knowledge_total else 0.0

    if knowledge_share < CONFIDENCE_FLOOR:
        return RouteDecision(
            route="general",
            confidence=best_score,
            scores=scores,
            matched=matched,
            reason=(
                f"Signals were split across knowledge domains "
                f"(best was {best} at {knowledge_share:.0%}), below the "
                f"{CONFIDENCE_FLOOR:.0%} confidence floor, so the question "
                "was sent to the fallback path."
            ),
        )

    return RouteDecision(
        route=best,
        confidence=best_score,
        scores=scores,
        matched=matched,
        reason=(
            f"Matched {len(matched[best])} {best} signal(s): "
            f"{', '.join(matched[best][:4])}."
        ),
    )


def route_to_agent_name(route: str) -> str:
    """Map a route name to its ADK agent name."""
    return ROUTE_TO_AGENT.get(route, ROUTE_TO_AGENT["general"])


def agent_name_to_route(agent_name: str) -> str:
    """Map an ADK agent name back to its route name."""
    return AGENT_TO_ROUTE.get(agent_name, "general")
