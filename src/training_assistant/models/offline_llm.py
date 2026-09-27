"""A deterministic ``BaseLlm`` that lets the whole agent graph run without a key.

Why this exists
---------------
Without credentials, an LLM-backed project is a pile of unrunnable code. A
reviewer cannot see the router route, cannot see retrieval ground an answer, and
cannot reproduce a single result. That is a poor place for a project to be.

``OfflineDemoLlm`` plugs into the *same* ADK graph as Gemini and speaks the same
protocol: it emits a ``transfer_to_agent`` function call when it is the router,
emits a retrieval function call when it is a specialist, and produces a final
text answer once tool results come back. Every piece of orchestration under test
is therefore the real thing. Only the language generation is swapped.

What it is not
--------------
It does not write prose. Its answers are **extractive**: it selects the
sentences and table rows from the retrieved passages that best match the
question, and labels them with their citations. Answers are therefore blunter
than Gemini's and are always tagged with :data:`OFFLINE_NOTICE` so nobody
mistakes an extract for generated text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Final, Iterable

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from ..knowledge.text_utils import tokenize
from ..routing.classifier import ROUTE_TO_AGENT
from ..routing.resolver import resolve_route

logger = logging.getLogger(__name__)

OFFLINE_MODEL_NAME: Final[str] = "offline-deterministic"

OFFLINE_NOTICE: Final[str] = (
    "_Offline mode: this answer was assembled directly from the retrieved "
    "passages, not written by a language model. Set `GOOGLE_API_KEY` in `.env` "
    "for Gemini-generated answers._"
)

TRANSFER_TOOL: Final[str] = "transfer_to_agent"

#: Roles an instance can play. One instance is created per agent.
ROUTER_ROLE: Final[str] = "router"
GENERAL_ROLE: Final[str] = "general"
RETRIEVAL_ROLES: Final[frozenset[str]] = frozenset({"company", "role", "admin"})

#: How many extracted lines to include in an answer.
MAX_EXTRACTED_LINES: Final[int] = 5

#: A line must overlap the question by at least this much (after rank weighting)
#: to be quoted. Set high enough that a weakly-related passage cannot contribute
#: a stray line: quoting a password rule in answer to a mileage question makes
#: the whole answer look untrustworthy, even though the top line was right.
MIN_LINE_SCORE: Final[float] = 0.12

#: How sharply to prefer lines from higher-ranked passages.
RANK_DECAY: Final[float] = 0.25

#: Only the top few passages may contribute lines, however well a line scores.
MAX_PASSAGES_QUOTED: Final[int] = 3

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_PASSAGE_HEADER_RE = re.compile(r"^\[(\d+)\]\s*(.+?)\s*$")

#: Where to send people when the documents cannot answer them.
HUMAN_ROUTES: Final[dict[str, str]] = {
    "company": "your onboarding buddy, or `#ask-people-ops`",
    "role": "your functional manager, or your People Partner",
    "admin": "`#ask-people-ops` for HR matters, `#ask-finance` for expenses, "
    "`#ask-it` for access, or `#ask-security` for security",
}


# --------------------------------------------------------------------------- #
# Helpers for reading an LlmRequest
# --------------------------------------------------------------------------- #


#: ADK hands a transferred-to agent a transcript of what the previous agent did,
#: and it arrives as ``role="user"`` **text** parts. Those parts must not be
#: mistaken for the employee's question: doing so sends the transfer receipt to
#: the retriever instead of the actual query, which silently destroys retrieval
#: quality while everything still appears to work.
_INJECTED_MARKERS: Final[tuple[str, ...]] = (
    "<<<BEGIN_QUOTED_AGENT_CONTENT>>>",
    "<<<END_QUOTED_AGENT_CONTENT>>>",
    "For context: below is a transcript",
)

_AGENT_TRANSCRIPT_RE = re.compile(r"^\[[\w.-]+\]\s+.*\btool\b", re.DOTALL)


def _is_injected(text: str) -> bool:
    """True when a user-role text part is ADK scaffolding, not a real utterance."""
    if any(marker in text for marker in _INJECTED_MARKERS):
        return True
    return bool(_AGENT_TRANSCRIPT_RE.match(text.strip()))


def _iter_parts(llm_request: LlmRequest) -> Iterable[tuple[str, types.Part]]:
    """Yield ``(role, part)`` for every part in the conversation."""
    for content in llm_request.contents or []:
        for part in content.parts or []:
            yield (content.role or ""), part


@dataclass(slots=True)
class _TurnView:
    """The current turn, isolated from earlier conversation history.

    ``question`` is the employee's latest genuine utterance and ``tail`` holds
    only the parts that came after it. Scoping to the turn is what makes
    multi-turn conversations work: without it, a function response from an
    earlier question looks like the answer to the current one, and the agent
    replies using stale passages.
    """

    question: str = ""
    tail: list[types.Part] = field(default_factory=list)

    @property
    def function_responses(self) -> list[types.FunctionResponse]:
        return [p.function_response for p in self.tail if p.function_response is not None]

    @property
    def function_calls(self) -> list[types.FunctionCall]:
        return [p.function_call for p in self.tail if p.function_call is not None]


def _view_current_turn(llm_request: LlmRequest) -> _TurnView:
    """Split the conversation into the current question and what followed it."""
    flat = list(_iter_parts(llm_request))

    question = ""
    boundary = -1
    for index, (role, part) in enumerate(flat):
        text = (part.text or "").strip()
        if role == "user" and text and not _is_injected(text):
            question = text
            boundary = index

    return _TurnView(
        question=question,
        tail=[part for _role, part in flat[boundary + 1 :]],
    )


def _retrieval_tool_name(llm_request: LlmRequest) -> str | None:
    """The specialist's own retrieval tool, discovered rather than hardcoded."""
    for name in sorted((llm_request.tools_dict or {}).keys()):
        if name != TRANSFER_TOOL:
            return name
    return None


def _usage(prompt_chars: int, output_chars: int) -> types.GenerateContentResponseUsageMetadata:
    """Approximate token accounting.

    ADK logs a warning when usage metadata is absent, and a trace with plausible
    numbers is more useful than one with none. The 4-chars-per-token rule is a
    rough industry approximation and is only ever used for display.
    """
    prompt_tokens = max(1, prompt_chars // 4)
    output_tokens = max(1, output_chars // 4)
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
        total_token_count=prompt_tokens + output_tokens,
    )


def _request_size(llm_request: LlmRequest) -> int:
    return sum(len(part.text or "") for _role, part in _iter_parts(llm_request))


# --------------------------------------------------------------------------- #
# Extractive answer composition
# --------------------------------------------------------------------------- #


def _split_passages(passages: str) -> list[tuple[int, str, str]]:
    """Parse the retriever's numbered context block.

    Returns ``(number, citation, body)`` triples. The format produced by
    ``Retriever.format_context`` is the contract being parsed here.
    """
    parsed: list[tuple[int, str, str]] = []
    for block in passages.split("\n\n---\n\n"):
        block = block.strip()
        if not block:
            continue
        head, _, body = block.partition("\n")
        match = _PASSAGE_HEADER_RE.match(head.strip())
        if match:
            citation = match.group(2)
            # Strip the trailing "(policy vX) (last reviewed ...)" annotations.
            citation = re.sub(r"\s*\((?:policy|last reviewed)[^)]*\)", "", citation)
            parsed.append((int(match.group(1)), citation.strip(), body.strip()))
        else:
            parsed.append((len(parsed) + 1, "", block))
    return parsed


_BULLET_RE = re.compile(r"^(?:[-*+]|\d+\.)\s+")


def _parent_path(citation: str) -> str:
    """The citation with its final heading removed, for sibling detection.

    ``"Offices > Office locations > Berlin (EMEA)"`` becomes
    ``"Offices > Office locations"``, and two passages sharing that parent are
    sibling subsections of one section.

    Returns ``""`` when the parent would be nothing more than the document
    title. At that depth "sibling" degenerates into "any section of the same
    document", which is far too broad: every section of the leave policy would
    qualify, and their leading lines crowd out the scored line that actually
    answers the question.
    """
    head, separator, _tail = citation.rpartition(" > ")
    if not separator or " > " not in head:
        return ""
    return head


def _candidate_lines(body: str) -> list[str]:
    """Break a passage into quotable units.

    The source Markdown is hard-wrapped, so physical lines are *not* quotable
    units: quoting them directly produces fragments cut mid-sentence. Wrapped
    prose is therefore reflowed into paragraphs before sentence splitting, and
    indented bullet continuations are joined onto their bullet.

    Table rows and bullets are kept whole, because a table row is the atomic
    unit of meaning in a policy document.
    """
    units: list[str] = []
    paragraph: list[str] = []
    bullet: list[str] | None = None

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(paragraph)
            units.extend(
                s.strip() for s in _SENTENCE_SPLIT_RE.split(joined) if s.strip()
            )
            paragraph.clear()

    def flush_bullet() -> None:
        nonlocal bullet
        if bullet:
            units.append(" ".join(bullet))
        bullet = None

    for raw in body.splitlines():
        stripped = raw.strip()

        # Blank line, or a table separator such as "|---|---|".
        if not stripped or set(stripped) <= set("|-: "):
            flush_bullet()
            flush_paragraph()
            continue

        # Headings are already captured in the citation, and blockquotes are
        # editorial asides (notes, disclaimers) rather than answer content.
        if stripped.startswith("#") or stripped.startswith(">"):
            flush_bullet()
            flush_paragraph()
            continue

        if stripped.startswith("|"):
            flush_bullet()
            flush_paragraph()
            units.append(stripped)
            continue

        if _BULLET_RE.match(stripped):
            flush_bullet()
            flush_paragraph()
            bullet = [stripped]
            continue

        # An indented line directly after a bullet continues that bullet.
        if bullet is not None and raw[:1] in (" ", "\t"):
            bullet.append(stripped)
            continue

        flush_bullet()
        paragraph.append(stripped)

    flush_bullet()
    flush_paragraph()
    return units


def _tidy(line: str) -> str:
    """Make a raw Markdown line readable in a chat bubble."""
    line = line.strip()
    if line.startswith("|"):
        cells = [c.strip() for c in line.strip("|").split("|")]
        cells = [c for c in cells if c]
        return " - ".join(cells)
    line = re.sub(r"^[-*+]\s+", "", line)
    line = re.sub(r"^\d+\.\s+", "", line)
    return line.strip()


#: Questions whose answer is a figure. For these, a line containing digits is
#: far more likely to be the answer than one that merely repeats the question's
#: wording, which is exactly what a table's header row does.
_QUANTITY_QUESTION_RE = re.compile(
    r"\b(how much|how many|how long|how often|cap|caps|rate|rates|limit|limits"
    r"|allowance|budget|entitlement|threshold|deadline|notice period|percentage"
    r"|amount|per night|per day|per person|per month|per year|per kilometre"
    r"|salary|cost|fee|days|weeks|months|hours)\b"
)
_HAS_DIGIT_RE = re.compile(r"\d")

#: Multipliers applied when the question is quantitative.
NUMERIC_LINE_BOOST: Final[float] = 1.6
NON_NUMERIC_LINE_PENALTY: Final[float] = 0.75


def _term_weights(tokens: set[str]) -> dict[str, float]:
    """Corpus rarity for each query token, normalised to a mean of roughly 1.

    Falls back to flat weights when the index is unavailable, so answer
    composition never depends on retrieval internals being reachable.
    """
    try:
        from ..knowledge.retriever import get_retriever

        embedder = get_retriever().embedder
        weights = {token: max(0.1, embedder.term_weight(token)) for token in tokens}
    except Exception:  # pragma: no cover - degrade to unweighted scoring
        logger.debug("Term weights unavailable; scoring lines unweighted.", exc_info=True)
        return {token: 1.0 for token in tokens}

    if not weights:
        return {}
    mean = sum(weights.values()) / len(weights)
    if mean <= 0:
        return {token: 1.0 for token in tokens}
    return {token: value / mean for token, value in weights.items()}


def _score_line(
    line: str,
    query_tokens: set[str],
    weights: dict[str, float],
    quantitative: bool,
) -> float:
    """How well one line answers the question.

    Overlap is weighted by term rarity rather than counted flatly. Without that,
    a table's header row ("Location | Cap per night") outscores the row holding
    the answer ("Mumbai | INR 9,000"), because the header repeats the common
    words from the question while the answer carries the one rare word that
    matters.
    """
    if not query_tokens:
        return 0.0
    line_tokens = set(tokenize(line))
    if not line_tokens:
        return 0.0

    overlap = query_tokens & line_tokens
    if not overlap:
        return 0.0

    gained = sum(weights.get(token, 1.0) for token in overlap)
    total = sum(weights.get(token, 1.0) for token in query_tokens)
    coverage = gained / total if total else 0.0

    # Mildly penalise very long lines so a whole table does not win purely by
    # containing every word.
    brevity = 1.0 / (1.0 + len(line_tokens) / 40.0)
    score = coverage * (0.7 + 0.3 * brevity)

    if quantitative:
        score *= (
            NUMERIC_LINE_BOOST
            if _HAS_DIGIT_RE.search(line)
            else NON_NUMERIC_LINE_PENALTY
        )

    return score


def _compose_grounded_answer(question: str, payload: dict[str, Any]) -> str:
    """Build an extractive, cited answer from a retrieval tool result."""
    passages = str(payload.get("passages") or "")
    parsed = _split_passages(passages)
    query_tokens = set(tokenize(question, drop_query_noise=True))
    weights = _term_weights(query_tokens)
    quantitative = bool(_QUANTITY_QUESTION_RE.search(question.lower()))

    scored: list[tuple[float, int, str, str]] = []
    for number, citation, body in parsed:
        if number > MAX_PASSAGES_QUOTED:
            continue
        # Trust the retriever's ordering: a line in the top-ranked passage is
        # more likely to be the answer than an equally-worded line further down.
        rank_weight = 1.0 / (1.0 + RANK_DECAY * max(0, number - 1))
        for line in _candidate_lines(body):
            score = _score_line(line, query_tokens, weights, quantitative) * rank_weight
            if score >= MIN_LINE_SCORE:
                scored.append((score, number, citation, _tidy(line)))

    scored.sort(key=lambda item: item[0], reverse=True)

    # Keep the best lines, de-duplicated, preserving passage order for reading.
    chosen: list[tuple[int, str, str]] = []
    seen: set[str] = set()

    def _add(number: int, citation: str, line: str) -> bool:
        """Append a line unless it duplicates one already chosen or is empty."""
        key = line.lower()[:120]
        if not line or key in seen or len(chosen) >= MAX_EXTRACTED_LINES:
            return False
        seen.add(key)
        chosen.append((number, citation, line))
        return True

    # Guarantee content from the top-ranked passage first, so the cap can never
    # squeeze it out. A question like "where are our offices?" is answered by an
    # address, and an address shares no words with the question, so pure
    # word-overlap scoring would drop the one line the employee actually needs.
    if parsed:
        top_number, top_citation, top_body = parsed[0]

        for line in [ln for _s, num, _c, ln in scored if num == top_number][:2]:
            _add(top_number, top_citation, line)

        for line in _candidate_lines(top_body)[:2]:
            if sum(1 for number, _c, _l in chosen if number == top_number) >= 2:
                break
            _add(top_number, top_citation, _tidy(line))

        # Sibling subsections of the same parent heading are usually parts of one
        # enumerated answer: "Office locations > Bengaluru", "> Austin",
        # "> Berlin". Retrieval surfaces them all, but only one of them tends to
        # share wording with the question, so include a leading line from each
        # sibling rather than answering with a third of the list.
        # Siblings are scanned across *all* retrieved passages, not just the
        # first MAX_PASSAGES_QUOTED. A sibling is a completion of the top
        # passage rather than an independent result, and the one that ranked
        # lowest is often the one the enumeration is missing: the Bengaluru
        # office chunk is the longest of the three and lands last, so capping
        # the scan produced an answer naming two of three offices.
        top_parent = _parent_path(top_citation)
        if top_parent:
            for number, citation, body in parsed[1:]:
                if _parent_path(citation) != top_parent:
                    continue
                for line in _candidate_lines(body)[:1]:
                    _add(number, citation, _tidy(line))

    for _score, number, citation, line in scored:
        if not _add(number, citation, line) and len(chosen) >= MAX_EXTRACTED_LINES:
            break

    if not chosen:
        return _compose_no_match_answer(str(payload.get("domain") or ""))

    chosen.sort(key=lambda item: item[0])

    citations = {number: citation for number, citation, _line in chosen if citation}

    lines_out = ["Here is what the onboarding documents say:", ""]
    for number, _citation, line in chosen:
        lines_out.append(f"- {line} [{number}]")

    if citations:
        lines_out.append("")
        lines_out.append("**Sources**")
        for number in sorted(citations):
            lines_out.append(f"- [{number}] {citations[number]}")

    lines_out.append("")
    lines_out.append(OFFLINE_NOTICE)
    return "\n".join(lines_out)


def _compose_no_match_answer(domain: str) -> str:
    """Honest answer for "retrieval found nothing"."""
    route = HUMAN_ROUTES.get(domain, "your onboarding buddy")
    return (
        "I could not find anything about that in the "
        f"{_DOMAIN_PHRASES.get(domain, 'company')} documents I have access to, "
        "so I would rather say so than guess.\n\n"
        f"The best next step is to ask {route}.\n\n" + OFFLINE_NOTICE
    )


_DOMAIN_PHRASES: Final[dict[str, str]] = {
    "company": "general company",
    "role": "role and team",
    "admin": "policy and administration",
}


def _compose_error_answer(payload: dict[str, Any]) -> str:
    message = str(payload.get("message") or "The knowledge base is unavailable.")
    return (
        f"I hit a problem searching the knowledge base: {message}\n\n"
        "Please tell your onboarding buddy, and try again once it is resolved.\n\n"
        + OFFLINE_NOTICE
    )


# --------------------------------------------------------------------------- #
# Fallback-path answers (no retrieval available)
# --------------------------------------------------------------------------- #

_IDENTITY_PATTERNS = (
    "who are you",
    "what are you",
    "what can you do",
    "what can you help",
    "how do you work",
    "are you an ai",
    "are you a bot",
    "what do you do",
)

_GREETING_PATTERNS = ("hello", "hi ", "hey", "good morning", "good afternoon", "namaste")

_IDENTITY_ANSWER = """\
I am the Manna's Analytics Training Assistant. I help new joiners find answers
without hunting through documents or waiting on a colleague.

I classify your question and send it to one of four places:

- **General company knowledge** - what Manna's does, our history, mission and
  values, offices and hybrid working, org structure and leadership, products,
  customers, and internal acronyms.
- **Role and team documents** - what each team and role does, tech stacks, the
  career grade ladder, promotion and performance process, on-call.
- **Policy and administration** - leave, expenses and travel, payroll and
  benefits, notice periods, code of conduct, security and compliance, mandatory
  training, system access, and internal tools.
- **A direct answer** - for general questions with no Manna's-specific answer.

Try asking:
- "How many days of annual leave do I get?"
- "When do new engineers join the on-call rotation?"
- "What does AR mean in a calendar invite?"

I always tell you which documents an answer came from, and I will say so plainly
when the documents do not cover something."""

_OFFLINE_GENERAL_ANSWER = """\
That looks like a general question rather than a Manna's-specific one, so there is
nothing in the company knowledge base to ground an answer in.

This assistant is running in **offline mode**, where the fallback path cannot
generate free-form answers: there is no language model behind it, only
deterministic retrieval over the company documents. Add a Gemini API key as
`GOOGLE_API_KEY` in `.env` and the general path will answer questions like this
properly.

What I can answer right now, from the company documents:
- Company information: history, mission and values, offices, org structure,
  products, customers, internal acronyms.
- Roles and teams: team charters, role expectations, grades and promotion,
  on-call, tech stacks.
- Policy and administration: leave, expenses, payroll and benefits, conduct,
  security, training, system access, internal tools."""


def _compose_general_answer(question: str) -> str:
    lowered = f" {question.lower().strip()} "

    if any(pattern in lowered for pattern in _IDENTITY_PATTERNS):
        return _IDENTITY_ANSWER

    words = question.strip().split()
    if len(words) <= 4 and any(lowered.strip().startswith(g.strip()) for g in _GREETING_PATTERNS):
        return (
            "Hello, and welcome to Manna's Analytics.\n\n"
            + _IDENTITY_ANSWER.split("\n\n", 1)[1]
        )

    return _OFFLINE_GENERAL_ANSWER


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class OfflineDemoLlm(BaseLlm):
    """Deterministic stand-in for Gemini, one instance per agent.

    Attributes:
        role: Which agent this instance backs. ``"router"`` emits transfers,
            ``"company"``/``"role"``/``"admin"`` drive retrieval then answer
            extractively, and ``"general"`` answers the fallback path.
    """

    role: str = ROUTER_ROLE

    # Pydantic v2 model config is inherited from BaseLlm; only a field is added.

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        turn = _view_current_turn(llm_request)
        prompt_chars = _request_size(llm_request)

        if self.role == ROUTER_ROLE:
            yield self._route(turn.question, prompt_chars)
            return

        if self.role in RETRIEVAL_ROLES:
            async for response in self._retrieve_then_answer(
                llm_request, turn, prompt_chars
            ):
                yield response
            return

        yield self._text_response(_compose_general_answer(turn.question), prompt_chars)

    # ------------------------------ router ------------------------------ #

    def _route(self, question: str, prompt_chars: int) -> LlmResponse:
        decision = resolve_route(question)
        target = ROUTE_TO_AGENT[decision.route]
        logger.info(
            "Offline router: %r -> %s (decided by %s)",
            question[:70],
            target,
            decision.decided_by,
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name=TRANSFER_TOOL,
                            args={
                                "agent_name": target,
                                "transfer_reason": decision.reason,
                            },
                        )
                    )
                ],
            ),
            usage_metadata=_usage(prompt_chars, len(target)),
        )

    # --------------------------- specialists ---------------------------- #

    async def _retrieve_then_answer(
        self, llm_request: LlmRequest, turn: _TurnView, prompt_chars: int
    ) -> AsyncGenerator[LlmResponse, None]:
        tool_name = _retrieval_tool_name(llm_request)
        responses = turn.function_responses

        # Nothing has come back yet in this turn, so search.
        if tool_name and not responses and not turn.function_calls:
            logger.info(
                "Offline %s agent: calling %s with %r",
                self.role,
                tool_name,
                turn.question[:70],
            )
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name=tool_name, args={"query": turn.question}
                            )
                        )
                    ],
                ),
                usage_metadata=_usage(prompt_chars, len(turn.question)),
            )
            return

        if not responses:
            # Either there is no tool, or a call was made and produced nothing.
            # Answer honestly rather than looping.
            logger.warning(
                "Offline %s agent has no retrieval result to answer from.", self.role
            )
            yield self._text_response(_compose_no_match_answer(self.role), prompt_chars)
            return

        payload = responses[-1].response or {}
        if not isinstance(payload, dict):
            payload = {"status": "error", "message": str(payload)}

        status = str(payload.get("status", "error"))

        if status == "ok":
            answer = _compose_grounded_answer(turn.question, payload)
        elif status == "no_match":
            answer = _compose_no_match_answer(str(payload.get("domain") or self.role))
        else:
            answer = _compose_error_answer(payload)

        yield self._text_response(answer, prompt_chars)

    # ------------------------------ shared ------------------------------ #

    def _text_response(self, text: str, prompt_chars: int) -> LlmResponse:
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            usage_metadata=_usage(prompt_chars, len(text)),
        )
