"""Orchestration: run the agent graph and return an inspectable answer.

This is the single public entry point used by the CLI, the Streamlit UI and the
evaluation harness. It owns three things the callers should not have to:

* **Session lifecycle.** One ADK session per conversation, so follow-up
  questions keep their context.
* **Trace capture.** ADK emits a stream of events; the interesting parts (which
  agent was chosen, what it searched for, which passages came back, how many
  model calls it took) are collected into an :class:`AssistantAnswer` so the UI
  can *show* the routing decision instead of just asserting one happened.
* **Graceful degradation.** A missing index or a failing model surfaces as an
  answer the employee can act on, not a stack trace in a chat window.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agents.graph import ROOT_AGENT_NAME, build_agent_graph
from .agents.tools import (
    STATE_DOMAIN,
    STATE_QUERY,
    STATE_SEARCH_SECONDS,
    STATE_SOURCES,
)
from .config import APP_NAME, DOMAIN_LABELS, Settings, get_settings
from .routing.classifier import RouteDecision, agent_name_to_route, classify

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SourceReference:
    """One retrieved passage, as shown to the user."""

    number: int
    citation: str
    score: float
    doc_id: str = ""
    source_path: str = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "citation": self.citation,
            "score": self.score,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
        }


@dataclass(slots=True)
class PhaseTimings:
    """Where the time went, broken down by pipeline phase.

    Boundaries are derived from the ADK event stream:

    ``routing``
        Question received until the router's transfer decision is observed.
        This is the router model call.
    ``retrieval``
        The specialist's tool call until its response arrives. Wall time,
        including ADK's tool-dispatch overhead.
    ``search``
        Time inside the retriever itself, reported by the tool. Always less
        than ``retrieval``; the difference is framework overhead.
    ``generation``
        Tool response until the final answer text. On the no-retrieval fallback
        path this runs from the transfer decision instead.
    ``overhead``
        Whatever the three phases do not account for, rather than silently
        dropping it. A large value here means time is going somewhere the
        decomposition does not model.
    """

    total_seconds: float = 0.0
    routing_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    search_seconds: float = 0.0
    generation_seconds: float = 0.0

    @property
    def overhead_seconds(self) -> float:
        accounted = self.routing_seconds + self.retrieval_seconds + self.generation_seconds
        return max(0.0, self.total_seconds - accounted)

    @property
    def framework_retrieval_overhead(self) -> float:
        """ADK tool-dispatch cost: measured retrieval span minus pure search."""
        return max(0.0, self.retrieval_seconds - self.search_seconds)

    def to_dict(self) -> dict[str, float]:
        return {
            "total_seconds": round(self.total_seconds, 4),
            "routing_seconds": round(self.routing_seconds, 4),
            "retrieval_seconds": round(self.retrieval_seconds, 4),
            "search_seconds": round(self.search_seconds, 4),
            "generation_seconds": round(self.generation_seconds, 4),
            "overhead_seconds": round(self.overhead_seconds, 4),
        }

    def format_line(self) -> str:
        return (
            f"total {self.total_seconds * 1000:7.1f}ms  "
            f"routing {self.routing_seconds * 1000:7.1f}ms  "
            f"retrieval {self.retrieval_seconds * 1000:7.1f}ms "
            f"(search {self.search_seconds * 1000:6.1f}ms)  "
            f"generation {self.generation_seconds * 1000:7.1f}ms"
        )


@dataclass(slots=True)
class AssistantAnswer:
    """An answer plus everything needed to explain how it was produced."""

    question: str
    answer: str
    route: str
    agent: str
    mode: str
    sources: list[SourceReference] = field(default_factory=list)
    retrieval_query: str = ""
    transfer_reason: str = ""
    rule_based_route: str = ""
    rule_based_confidence: float = 0.0
    rule_based_reason: str = ""
    llm_calls: int = 0
    tool_calls: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    timings: PhaseTimings = field(default_factory=PhaseTimings)
    error: str = ""

    @property
    def route_label(self) -> str:
        """Human-readable name of the knowledge source that was used."""
        return DOMAIN_LABELS.get(self.route, self.route)

    @property
    def used_retrieval(self) -> bool:
        return bool(self.sources)

    @property
    def route_agreed_with_rules(self) -> bool:
        """Whether the agent's route matched the deterministic classifier."""
        return self.route == self.rule_based_route

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "route_label": self.route_label,
            "agent": self.agent,
            "mode": self.mode,
            "sources": [s.to_dict() for s in self.sources],
            "retrieval_query": self.retrieval_query,
            "transfer_reason": self.transfer_reason,
            "rule_based_route": self.rule_based_route,
            "rule_based_confidence": round(self.rule_based_confidence, 4),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "timings": self.timings.to_dict(),
            "error": self.error,
        }


class TrainingAssistant:
    """Runs the onboarding agent graph over a single conversation.

    Example::

        assistant = TrainingAssistant()
        result = assistant.ask("How many days of annual leave do I get?")
        print(result.route)    # 'admin'
        print(result.answer)

    One instance holds one ADK session, so it represents one employee's
    conversation. Create a new instance (or call :meth:`reset`) for a new one.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        user_id: str = "new_employee",
        session_id: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.user_id = user_id
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"

        self._session_service = InMemorySessionService()
        self._agent = build_agent_graph(self.settings)
        self._runner = Runner(
            app_name=APP_NAME,
            agent=self._agent,
            session_service=self._session_service,
        )
        self._session_ready = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ask(self, question: str) -> AssistantAnswer:
        """Answer ``question`` synchronously.

        Works whether or not the caller already has a running event loop.
        Scripts and Streamlit normally have none, so ``asyncio.run`` is used
        directly. Some hosts (notebooks, certain Streamlit builds) do have one,
        and calling ``asyncio.run`` there raises; in that case the coroutine is
        handed to a dedicated worker thread with its own loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ask_async(question))

        result: dict[str, AssistantAnswer] = {}
        error: dict[str, BaseException] = {}

        def _worker() -> None:
            try:
                result["value"] = asyncio.run(self.ask_async(question))
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
                error["value"] = exc

        thread = threading.Thread(target=_worker, name="training-assistant-ask")
        thread.start()
        thread.join()

        if "value" in error:
            raise error["value"]
        return result["value"]

    async def ask_async(self, question: str) -> AssistantAnswer:
        """Answer ``question``, returning the answer and its full trace."""
        question = (question or "").strip()
        rule_decision = classify(question)

        if not question:
            return AssistantAnswer(
                question=question,
                answer="Ask me anything about working at Manna's Analytics.",
                route="general",
                agent="",
                mode=self.settings.mode,
                rule_based_route=rule_decision.route,
                rule_based_confidence=rule_decision.confidence,
                rule_based_reason=rule_decision.reason,
            )

        await self._ensure_session()
        started = time.perf_counter()

        try:
            trace = await self._run(question)
        except Exception as exc:  # pragma: no cover - surfaced to the user
            logger.exception("Agent run failed")
            return self._failure_answer(question, rule_decision, exc, started)

        elapsed = time.perf_counter() - started
        route = self._resolve_route(trace, rule_decision)
        state = await self._session_state()
        timings = trace.phase_timings(
            total_seconds=elapsed,
            search_seconds=float(state.get(STATE_SEARCH_SECONDS) or 0.0),
        )

        return AssistantAnswer(
            question=question,
            answer=trace.answer_text or self._empty_answer_text(),
            route=route,
            agent=trace.responding_agent,
            mode=self.settings.mode,
            sources=await self._collect_sources(),
            retrieval_query=await self._state_value(STATE_QUERY),
            transfer_reason=trace.transfer_reason,
            rule_based_route=rule_decision.route,
            rule_based_confidence=rule_decision.confidence,
            rule_based_reason=rule_decision.reason,
            llm_calls=trace.llm_calls,
            tool_calls=trace.tool_calls,
            elapsed_seconds=elapsed,
            timings=timings,
        )

    def reset(self) -> None:
        """Start a fresh conversation, discarding history and cached sources."""
        self.session_id = f"session-{uuid.uuid4().hex[:12]}"
        self._session_ready = False

    def describe(self) -> str:
        return f"{self.settings.describe()} | session={self.session_id}"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @dataclass(slots=True)
    class _Trace:
        answer_text: str = ""
        responding_agent: str = ""
        transferred_to: str = ""
        transfer_reason: str = ""
        llm_calls: int = 0
        tool_calls: list[str] = field(default_factory=list)

        # Phase boundaries, as perf_counter readings taken when the
        # corresponding event arrives. ``None`` means the phase never occurred,
        # which is normal: the fallback path has no retrieval.
        started_at: float = 0.0
        route_decided_at: float | None = None
        tool_called_at: float | None = None
        tool_returned_at: float | None = None
        answered_at: float | None = None

        def phase_timings(self, total_seconds: float, search_seconds: float) -> PhaseTimings:
            """Turn the recorded boundaries into durations."""
            timings = PhaseTimings(
                total_seconds=total_seconds,
                search_seconds=search_seconds,
            )

            if self.route_decided_at is not None:
                timings.routing_seconds = max(
                    0.0, self.route_decided_at - self.started_at
                )

            if self.tool_called_at is not None and self.tool_returned_at is not None:
                timings.retrieval_seconds = max(
                    0.0, self.tool_returned_at - self.tool_called_at
                )

            # Generation runs from whichever boundary most recently preceded the
            # answer: the tool response on a retrieval path, otherwise the
            # routing decision.
            generation_start = self.tool_returned_at or self.route_decided_at
            if self.answered_at is not None and generation_start is not None:
                timings.generation_seconds = max(
                    0.0, self.answered_at - generation_start
                )

            return timings

    async def _ensure_session(self) -> None:
        if self._session_ready:
            return
        await self._session_service.create_session(
            app_name=APP_NAME,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        self._session_ready = True

    async def _run(self, question: str) -> "TrainingAssistant._Trace":
        """Drive the runner and fold the event stream into a trace."""
        trace = self._Trace(started_at=time.perf_counter())
        message = types.Content(role="user", parts=[types.Part(text=question)])

        async for event in self._runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=message,
            # Clear last turn's retrieval provenance before this turn starts.
            # Session state persists across turns, so without this a question
            # answered on the no-retrieval fallback path would still display the
            # previous question's sources, which is actively misleading.
            state_delta={
                STATE_SOURCES: [],
                STATE_QUERY: "",
                STATE_DOMAIN: "",
                STATE_SEARCH_SECONDS: 0.0,
            },
        ):
            self._absorb(event, trace)

        return trace

    def _absorb(self, event: Event, trace: "TrainingAssistant._Trace") -> None:
        """Extract the interesting parts of one ADK event."""
        now = time.perf_counter()

        actions = event.actions
        if actions is not None:
            if getattr(actions, "transfer_to_agent", None):
                trace.transferred_to = actions.transfer_to_agent
                if trace.route_decided_at is None:
                    trace.route_decided_at = now
                reason = getattr(actions, "transfer_reason", "") or ""
                if reason:
                    trace.transfer_reason = reason

        content = event.content
        if content is None:
            return

        for part in content.parts or []:
            if part.function_response is not None:
                trace.tool_returned_at = now

            if part.function_call is not None:
                name = part.function_call.name or ""
                if name and name != "transfer_to_agent":
                    trace.tool_calls.append(name)
                    if trace.tool_called_at is None:
                        trace.tool_called_at = now
                elif name == "transfer_to_agent":
                    if trace.route_decided_at is None:
                        trace.route_decided_at = now
                    args = part.function_call.args or {}
                    target = args.get("agent_name")
                    if target:
                        trace.transferred_to = str(target)
                    reason = args.get("transfer_reason")
                    if reason and not trace.transfer_reason:
                        trace.transfer_reason = str(reason)

            if part.text and part.text.strip() and not event.partial:
                # The last non-partial model text wins: that is the answer the
                # employee sees. Intermediate agents rarely emit text, but a
                # live model occasionally narrates before calling a tool.
                trace.answer_text = part.text.strip()
                trace.answered_at = now
                trace.responding_agent = event.author or trace.responding_agent

        if event.usage_metadata is not None or event.content is not None:
            # Count one model turn per event that carries model output.
            if (event.author or "") and not event.partial:
                has_output = any(
                    (p.text and p.text.strip()) or p.function_call is not None
                    for p in (content.parts or [])
                )
                if has_output:
                    trace.llm_calls += 1

    def _resolve_route(
        self, trace: "TrainingAssistant._Trace", rule_decision: RouteDecision
    ) -> str:
        """Work out which route actually handled the question.

        Preference order: the agent that produced the answer, then the transfer
        target, then the deterministic classifier as a last resort. A live model
        can name a non-existent agent, in which case ADK will not transfer and
        the rule-based route is the honest label.
        """
        for candidate in (trace.responding_agent, trace.transferred_to):
            if candidate and candidate != ROOT_AGENT_NAME:
                route = agent_name_to_route(candidate)
                if route:
                    return route
        return rule_decision.route

    async def _session_state(self) -> dict[str, Any]:
        session = await self._session_service.get_session(
            app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id
        )
        if session is None:
            return {}
        return dict(session.state or {})

    async def _state_value(self, key: str) -> str:
        return str((await self._session_state()).get(key) or "")

    async def _collect_sources(self) -> list[SourceReference]:
        """Read the passages the retrieval tool recorded for this turn."""
        raw = (await self._session_state()).get(STATE_SOURCES) or []
        sources: list[SourceReference] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            sources.append(
                SourceReference(
                    number=int(item.get("rank", len(sources) + 1)),
                    citation=str(item.get("citation", "")),
                    score=float(item.get("score", 0.0)),
                    doc_id=str(item.get("doc_id", "")),
                    source_path=str(item.get("source_path", "")),
                    text=str(item.get("text", "")),
                )
            )
        return sources

    def _empty_answer_text(self) -> str:
        return (
            "I could not produce an answer for that. Please rephrase the "
            "question, or ask your onboarding buddy."
        )

    def _failure_answer(
        self,
        question: str,
        rule_decision: RouteDecision,
        exc: Exception,
        started: float,
    ) -> AssistantAnswer:
        hint = ""
        if not self.settings.offline:
            hint = (
                " If this persists, check that GOOGLE_API_KEY is valid and that "
                f"the model name '{self.settings.model}' is available to your "
                "account, or set TRAINING_ASSISTANT_FORCE_OFFLINE=true to run "
                "without the API."
            )
        return AssistantAnswer(
            question=question,
            answer=(
                "Something went wrong while answering that, so I would rather "
                "not guess. Please try again, and tell your onboarding buddy if "
                "it keeps happening." + hint
            ),
            route=rule_decision.route,
            agent="",
            mode=self.settings.mode,
            rule_based_route=rule_decision.route,
            rule_based_confidence=rule_decision.confidence,
            rule_based_reason=rule_decision.reason,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
