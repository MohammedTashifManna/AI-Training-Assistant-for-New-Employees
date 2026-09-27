"""Latency benchmarks, broken down by pipeline phase.

What is measured
----------------
Each request is decomposed using boundaries taken from the ADK event stream
(see :class:`training_assistant.assistant.PhaseTimings`):

============  ==========================================================
``routing``   Question received until the router's transfer decision.
``retrieval`` The specialist's tool call until its response returns.
``search``    Time inside the retriever itself, reported by the tool.
``generation``Tool response until the final answer text.
============  ==========================================================

Total is measured independently, end to end, so ``overhead`` -- the part the
three phases do not explain -- is visible rather than hidden. A decomposition
that always sums exactly to the total is usually a decomposition that is
quietly discarding something.

Why a warmup matters
--------------------
The first request in a process loads the index from disk and builds BM25
statistics: roughly 500 ms against about 5 ms for every request afterwards.
Benchmarking without a warmup measures disk I/O once and reports it as routing
latency forever, which is why :func:`test_cold_start_is_reported_separately`
measures it deliberately and every other test here uses the warm fixture.

Reading the numbers
-------------------
Percentiles, not means. The mean hides the tail, and the tail is what an
employee experiences on the awkward question. Thresholds below are generous
against measured offline performance because CI machines are noisy and a flaky
performance test gets disabled, which is worse than having none.
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.latency, pytest.mark.slow]

# --------------------------------------------------------------------------- #
# Budgets
#
# Offline mode does no network I/O, so these bound local compute. In live mode
# the Gemini round trips dominate and the meaningful budget is different, which
# is why the live thresholds are separate and much looser.
# --------------------------------------------------------------------------- #

OFFLINE_BUDGETS = {
    "total_p95": 0.60,
    "routing_p95": 0.30,
    "retrieval_p95": 0.30,
    "generation_p95": 0.20,
}

LIVE_BUDGETS = {
    "total_p95": 25.0,
    "routing_p95": 12.0,
    "retrieval_p95": 8.0,
    "generation_p95": 20.0,
}

#: Routing and retrieval should be a small share of a live request, where model
#: latency dominates. Offline there is no model, so the split is not meaningful
#: and this check is skipped.
MAX_ROUTING_SHARE_LIVE = 0.60


@pytest.fixture(scope="module")
def budgets(settings) -> dict[str, float]:
    return OFFLINE_BUDGETS if settings.offline else LIVE_BUDGETS


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def measured(warm_assistant, routing_questions, latency, latency_runs):
    """Run every question ``--latency-runs`` times and record each request.

    A single shared assistant is reused deliberately: creating one per request
    would rebuild the agent graph each time and measure construction rather than
    answering. ``reset()`` between requests keeps turns independent without
    paying that cost.
    """
    for _ in range(latency_runs):
        for question in routing_questions:
            warm_assistant.reset()
            answer = warm_assistant.ask(question["question"])
            latency.record(question["id"], answer)
    return latency


# --------------------------------------------------------------------------- #
# Cold start, measured on purpose
# --------------------------------------------------------------------------- #


def test_cold_start_is_reported_separately(assistant_factory):
    """Quantify the one-off index load instead of letting it pollute averages.

    The retriever is cached process-wide, so by the time this runs another test
    has usually already paid the load cost and a naive "first request" is not
    cold at all. The cache is therefore cleared explicitly, which makes the
    measurement independent of test ordering. Measuring the warm request
    afterwards repopulates the cache for the tests that follow.
    """
    from training_assistant.knowledge.retriever import reset_retriever_cache

    reset_retriever_cache()

    assistant = assistant_factory()
    started = time.perf_counter()
    first = assistant.ask("What is the nightly hotel cap for Mumbai?")
    cold = time.perf_counter() - started

    assistant.reset()
    started = time.perf_counter()
    second = assistant.ask("What is the per kilometre mileage rate in India?")
    warm = time.perf_counter() - started

    assert not first.error and not second.error

    print(
        f"\n  cold start: {cold * 1000:.1f}ms   warm: {warm * 1000:.1f}ms   "
        f"ratio: {cold / max(warm, 1e-6):.1f}x"
    )

    # The point is that the gap is large and paid once, not that it is small.
    # Asserting the warm path is faster is what justifies the warmup fixture
    # every other test in this file depends on.
    assert warm < cold, (
        f"Warm request ({warm * 1000:.1f}ms) was not faster than the cold one "
        f"({cold * 1000:.1f}ms). Index caching may be broken, in which case "
        "every request is paying the load cost."
    )


# --------------------------------------------------------------------------- #
# Internal consistency of the decomposition
# --------------------------------------------------------------------------- #


def test_phase_decomposition_is_coherent(measured):
    """Phases must be non-negative and never exceed the measured total.

    If they do, the boundaries are being read from the event stream incorrectly
    and every latency number in this file is wrong.
    """
    for sample in measured.samples:
        phases = (sample.routing, sample.retrieval, sample.generation)
        assert all(value >= 0 for value in phases), f"{sample.question_id}: negative phase"
        assert sample.search >= 0
        assert sample.search <= sample.retrieval + 1e-6, (
            f"{sample.question_id}: pure search time ({sample.search:.4f}s) exceeds "
            f"the retrieval span ({sample.retrieval:.4f}s)"
        )
        assert sum(phases) <= sample.total + 1e-3, (
            f"{sample.question_id}: phases sum to {sum(phases):.4f}s but total "
            f"was {sample.total:.4f}s"
        )


def test_fallback_route_does_no_retrieval(measured):
    """The no-retrieval path must cost nothing in retrieval.

    A non-zero value here would mean the fallback agent acquired a retrieval
    tool, which is the architectural guarantee this project rests on.
    """
    fallback = [s for s in measured.samples if s.route == "general"]
    assert fallback, "No fallback-route samples were recorded"

    for sample in fallback:
        assert sample.retrieval == 0.0, (
            f"{sample.question_id} took the fallback route but spent "
            f"{sample.retrieval * 1000:.1f}ms in retrieval"
        )
        assert sample.search == 0.0


# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "phase,budget_key",
    [
        ("total", "total_p95"),
        ("routing", "routing_p95"),
        ("retrieval", "retrieval_p95"),
        ("generation", "generation_p95"),
    ],
)
def test_phase_stays_within_budget(phase, budget_key, measured, budgets, settings):
    """Each phase's 95th percentile must sit inside its budget."""
    stats = measured.stats(phase)
    if stats["n"] == 0:
        pytest.skip(f"No samples for {phase}")

    budget = budgets[budget_key]
    assert stats["p95"] <= budget, (
        f"{phase} p95 is {stats['p95'] * 1000:.1f}ms, over the "
        f"{budget * 1000:.0f}ms budget for {settings.mode} mode "
        f"(mean {stats['mean'] * 1000:.1f}ms, max {stats['max'] * 1000:.1f}ms, "
        f"n={int(stats['n'])})"
    )


@pytest.mark.parametrize("route", ["company", "role", "admin", "general"])
def test_no_route_is_pathologically_slow(route, measured, budgets):
    """Catch one route degrading while the overall average stays acceptable."""
    stats = measured.stats("total", route)
    if stats["n"] == 0:
        pytest.skip(f"No samples for route {route!r}")

    budget = budgets["total_p95"]
    assert stats["p95"] <= budget, (
        f"Route {route!r} p95 total is {stats['p95'] * 1000:.1f}ms, over the "
        f"{budget * 1000:.0f}ms budget (n={int(stats['n'])})"
    )


def test_retrieval_overhead_is_mostly_real_work(measured, settings):
    """Most of the retrieval span should be search, not framework overhead.

    A widening gap between the retrieval span and pure search time means ADK's
    tool dispatch is becoming the cost, which is a different problem from slow
    search and is fixed differently.
    """
    retrieval_samples = [s for s in measured.samples if s.retrieval > 0]
    if not retrieval_samples:
        pytest.skip("No retrieval samples")

    span = sum(s.retrieval for s in retrieval_samples)
    search = sum(s.search for s in retrieval_samples)
    share = search / span if span else 0.0

    print(
        f"\n  pure search is {share:.1%} of the retrieval span "
        f"({search * 1000:.1f}ms of {span * 1000:.1f}ms across "
        f"{len(retrieval_samples)} samples)"
    )

    # Deliberately loose: on a corpus this small the absolute numbers are tiny,
    # so the ratio is noisy. The assertion catches an order-of-magnitude shift.
    assert share > 0.02, (
        f"Only {share:.1%} of retrieval time is actual search; ADK tool "
        "dispatch appears to dominate."
    )


def test_routing_is_not_the_bottleneck_in_live_mode(measured, settings):
    """Routing should be a minority of a live request's time.

    Skipped offline, where there is no model call and the split says nothing
    useful about production behaviour.
    """
    if settings.offline:
        pytest.skip(
            "Offline mode has no model latency, so the phase split is not "
            "representative of production"
        )

    total = measured.stats("total")["mean"]
    routing = measured.stats("routing")["mean"]
    share = routing / total if total else 0.0

    assert share <= MAX_ROUTING_SHARE_LIVE, (
        f"Routing is {share:.1%} of mean total latency, above "
        f"{MAX_ROUTING_SHARE_LIVE:.0%}. The router is doing too much work for a "
        "node whose only job is to choose a destination."
    )


def test_latency_is_stable_across_repeats(measured, latency_runs):
    """Repeated identical requests should not drift.

    Growing latency across repeats points at unbounded accumulation, most likely
    conversation history that is never trimmed.
    """
    if latency_runs < 2:
        pytest.skip("Needs --latency-runs >= 2")

    per_question: dict[str, list[float]] = {}
    for sample in measured.samples:
        per_question.setdefault(sample.question_id, []).append(sample.total)

    drifting: list[str] = []
    for qid, values in per_question.items():
        if len(values) < 2:
            continue
        first, last = values[0], values[-1]
        # Allow generous absolute slack so millisecond-scale noise on a tiny
        # corpus cannot fail the build.
        if last > max(first * 5.0, first + 0.5):
            drifting.append(
                f"{qid}: {first * 1000:.1f}ms -> {last * 1000:.1f}ms"
            )

    assert not drifting, "Latency grew across repeats:\n  " + "\n  ".join(drifting)
