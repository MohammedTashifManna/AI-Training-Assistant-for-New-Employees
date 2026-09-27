"""Routing accuracy tests for the ADK decision node.

What is being tested
--------------------
The root ``onboarding_router`` agent must send each question to exactly one of
four destinations: the company knowledge base, the role and team documents, the
policy and admin documents, or the direct-LLM fallback. Sending a leave question
to the engineering charter produces a confident answer from the wrong book,
which is worse than no answer at all.

Three layers are measured separately, because they can disagree and the
disagreement is informative:

``classify()``
    The deterministic keyword classifier on its own. A cheap baseline the agent
    has to beat to justify its cost.
``resolve_route()``
    Keyword rules plus a retrieval probe when the rules are unsure.
Full pipeline
    The actual ADK graph, including ``transfer_to_agent`` and the specialist
    that answered. This is the number that matters; the others explain it.

Metrics
-------
Accuracy alone is misleading on a four-class problem with uneven support, so
per-category precision, recall and F1 are computed and asserted individually.
A router that sends everything to ``admin`` would score 25% accuracy but 100%
recall on ``admin``, and only the per-class view exposes that.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest

from conftest import publish_scorecard

from training_assistant.routing.classifier import ROUTES, classify
from training_assistant.routing.resolver import resolve_route

pytestmark = pytest.mark.router

# --------------------------------------------------------------------------- #
# Thresholds
#
# Set from measured behaviour, a little below current performance so ordinary
# noise does not fail the build but a real regression does. They are assertions
# about the product, not descriptions of today's run.
# --------------------------------------------------------------------------- #

MIN_OVERALL_ACCURACY = 0.90
MIN_MACRO_F1 = 0.90
MIN_PER_CATEGORY_RECALL = 0.80
MIN_PER_CATEGORY_PRECISION = 0.80

#: The fallback path must not become a dumping ground for questions the
#: knowledge base could have answered.
MAX_FALSE_FALLBACK_RATE = 0.10


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #


@dataclass
class ClassMetrics:
    route: str
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def support(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class RoutingScore:
    """Aggregate routing performance for one predictor."""

    name: str
    expected: list[str]
    predicted: list[str]
    ids: list[str]
    questions: list[str]

    @property
    def total(self) -> int:
        return len(self.expected)

    @property
    def correct(self) -> int:
        return sum(1 for e, p in zip(self.expected, self.predicted) if e == p)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def per_class(self) -> dict[str, ClassMetrics]:
        metrics: dict[str, ClassMetrics] = {}
        for route in ROUTES:
            tp = sum(
                1 for e, p in zip(self.expected, self.predicted) if e == route and p == route
            )
            fp = sum(
                1 for e, p in zip(self.expected, self.predicted) if e != route and p == route
            )
            fn = sum(
                1 for e, p in zip(self.expected, self.predicted) if e == route and p != route
            )
            metrics[route] = ClassMetrics(route, tp, fp, fn)
        return metrics

    @property
    def macro_f1(self) -> float:
        scores = [m.f1 for m in self.per_class().values() if m.support > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def false_fallback_rate(self) -> float:
        """Share of answerable questions dumped onto the fallback path."""
        answerable = [
            (e, p) for e, p in zip(self.expected, self.predicted) if e != "general"
        ]
        if not answerable:
            return 0.0
        return sum(1 for e, p in answerable if p == "general") / len(answerable)

    def confusion(self) -> Counter:
        return Counter(zip(self.expected, self.predicted))

    def mistakes(self) -> list[tuple[str, str, str, str]]:
        return [
            (qid, expected, predicted, question)
            for qid, expected, predicted, question in zip(
                self.ids, self.expected, self.predicted, self.questions
            )
            if expected != predicted
        ]

    def report_lines(self) -> list[str]:
        lines = [
            f"  {self.name}",
            f"    accuracy  : {self.correct}/{self.total} = {self.accuracy:.1%}",
            f"    macro F1  : {self.macro_f1:.3f}",
            f"    false fallback rate: {self.false_fallback_rate:.1%}",
            "",
            f"    {'route':<10}{'prec':>8}{'recall':>8}{'F1':>8}{'support':>9}",
            "    " + "-" * 43,
        ]
        for route, m in self.per_class().items():
            if m.support == 0 and m.false_positive == 0:
                continue
            lines.append(
                f"    {route:<10}{m.precision:>8.3f}{m.recall:>8.3f}"
                f"{m.f1:>8.3f}{m.support:>9}"
            )

        confusion = self.confusion()
        lines.append("")
        lines.append("    confusion (rows = expected, columns = predicted)")
        lines.append("      " + "".join(f"{r:>10}" for r in ROUTES))
        for expected in ROUTES:
            row = "".join(f"{confusion[(expected, p)]:>10}" for p in ROUTES)
            lines.append(f"      {expected:<10}{row}")

        mistakes = self.mistakes()
        if mistakes:
            lines.append("")
            lines.append(f"    misroutes ({len(mistakes)}):")
            for qid, expected, predicted, question in mistakes:
                lines.append(
                    f"      {qid}  expected={expected:<8} got={predicted:<8} :: {question[:52]}"
                )
        return lines


def _score(name: str, questions, predictor) -> RoutingScore:
    return RoutingScore(
        name=name,
        expected=[q["expected_route"] for q in questions],
        predicted=[predictor(q) for q in questions],
        ids=[q["id"] for q in questions],
        questions=[q["question"] for q in questions],
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def classifier_score(routing_questions) -> RoutingScore:
    return _score(
        "Stage 1 only - deterministic keyword classifier",
        routing_questions,
        lambda q: classify(q["question"]).route,
    )


@pytest.fixture(scope="module")
def resolver_score(routing_questions, retriever) -> RoutingScore:
    return _score(
        "Stage 1 + 2 - keyword rules with retrieval probe",
        routing_questions,
        lambda q: resolve_route(q["question"]).route,
    )


@pytest.fixture(scope="module")
def pipeline_score(routing_questions, pipeline_results) -> RoutingScore:
    return _score(
        "Full ADK pipeline - router agent and specialist",
        routing_questions,
        lambda q: pipeline_results[q["id"]].route,
    )


# --------------------------------------------------------------------------- #
# Dataset sanity
# --------------------------------------------------------------------------- #


def test_dataset_is_balanced_and_well_formed(routing_questions, dataset):
    """A skewed or malformed dataset makes every downstream metric meaningless."""
    assert len(routing_questions) == 20, "Expected 20 routing questions."

    ids = [q["id"] for q in routing_questions]
    assert len(set(ids)) == len(ids), "Duplicate question ids."

    by_route = Counter(q["expected_route"] for q in routing_questions)
    assert set(by_route) == set(ROUTES), f"Routes covered: {sorted(by_route)}"
    assert all(count == 5 for count in by_route.values()), (
        f"Expected 5 questions per route, got {dict(by_route)}"
    )

    category_map = dataset["categories"]
    for question in routing_questions:
        assert question["question"].strip(), f"{question['id']} has an empty question"
        assert category_map[question["category"]] == question["expected_route"], (
            f"{question['id']}: category {question['category']} does not map to "
            f"route {question['expected_route']}"
        )
        # Retrieval routes need ground truth; the fallback route must not have any.
        if question["expected_route"] == "general":
            assert not question["reference_chunks"], (
                f"{question['id']} is a fallback question but carries reference chunks"
            )
        else:
            assert question["reference_chunks"], (
                f"{question['id']} is a retrieval question with no reference chunks"
            )


# --------------------------------------------------------------------------- #
# Per-question routing, one test case each
# --------------------------------------------------------------------------- #


def _cases(dataset_path=None):
    """Load ids and questions at collection time for parametrisation."""
    import json
    from pathlib import Path

    path = dataset_path or Path(__file__).resolve().parent / "eval_dataset.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        pytest.param(q["id"], q["question"], q["expected_route"], id=q["id"])
        for q in payload["routing_questions"]
    ]


@pytest.mark.parametrize("qid,question,expected", _cases())
def test_question_routes_correctly(qid, question, expected, pipeline_results):
    """Each question reaches its expected knowledge source.

    Parametrised so a failure names the offending question instead of reporting
    "routing accuracy was 85%" and leaving you to find out which one broke.
    """
    answer = pipeline_results[qid]
    assert not answer.error, f"{qid} raised: {answer.error}"
    assert answer.route == expected, (
        f"{qid} routed to {answer.route!r}, expected {expected!r}\n"
        f"  question: {question}\n"
        f"  agent   : {answer.agent}\n"
        f"  reason  : {answer.transfer_reason or 'n/a'}"
    )


@pytest.mark.parametrize("qid,question,expected", _cases())
def test_retrieval_scoping_matches_route(qid, question, expected, pipeline_results):
    """Retrieval happens on knowledge routes and never on the fallback route.

    This is the guarantee that makes the routing decision binding: the fallback
    agent has no retrieval tool, so it cannot cite a company document, and the
    specialists cannot read outside their own collection.
    """
    answer = pipeline_results[qid]

    if expected == "general":
        assert not answer.sources, (
            f"{qid} took the fallback path but returned "
            f"{len(answer.sources)} source(s); the fallback agent must have no "
            "retrieval tool"
        )
        assert not answer.tool_calls, f"{qid} fallback path called tools: {answer.tool_calls}"
    else:
        assert answer.sources, f"{qid} routed to {expected} but retrieved nothing"
        assert answer.tool_calls, f"{qid} routed to {expected} but called no tool"
        for source in answer.sources:
            assert source.citation, f"{qid} returned a source with no citation"


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #


def test_overall_routing_accuracy(pipeline_score, request):
    publish_scorecard(
        request.config,
        "ROUTING ACCURACY",
        pipeline_score.report_lines(),
    )
    assert pipeline_score.accuracy >= MIN_OVERALL_ACCURACY, (
        f"Routing accuracy {pipeline_score.accuracy:.1%} is below the "
        f"{MIN_OVERALL_ACCURACY:.0%} threshold.\n"
        + "\n".join(pipeline_score.report_lines())
    )


def test_macro_f1_across_categories(pipeline_score):
    """Macro F1 weights every category equally, so a neglected route shows up."""
    assert pipeline_score.macro_f1 >= MIN_MACRO_F1, (
        f"Macro F1 {pipeline_score.macro_f1:.3f} is below {MIN_MACRO_F1:.2f}.\n"
        + "\n".join(pipeline_score.report_lines())
    )


@pytest.mark.parametrize("route", ROUTES)
def test_per_category_precision_and_recall(route, pipeline_score):
    """Every category must hold its own, not just the average."""
    metrics = pipeline_score.per_class()[route]
    if metrics.support == 0:
        pytest.skip(f"No questions expect route {route!r}")

    assert metrics.recall >= MIN_PER_CATEGORY_RECALL, (
        f"Recall for {route!r} is {metrics.recall:.1%} "
        f"({metrics.true_positive}/{metrics.support}), below "
        f"{MIN_PER_CATEGORY_RECALL:.0%}. Questions that belong to {route!r} are "
        "being sent elsewhere."
    )
    assert metrics.precision >= MIN_PER_CATEGORY_PRECISION, (
        f"Precision for {route!r} is {metrics.precision:.1%}, below "
        f"{MIN_PER_CATEGORY_PRECISION:.0%}. Questions from other categories are "
        f"leaking into {route!r}."
    )


def test_answerable_questions_are_not_dumped_on_the_fallback(pipeline_score):
    """The fallback path is a destination, not an escape hatch.

    A router under pressure tends to over-use the fallback, because it is never
    obviously wrong. It is wrong here: the answer existed in the corpus.
    """
    rate = pipeline_score.false_fallback_rate
    assert rate <= MAX_FALSE_FALLBACK_RATE, (
        f"{rate:.1%} of answerable questions were routed to the fallback path, "
        f"above the {MAX_FALSE_FALLBACK_RATE:.0%} limit."
    )


def test_pipeline_is_at_least_as_accurate_as_the_keyword_baseline(
    pipeline_score, resolver_score, classifier_score, request
):
    """The agent pipeline must not underperform the cheap baseline.

    If it does, the routing complexity is not paying for itself and the
    deterministic classifier should be used directly.
    """
    publish_scorecard(
        request.config,
        "ROUTING - LAYER COMPARISON",
        classifier_score.report_lines() + [""] + resolver_score.report_lines(),
    )

    assert pipeline_score.accuracy >= classifier_score.accuracy, (
        f"Full pipeline ({pipeline_score.accuracy:.1%}) is less accurate than the "
        f"keyword classifier alone ({classifier_score.accuracy:.1%})."
    )


def test_routing_is_deterministic_within_a_run(routing_questions, assistant_factory):
    """The same question must route the same way twice.

    Offline the pipeline is fully deterministic, so any variation here is a
    state leak between sessions rather than model sampling.
    """
    sample = [q for q in routing_questions if q["id"] in {"R02", "R07", "R12", "R18"}]
    for question in sample:
        first = assistant_factory().ask(question["question"]).route
        second = assistant_factory().ask(question["question"]).route
        assert first == second, (
            f"{question['id']} routed to {first!r} then {second!r} on identical input"
        )


def test_conversation_history_does_not_leak_routing(assistant_factory):
    """A follow-up on a different subject must be re-routed, not inherited.

    Turn-scoping bugs in a multi-agent loop typically show up exactly here: the
    second question gets answered from the first question's retrieval.
    """
    assistant = assistant_factory()

    first = assistant.ask("What is the nightly hotel cap for a work trip to Mumbai?")
    assert first.route == "admin", f"setup question routed to {first.route}"

    second = assistant.ask("What technology stack does the engineering team use?")
    assert second.route == "role", (
        f"follow-up routed to {second.route!r}, expected 'role'. The previous "
        "turn's route appears to have carried over."
    )
    assert second.retrieval_query != first.retrieval_query, (
        "The follow-up reused the previous turn's retrieval query: "
        f"{second.retrieval_query!r}"
    )
