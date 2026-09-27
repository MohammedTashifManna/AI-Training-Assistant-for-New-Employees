"""Shared pytest fixtures, the latency logger, and terminal reporting.

Three things live here:

**Path and environment setup** so ``pytest`` works from a clean clone without an
install step, and so a stray ``GOOGLE_API_KEY`` in the shell cannot silently
switch the suite between offline and live mode mid-run.

**One shared execution pass.** Running the agent pipeline is the expensive part
of this suite, so the 20 dataset questions are executed exactly once in a
session-scoped fixture and the resulting :class:`AssistantAnswer` objects are
shared by the routing and faithfulness suites. Re-running them per test would
triple the work and, worse, allow the two suites to disagree about what the
system did.

**The latency logger.** :class:`LatencyRecorder` accumulates per-phase samples
(routing, retrieval, search, generation) tagged by route, and prints a summary
table at the end of the run via ``pytest_terminal_summary``.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

# --------------------------------------------------------------------------- #
# Import path
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"


# --------------------------------------------------------------------------- #
# CLI options
# --------------------------------------------------------------------------- #


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--latency-runs",
        action="store",
        default="3",
        help="Repetitions per question in the latency benchmark (default: 3).",
    )
    parser.addoption(
        "--judge",
        action="store",
        default="auto",
        choices=["auto", "gemini", "grounding"],
        help=(
            "Faithfulness judge. 'auto' uses Gemini when an API key is present "
            "and the deterministic grounding checker otherwise. 'gemini' forces "
            "the LLM judge and fails if no key is available. 'grounding' forces "
            "the deterministic checker."
        ),
    )
    parser.addoption(
        "--allow-live",
        action="store_true",
        default=False,
        help=(
            "Permit the pipeline itself to call Gemini. By default the suite "
            "pins the pipeline to offline mode so results are deterministic and "
            "free. The faithfulness judge is controlled separately by --judge."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "router: routing accuracy tests")
    config.addinivalue_line("markers", "faithfulness: grounding and hallucination tests")
    config.addinivalue_line("markers", "latency: performance benchmarks")
    config.addinivalue_line("markers", "slow: takes more than a second or two")

    # Pin the pipeline to offline mode unless explicitly allowed otherwise.
    # Without this, whether the suite hits the network depends on the shell it
    # happens to be launched from, which makes results irreproducible.
    if not config.getoption("--allow-live"):
        os.environ["TRAINING_ASSISTANT_FORCE_OFFLINE"] = "true"


# --------------------------------------------------------------------------- #
# Core fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def settings():
    """Resolved project settings, with the logging noise turned down."""
    from training_assistant.config import get_settings, reset_settings_cache
    from training_assistant.logging_setup import configure_logging, quiet_third_party

    configure_logging(False)
    quiet_third_party()

    import logging

    logging.getLogger("training_assistant").setLevel(logging.WARNING)

    reset_settings_cache()
    return get_settings()


@pytest.fixture(scope="session")
def dataset() -> dict[str, Any]:
    """The evaluation dataset, loaded once."""
    if not DATASET_PATH.is_file():
        pytest.fail(f"Evaluation dataset not found at {DATASET_PATH}")
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def routing_questions(dataset) -> list[dict[str, Any]]:
    return dataset["routing_questions"]


@pytest.fixture(scope="session")
def hallucination_probes(dataset) -> list[dict[str, Any]]:
    return dataset["hallucination_probes"]


@pytest.fixture(scope="session")
def retriever(settings):
    """The loaded knowledge index.

    Skips the whole suite with an actionable message when the index is missing,
    rather than failing every test with a FileNotFoundError.
    """
    from training_assistant.knowledge.retriever import load_retriever

    if not settings.index_path.is_file():
        pytest.skip(
            f"No knowledge index at {settings.index_path}. "
            "Build it first: python scripts/build_index.py"
        )
    return load_retriever()


@pytest.fixture(scope="session")
def assistant_factory(settings, retriever):
    """Factory for fresh assistants.

    Each call returns a new instance with its own ADK session, so a test cannot
    inherit conversation history from another test.
    """
    from training_assistant.assistant import TrainingAssistant

    def _make():
        return TrainingAssistant(settings)

    return _make


@pytest.fixture(scope="session")
def warm_assistant(assistant_factory):
    """An assistant that has already paid the one-off costs.

    The first question of a process loads the index from disk and builds BM25
    statistics, which on this corpus is roughly 500 ms against about 5 ms for
    every later question. Benchmarking without a warmup measures disk I/O and
    reports it as routing latency.
    """
    assistant = assistant_factory()
    assistant.ask("warmup: what is the leave policy?")
    assistant.reset()
    return assistant


@pytest.fixture(scope="session")
def pipeline_results(assistant_factory, routing_questions) -> dict[str, Any]:
    """Run every dataset question once; share the answers across suites.

    Returns ``{question_id: AssistantAnswer}``.
    """
    results: dict[str, Any] = {}
    for item in routing_questions:
        assistant = assistant_factory()
        results[item["id"]] = assistant.ask(item["question"])
    return results


@pytest.fixture(scope="session")
def probe_results(assistant_factory, hallucination_probes) -> dict[str, Any]:
    """Run the hallucination probes once."""
    results: dict[str, Any] = {}
    for item in hallucination_probes:
        assistant = assistant_factory()
        results[item["id"]] = assistant.ask(item["question"])
    return results


# --------------------------------------------------------------------------- #
# Latency logger
# --------------------------------------------------------------------------- #


@dataclass
class LatencySample:
    """One measured request."""

    question_id: str
    route: str
    total: float
    routing: float
    retrieval: float
    search: float
    generation: float

    @classmethod
    def from_answer(cls, question_id: str, answer) -> "LatencySample":
        t = answer.timings
        return cls(
            question_id=question_id,
            route=answer.route,
            total=t.total_seconds,
            routing=t.routing_seconds,
            retrieval=t.retrieval_seconds,
            search=t.search_seconds,
            generation=t.generation_seconds,
        )


PHASES = ("total", "routing", "retrieval", "search", "generation")


@dataclass
class LatencyRecorder:
    """Accumulates latency samples and summarises them.

    Percentiles rather than means: a mean hides the tail, and the tail is what a
    new employee actually experiences when they ask the awkward question.
    """

    samples: list[LatencySample] = field(default_factory=list)

    def record(self, question_id: str, answer) -> LatencySample:
        sample = LatencySample.from_answer(question_id, answer)
        self.samples.append(sample)
        return sample

    # ------------------------------ queries ------------------------------ #

    def values(self, phase: str, route: str | None = None) -> list[float]:
        rows = self.samples if route is None else [s for s in self.samples if s.route == route]
        return sorted(getattr(s, phase) for s in rows)

    @staticmethod
    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
        return values[index]

    def stats(self, phase: str, route: str | None = None) -> dict[str, float]:
        values = self.values(phase, route)
        if not values:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "p50": self.percentile(values, 0.50),
            "p95": self.percentile(values, 0.95),
            "max": values[-1],
        }

    @property
    def routes(self) -> list[str]:
        return sorted({s.route for s in self.samples})

    # ------------------------------ output ------------------------------- #

    def summary_lines(self) -> list[str]:
        if not self.samples:
            return ["  (no latency samples recorded)"]

        lines: list[str] = []
        header = (
            f"  {'scope':<24}{'n':>4}{'mean':>10}{'p50':>10}{'p95':>10}{'max':>10}"
        )

        lines.append("  Per phase, all routes (milliseconds)")
        lines.append(header)
        lines.append("  " + "-" * 66)
        for phase in PHASES:
            s = self.stats(phase)
            lines.append(
                f"  {phase:<24}{int(s['n']):>4}{s['mean'] * 1000:>10.1f}"
                f"{s['p50'] * 1000:>10.1f}{s['p95'] * 1000:>10.1f}{s['max'] * 1000:>10.1f}"
            )

        lines.append("")
        lines.append("  Total latency by route (milliseconds)")
        lines.append(header)
        lines.append("  " + "-" * 66)
        for route in self.routes:
            s = self.stats("total", route)
            lines.append(
                f"  {route:<24}{int(s['n']):>4}{s['mean'] * 1000:>10.1f}"
                f"{s['p50'] * 1000:>10.1f}{s['p95'] * 1000:>10.1f}{s['max'] * 1000:>10.1f}"
            )

        lines.append("")
        lines.append("  Phase share of total (mean, all routes)")
        total_mean = self.stats("total")["mean"] or 1.0
        for phase in ("routing", "retrieval", "generation"):
            share = self.stats(phase)["mean"] / total_mean
            bar = "#" * int(round(share * 40))
            lines.append(f"    {phase:<12}{share:>6.1%}  {bar}")

        unattributed = 1.0 - sum(
            self.stats(p)["mean"] / total_mean
            for p in ("routing", "retrieval", "generation")
        )
        lines.append(f"    {'unattributed':<12}{max(0.0, unattributed):>6.1%}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": len(self.samples),
            "phases": {phase: self.stats(phase) for phase in PHASES},
            "by_route": {
                route: {phase: self.stats(phase, route) for phase in PHASES}
                for route in self.routes
            },
            "samples": [
                {
                    "question_id": s.question_id,
                    "route": s.route,
                    **{phase: round(getattr(s, phase), 6) for phase in PHASES},
                }
                for s in self.samples
            ],
        }


@pytest.fixture(scope="session")
def latency(request: pytest.FixtureRequest) -> Iterator[LatencyRecorder]:
    """Session-wide latency recorder, summarised in the terminal report."""
    recorder = LatencyRecorder()
    request.config._latency_recorder = recorder  # noqa: SLF001 - shared with the hook
    yield recorder


@pytest.fixture(scope="session")
def latency_runs(request: pytest.FixtureRequest) -> int:
    try:
        return max(1, int(request.config.getoption("--latency-runs")))
    except (TypeError, ValueError):
        return 3


@pytest.fixture(scope="session")
def judge_choice(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--judge"))


# --------------------------------------------------------------------------- #
# Reporting hooks
# --------------------------------------------------------------------------- #


def pytest_terminal_summary(
    terminalreporter, exitstatus, config: pytest.Config
) -> None:
    """Print the latency table and any scorecards the suites published."""
    write = terminalreporter.write_line

    scorecards = getattr(config, "_scorecards", None)
    if scorecards:
        for title, lines in scorecards.items():
            write("")
            write("=" * 74)
            write(f"  {title}")
            write("=" * 74)
            for line in lines:
                write(line)

    recorder: LatencyRecorder | None = getattr(config, "_latency_recorder", None)
    if recorder and recorder.samples:
        write("")
        write("=" * 74)
        write("  LATENCY BREAKDOWN")
        write("=" * 74)
        for line in recorder.summary_lines():
            write(line)

        report_path = PROJECT_ROOT / "docs" / "latency_report.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(recorder.to_dict(), indent=2), encoding="utf-8"
            )
            write("")
            write(f"  latency report written to {report_path.relative_to(PROJECT_ROOT)}")
        except OSError as exc:  # pragma: no cover - best effort
            write(f"  could not write latency report: {exc}")


def publish_scorecard(config: pytest.Config, title: str, lines: list[str]) -> None:
    """Register a block of text to print in the terminal summary.

    Used by the suites to surface aggregate metrics, which pytest has no native
    concept of: it reports pass/fail per test, not "routing accuracy was 95%".
    """
    if not hasattr(config, "_scorecards"):
        config._scorecards = {}  # noqa: SLF001
    config._scorecards[title] = lines  # noqa: SLF001
