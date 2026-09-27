"""Evaluate the decision node and the retrieval that follows it.

Three things are measured:

**Routing accuracy** for both the full agent pipeline and the deterministic
rule-based classifier, per route and overall, with a confusion matrix. Reporting
both matters: the rule baseline is what the agent has to beat to justify itself.

**Generalisation**, via a held-out paraphrase probe written in casual,
indirect language and never used to tune the routing rules. The main question
set was authored alongside those rules, so its score is an upper bound, and
publishing only that number would be misleading.

**Grounding**, for the three retrieval routes: did the answer actually come back
with source passages, and does it cite them? A correctly routed answer with no
citations is still a failure for this use case.

Usage::

    python scripts/evaluate_routing.py                # full evaluation
    python scripts/evaluate_routing.py --rules-only   # classifier only, no agents
    python scripts/evaluate_routing.py --probe-only
    python scripts/evaluate_routing.py --json report.json
    python scripts/evaluate_routing.py --limit 10     # quick smoke run
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (path setup side effect)

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training_assistant.assistant import TrainingAssistant
from training_assistant.config import EVAL_QUESTIONS_PATH, DATA_DIR, get_settings
from training_assistant.logging_setup import configure_logging
from training_assistant.routing.classifier import ROUTES, classify
from training_assistant.routing.resolver import resolve_route

PROBE_PATH = DATA_DIR / "eval" / "paraphrase_probe.json"

RETRIEVAL_ROUTES = ("company", "role", "admin")
CITATION_RE = re.compile(r"\[\d+\]")

# Capture exact retry delay seconds from Google Gemini error payloads
RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)

RULE = "=" * 74
THIN = "-" * 74


# --------------------------------------------------------------------------- #
# Helper for robust API calls with exponential backoff & auto-delay
# --------------------------------------------------------------------------- #


def call_api_with_retry(
    fn, *args, max_retries: int = 6, default_delay: float = 15.0, **kwargs
):
    """Executes an API call function, retrying on 429/ResourceExhausted quota errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            err_msg = str(exc)

            # Check for 429 / RESOURCE_EXHAUSTED rate or daily limit errors
            if any(
                term in err_msg
                for term in (
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "ResourceExhausted",
                    "Quota exceeded",
                )
            ):
                if attempt == max_retries:
                    print(
                        f"\n[Rate Limit] Maximum retries ({max_retries}) exhausted for query."
                    )
                    raise exc

                # Parse requested retry duration from error payload if Google provided it
                match = RETRY_DELAY_RE.search(err_msg)
                if match:
                    sleep_time = float(match.group(1)) + 2.0  # Buffer extra 2 seconds
                else:
                    sleep_time = default_delay * (1.5 ** (attempt - 1))

                print(
                    f"\n[Rate Limit / Quota Exceeded] Throttled. Waiting {sleep_time:.1f}s "
                    f"before attempt {attempt + 1}/{max_retries}...",
                    flush=True,
                )
                time.sleep(sleep_time)
            else:
                raise exc


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class Outcome:
    """One evaluated question."""

    qid: str
    question: str
    expected: str
    predicted: str
    correct: bool
    source_count: int = 0
    has_citation: bool = False
    elapsed: float = 0.0
    agent: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.qid,
            "question": self.question,
            "expected_route": self.expected,
            "predicted_route": self.predicted,
            "correct": self.correct,
            "source_count": self.source_count,
            "has_citation": self.has_citation,
            "elapsed_seconds": round(self.elapsed, 3),
            "agent": self.agent,
            "error": self.error,
        }


@dataclass
class Report:
    """Aggregated results for one evaluation pass."""

    name: str
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def correct(self) -> int:
        return sum(1 for o in self.outcomes if o.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def confusion(self) -> Counter:
        return Counter((o.expected, o.predicted) for o in self.outcomes)

    def per_route(self) -> dict[str, dict[str, float]]:
        """Precision, recall and F1 for each route."""
        metrics: dict[str, dict[str, float]] = {}
        for route in ROUTES:
            true_positive = sum(
                1 for o in self.outcomes if o.expected == route and o.predicted == route
            )
            false_positive = sum(
                1 for o in self.outcomes if o.expected != route and o.predicted == route
            )
            false_negative = sum(
                1 for o in self.outcomes if o.expected == route and o.predicted != route
            )
            support = true_positive + false_negative

            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive)
                else 0.0
            )
            recall = true_positive / support if support else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            metrics[route] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(support),
            }
        return metrics

    def macro_f1(self) -> float:
        scores = [m["f1"] for m in self.per_route().values() if m["support"] > 0]
        return sum(scores) / len(scores) if scores else 0.0

    def misroutes(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.correct]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1(), 4),
            "per_route": {
                route: {k: round(v, 4) for k, v in metrics.items()}
                for route, metrics in self.per_route().items()
            },
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_questions(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Question set not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") or []
    return questions[:limit] if limit else questions


# --------------------------------------------------------------------------- #
# Evaluation passes
# --------------------------------------------------------------------------- #


def evaluate_rules(questions: list[dict[str, Any]], name: str) -> Report:
    """Score the deterministic classifier. No model, no retrieval."""
    report = Report(name=name)
    for item in questions:
        decision = classify(item["question"])
        report.outcomes.append(
            Outcome(
                qid=item.get("id", "?"),
                question=item["question"],
                expected=item["expected_route"],
                predicted=decision.route,
                correct=decision.route == item["expected_route"],
            )
        )
    return report


def evaluate_resolver(questions: list[dict[str, Any]], name: str) -> Report:
    """Score the two-stage resolver: keyword rules, then a retrieval probe."""
    report = Report(name=name)
    for item in questions:
        resolved = resolve_route(item["question"])
        report.outcomes.append(
            Outcome(
                qid=item.get("id", "?"),
                question=item["question"],
                expected=item["expected_route"],
                predicted=resolved.route,
                correct=resolved.route == item["expected_route"],
                agent=resolved.decided_by,
            )
        )
    return report


def evaluate_pipeline(questions: list[dict[str, Any]], name: str) -> Report:
    """Score the full agent pipeline, including retrieval and answering.

    A fresh assistant per question keeps the turns independent, so conversation
    history cannot leak routing signal from one question into the next.
    """
    settings = get_settings()
    report = Report(name=name)

    total_q = len(questions)
    for index, item in enumerate(questions, start=1):
        assistant = TrainingAssistant(settings)

        try:
            # Execute query through rate-limit retry logic
            result = call_api_with_retry(assistant.ask, item["question"])

            report.outcomes.append(
                Outcome(
                    qid=item.get("id", "?"),
                    question=item["question"],
                    expected=item["expected_route"],
                    predicted=result.route,
                    correct=result.route == item["expected_route"],
                    source_count=len(result.sources),
                    has_citation=bool(CITATION_RE.search(result.answer)),
                    elapsed=result.elapsed_seconds,
                    agent=result.agent,
                    error=result.error,
                )
            )
            print(
                f"  [{index:>2}/{total_q}] {item.get('id','?'):<4} "
                f"expected={item['expected_route']:<8} got={result.route:<8} "
                f"{'ok' if result.route == item['expected_route'] else 'MISROUTED'}",
                flush=True,
            )
        except Exception as exc:
            # Catch exceptions to prevent crashing the rest of the batch evaluation
            print(
                f"  [{index:>2}/{total_q}] {item.get('id','?'):<4} ERROR: {exc}",
                flush=True,
            )
            report.outcomes.append(
                Outcome(
                    qid=item.get("id", "?"),
                    question=item["question"],
                    expected=item["expected_route"],
                    predicted="error",
                    correct=False,
                    error=str(exc),
                )
            )

        # 15s pause ensures execution stays well within Free Tier RPM allowances
        if index < total_q:
            time.sleep(15.0)

    return report


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


def _print_confusion(report: Report) -> None:
    confusion = report.confusion()
    print(f"  {'expected \\ predicted':<22}" + "".join(f"{r:>10}" for r in ROUTES))
    for expected in ROUTES:
        row = "".join(f"{confusion[(expected, p)]:>10}" for p in ROUTES)
        print(f"  {expected:<22}{row}")


def _print_report(report: Report) -> None:
    print()
    print(RULE)
    print(f"  {report.name}")
    print(RULE)
    print(f"  accuracy : {report.correct}/{report.total} = {report.accuracy:.1%}")
    print(f"  macro F1 : {report.macro_f1():.3f}")
    print()
    _print_confusion(report)
    print()
    print(f"  {'route':<10}{'precision':>11}{'recall':>9}{'F1':>8}{'support':>9}")
    print("  " + THIN[:47])
    for route, metrics in report.per_route().items():
        if metrics["support"] == 0:
            continue
        print(
            f"  {route:<10}{metrics['precision']:>11.3f}"
            f"{metrics['recall']:>9.3f}{metrics['f1']:>8.3f}"
            f"{int(metrics['support']):>9}"
        )

    misroutes = report.misroutes()
    if misroutes:
        print()
        print(f"  misrouted ({len(misroutes)}):")
        for outcome in misroutes:
            print(
                f"    {outcome.qid:<5} expected={outcome.expected:<8} "
                f"got={outcome.predicted:<8} :: {outcome.question[:60]}"
            )

    retrieval_outcomes = [
        o
        for o in report.outcomes
        if o.expected in RETRIEVAL_ROUTES and o.predicted in RETRIEVAL_ROUTES
    ]
    if retrieval_outcomes and any(o.source_count for o in retrieval_outcomes):
        with_sources = sum(1 for o in retrieval_outcomes if o.source_count > 0)
        with_citations = sum(1 for o in retrieval_outcomes if o.has_citation)
        mean_sources = sum(o.source_count for o in retrieval_outcomes) / len(
            retrieval_outcomes
        )
        print()
        print("  grounding (questions routed to a retrieval path):")
        print(
            f"    retrieved >=1 passage : {with_sources}/{len(retrieval_outcomes)} "
            f"({with_sources / len(retrieval_outcomes):.1%})"
        )
        print(
            f"    answer cites a source : {with_citations}/{len(retrieval_outcomes)} "
            f"({with_citations / len(retrieval_outcomes):.1%})"
        )
        print(f"    mean passages used    : {mean_sources:.2f}")

    errors = [o for o in report.outcomes if o.error]
    if errors:
        print()
        print(f"  errors ({len(errors)}):")
        for outcome in errors[:5]:
            print(f"    {outcome.qid}: {outcome.error[:90]}")

    latencies = [o.elapsed for o in report.outcomes if o.elapsed > 0]
    if latencies:
        latencies.sort()
        print()
        print(
            f"  latency: median {latencies[len(latencies) // 2]:.2f}s  "
            f"max {latencies[-1]:.2f}s"
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Score the rule-based classifier only (fast, no agent runs).",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Score only the held-out paraphrase probe.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate the first N questions."
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Write a JSON report here."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Debug logging."
    )
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    if not args.verbose:
        logging.getLogger("training_assistant").setLevel(logging.WARNING)

    settings = get_settings()
    print(RULE)
    print("  Routing and grounding evaluation")
    print(RULE)
    print(f"  {settings.describe()}")

    if not settings.index_path.is_file():
        print(
            f"\nERROR: no knowledge index at {settings.index_path}\n"
            "Build it first:  python scripts/evaluate_routing.py",
            file=sys.stderr,
        )
        return 1

    try:
        main_questions = _load_questions(EVAL_QUESTIONS_PATH, args.limit)
        probe_questions = _load_questions(PROBE_PATH, args.limit)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    reports: list[Report] = []

    if not args.probe_only:
        print(f"\n  labelled set : {len(main_questions)} questions")
    print(f"  probe set    : {len(probe_questions)} questions (held out, untuned)")

    # --- rule-based baseline -------------------------------------------- #
    if not args.probe_only:
        reports.append(
            evaluate_rules(main_questions, "Rule-based classifier - labelled set")
        )
    reports.append(
        evaluate_rules(probe_questions, "Rule-based classifier - held-out probe")
    )

    # --- two-stage resolver (rules + retrieval probe) -------------------- #
    if not args.probe_only:
        reports.append(
            evaluate_resolver(main_questions, "Rules + retrieval probe - labelled set")
        )
    reports.append(
        evaluate_resolver(probe_questions, "Rules + retrieval probe - held-out probe")
    )

    # --- full pipeline --------------------------------------------------- #
    if not args.rules_only:
        if not args.probe_only:
            print()
            print("  Running the full agent pipeline over the labelled set...")
            reports.append(
                evaluate_pipeline(main_questions, "Full agent pipeline - labelled set")
            )
        print()
        print("  Running the full agent pipeline over the held-out probe...")
        reports.append(
            evaluate_pipeline(probe_questions, "Full agent pipeline - held-out probe")
        )

    for report in reports:
        _print_report(report)

    # --- summary --------------------------------------------------------- #
    print()
    print(RULE)
    print("  Summary")
    print(RULE)
    for report in reports:
        print(
            f"  {report.name:<46} {report.accuracy:>7.1%}  macroF1 {report.macro_f1():.3f}"
        )
    print()
    print(
        "  Note: the labelled set was authored alongside the routing rules, so\n"
        "  its accuracy is an upper bound. The held-out probe is the honest\n"
        "  generalisation signal."
    )

    if args.json:
        payload = {
            "settings": settings.describe(),
            "reports": [report.to_dict() for report in reports],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n  JSON report written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())