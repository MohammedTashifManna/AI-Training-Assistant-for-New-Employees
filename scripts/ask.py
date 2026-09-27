"""Command-line interface to the Training Assistant.

Ask one question::

    python scripts/ask.py "How many days of annual leave do I get?"

Show the routing trace and sources::

    python scripts/ask.py "What is the mileage rate?" --trace

Interactive chat (keeps conversation context)::

    python scripts/ask.py

Machine-readable output::

    python scripts/ask.py "Who is the CEO?" --json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (path setup side effect)

import argparse
import json
import sys

from training_assistant.assistant import AssistantAnswer, TrainingAssistant
from training_assistant.config import get_settings
from training_assistant.logging_setup import configure_logging

RULE = "-" * 72


def _print_answer(result: AssistantAnswer, show_trace: bool) -> None:
    print()
    print(f"ROUTE  {result.route}  ({result.route_label})")
    print(f"AGENT  {result.agent or 'n/a'}")
    print(RULE)
    print(result.answer)
    print(RULE)

    if result.sources:
        print("SOURCES")
        for source in result.sources:
            print(f"  [{source.number}] {source.citation}   (relevance {source.score:.3f})")
    else:
        print("SOURCES  none (no retrieval on this path)")

    if show_trace:
        print()
        timings = result.timings
        print("TRACE")
        print(f"  mode              : {result.mode}")
        print(f"  retrieval query   : {result.retrieval_query or 'n/a'}")
        print(f"  tool calls        : {', '.join(result.tool_calls) or 'none'}")
        print(f"  model turns       : {result.llm_calls}")
        print(f"  elapsed           : {result.elapsed_seconds:.2f}s")
        print(
            f"  phases            : routing {timings.routing_seconds * 1000:.0f}ms"
            f" | retrieval {timings.retrieval_seconds * 1000:.0f}ms"
            f" (search {timings.search_seconds * 1000:.0f}ms)"
            f" | generation {timings.generation_seconds * 1000:.0f}ms"
            f" | other {timings.overhead_seconds * 1000:.0f}ms"
        )
        print(f"  rule-based route  : {result.rule_based_route} "
              f"(confidence {result.rule_based_confidence:.2f})")
        print(f"  routes agree      : {result.route_agreed_with_rules}")
        if result.transfer_reason:
            print(f"  transfer reason   : {result.transfer_reason}")
        if result.error:
            print(f"  error             : {result.error}")
    print()


def _interactive(assistant: TrainingAssistant, show_trace: bool) -> int:
    print("=" * 72)
    print("  Manna's Analytics - AI Training Assistant for New Employees")
    print("=" * 72)
    print(f"  {assistant.describe()}")
    print()
    print("  Ask a question, or type one of: /trace  /reset  /quit")
    print()

    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not question:
            continue

        lowered = question.lower()
        if lowered in {"/quit", "/exit", "quit", "exit"}:
            print("Bye.")
            return 0
        if lowered == "/reset":
            assistant.reset()
            print("Conversation reset.\n")
            continue
        if lowered == "/trace":
            show_trace = not show_trace
            print(f"Trace {'on' if show_trace else 'off'}.\n")
            continue

        result = assistant.ask(question)
        _print_answer(result, show_trace)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the AI Training Assistant a question.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", nargs="*", help="The question to ask.")
    parser.add_argument("--trace", action="store_true", help="Show the routing trace.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    settings = get_settings()
    if not settings.index_path.is_file():
        print(
            f"ERROR: no knowledge index at {settings.index_path}\n"
            "Build it first:  python scripts/build_index.py",
            file=sys.stderr,
        )
        return 1

    assistant = TrainingAssistant(settings)
    question = " ".join(args.question).strip()

    if not question:
        return _interactive(assistant, args.trace)

    result = assistant.ask(question)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_answer(result, args.trace)

    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
