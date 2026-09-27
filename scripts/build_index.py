"""Build the retrieval index from the Markdown knowledge base.

Usage::

    python scripts/build_index.py
    python scripts/build_index.py --verbose
    python scripts/build_index.py --dry-run     # chunking report only

The index is written to ``index/knowledge_index_<backend>.json``, where backend
is ``tfidf`` in offline mode and ``gemini`` when an API key is configured. The
filename carries the backend so the two can never be mixed.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (path setup side effect)

import argparse
import logging
import sys
from collections import Counter

from training_assistant.config import get_settings
from training_assistant.knowledge.documents import load_all_chunks
from training_assistant.knowledge.embeddings import build_embedder
from training_assistant.knowledge.vector_store import build_store


def _report(domain_chunks: dict[str, list]) -> None:
    print("\nKnowledge base contents")
    print("-" * 68)
    grand_total = 0
    for domain, chunks in domain_chunks.items():
        docs = Counter(chunk.doc_id for chunk in chunks)
        lengths = [len(chunk.text) for chunk in chunks] or [0]
        print(f"\n  domain: {domain}")
        print(f"    documents : {len(docs)}")
        print(f"    chunks    : {len(chunks)}")
        print(
            f"    chunk chars: min={min(lengths)} "
            f"mean={sum(lengths) // max(1, len(lengths))} max={max(lengths)}"
        )
        for doc_id, count in sorted(docs.items()):
            print(f"      - {doc_id:<34} {count:>3} chunks")
        grand_total += len(chunks)
    print("-" * 68)
    print(f"  total chunks: {grand_total}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report chunking without embedding or writing the index.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    settings = get_settings()
    print("=" * 68)
    print("  Building knowledge index")
    print("=" * 68)
    print(f"  {settings.describe()}")

    try:
        domain_chunks = load_all_chunks()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    _report(domain_chunks)

    empty = [domain for domain, chunks in domain_chunks.items() if not chunks]
    if empty:
        print(f"ERROR: no content found for domain(s): {', '.join(empty)}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry run: nothing embedded, nothing written.")
        return 0

    if settings.offline:
        print("Embedding with the local TF-IDF backend (no API key configured).")
    else:
        print(f"Embedding with {settings.embed_model} via the Gemini API.")

    embedder = build_embedder(settings)
    try:
        store = build_store(domain_chunks, embedder)
    except Exception as exc:
        print(f"\nERROR: embedding failed: {exc}", file=sys.stderr)
        if not settings.offline:
            print(
                "Hint: check GOOGLE_API_KEY, or set "
                "TRAINING_ASSISTANT_FORCE_OFFLINE=true to build a local index.",
                file=sys.stderr,
            )
        return 1

    store.save(settings.index_path)

    print(f"\nIndex written to: {settings.index_path}")
    print(f"  backend : {store.backend}")
    print(f"  vectors : {store.total_chunks()} x {store.dim}")
    print(f"  domains : {store.stats()}")
    print("\nNext: python scripts/ask.py \"How many days of annual leave do I get?\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
