"""Hybrid retrieval over the domain-partitioned knowledge index.

Each search combines two signals:

**Dense / semantic** - cosine similarity between the query vector and each chunk
vector. Good at matching intent when the wording differs.

**Lexical / BM25** - Okapi BM25 over the same tokenisation. Good at the exact
identifiers a policy corpus is full of: "L5", "Sev-1", "26 weeks", "Expensify".

Neither alone is adequate here. Dense-only misses precise figures; BM25-only
misses paraphrases such as "how much holiday do I get" against a document that
says "privilege leave entitlement". The two are blended with a configurable
weight.

Retrieval is always **scoped to a single domain**, which is what makes the
router's decision meaningful: an ``admin`` question cannot pull in a ``role``
passage.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from ..config import Settings, get_settings
from .documents import Chunk
from .embeddings import Embedder, build_embedder
from .text_utils import tokenize
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

#: Okapi BM25 term-saturation parameter, at its conventional value.
BM25_K1 = 1.5

#: Length-normalisation strength, reduced from the conventional 0.75.
#:
#: A policy corpus is full of deliberately comprehensive summary tables, and at
#: 0.75 BM25 penalises them for being long: the "Entitlements at a glance" table
#: holding the headline figure ("22 working days") ranked below two shorter
#: sections that merely mention privilege leave in passing. Lowering it to 0.6
#: put the table back in the top three for that question and changed nothing
#: else in a sweep across 11 fact-lookup queries, while off-topic questions
#: still correctly return no passages at all.
BM25_B = 0.6

#: BM25 scores are unbounded, so they are squashed with ``s / (s + SATURATION)``
#: into [0, 1). A min-max normalisation was rejected because it would force the
#: best candidate to 1.0 even when nothing in the corpus is relevant, which
#: would defeat the "admit you don't know" threshold.
BM25_SATURATION = 6.0

#: How many candidates the lexical and dense stages each contribute before
#: blending. Kept well above top_k so a chunk strong in only one signal still
#: reaches the blend.
CANDIDATE_POOL = 25

#: How many sibling sections may be pulled in to complete an enumeration.
#:
#: Some answers are split across sibling subsections of one parent heading:
#: "Office locations > Bengaluru", "> Austin", "> Berlin". Ranked independently,
#: the longest sibling loses on term density and falls outside top_k, so the
#: assistant answers "which offices do we have?" with two of the three and looks
#: confidently wrong. Siblings are completions of an already-selected passage
#: rather than independent results, so they are allowed to exceed ``top_k``.
SIBLING_EXPANSION_LIMIT = 2


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk plus the scores that got it selected."""

    chunk: Chunk
    score: float
    semantic_score: float
    lexical_score: float
    rank: int

    @property
    def citation(self) -> str:
        return self.chunk.citation

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "citation": self.citation,
            "score": round(self.score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "doc_id": self.chunk.doc_id,
            "domain": self.chunk.domain,
            "source_path": self.chunk.source_path,
            "text": self.chunk.text,
        }


class _Bm25Index:
    """Okapi BM25 statistics for one domain."""

    def __init__(self, texts: Sequence[str]) -> None:
        self._doc_tokens: list[Counter[str]] = [Counter(tokenize(t)) for t in texts]
        self._doc_lengths = np.asarray(
            [sum(c.values()) for c in self._doc_tokens], dtype=np.float32
        )
        self._avg_length = (
            float(self._doc_lengths.mean()) if len(self._doc_lengths) else 0.0
        ) or 1.0

        document_frequency: Counter[str] = Counter()
        for counter in self._doc_tokens:
            document_frequency.update(counter.keys())

        total = max(1, len(self._doc_tokens))
        self._idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

        # Inverted index: term -> list of (doc_index, term_frequency).
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for index, counter in enumerate(self._doc_tokens):
            for term, freq in counter.items():
                self._postings.setdefault(term, []).append((index, freq))

    def score(self, query: str) -> np.ndarray:
        """Raw BM25 score for every document in the domain."""
        scores = np.zeros(len(self._doc_tokens), dtype=np.float32)
        query_terms = tokenize(query, drop_query_noise=True)
        if not query_terms:
            return scores

        for term in set(query_terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_index, term_freq in postings:
                length_norm = BM25_B * (
                    self._doc_lengths[doc_index] / self._avg_length
                )
                denominator = term_freq + BM25_K1 * (1.0 - BM25_B + length_norm)
                scores[doc_index] += idf * (term_freq * (BM25_K1 + 1.0)) / denominator

        return scores


class Retriever:
    """Hybrid, domain-scoped retrieval over a loaded :class:`VectorStore`."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings or get_settings()
        self._bm25: dict[str, _Bm25Index] = {}

    # ------------------------------------------------------------------ #

    def _bm25_for(self, domain: str) -> _Bm25Index:
        if domain not in self._bm25:
            collection = self.store.collection(domain)
            self._bm25[domain] = _Bm25Index(
                [chunk.embedding_text for chunk in collection.chunks]
            )
        return self._bm25[domain]

    def search(
        self,
        domain: str,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return the best passages for ``query`` within ``domain``.

        Results below ``min_score`` are dropped, which is how the assistant
        learns to say "this is not in the documents I have" instead of
        confidently citing an unrelated section.
        """
        top_k = top_k or self.settings.top_k
        min_score = self.settings.min_score if min_score is None else min_score

        collection = self.store.collection(domain)
        if not len(collection) or not query.strip():
            return []

        # --- dense half ---
        try:
            query_vector = self.embedder.embed_query(query)
            semantic = self.store.similarity(domain, query_vector)
        except Exception:  # pragma: no cover - network/credential failure
            logger.warning(
                "Semantic scoring failed for domain %s; "
                "falling back to lexical-only retrieval.",
                domain,
                exc_info=True,
            )
            semantic = np.zeros(len(collection), dtype=np.float32)

        semantic = np.clip(semantic, 0.0, 1.0)

        # --- lexical half ---
        raw_lexical = self._bm25_for(domain).score(query)
        lexical = raw_lexical / (raw_lexical + BM25_SATURATION)

        # --- blend ---
        weight = self.settings.semantic_weight
        hybrid = weight * semantic + self.settings.lexical_weight * lexical

        # Union of the strongest candidates from each signal, so a chunk that
        # only one signal likes still gets considered.
        pool = min(CANDIDATE_POOL, len(collection))
        candidates = set(np.argsort(-hybrid)[:pool].tolist())
        candidates.update(np.argsort(-semantic)[: max(1, pool // 2)].tolist())
        candidates.update(np.argsort(-lexical)[: max(1, pool // 2)].tolist())

        ordered = sorted(candidates, key=lambda i: float(hybrid[i]), reverse=True)

        results: list[RetrievedChunk] = []
        for index in ordered:
            score = float(hybrid[index])
            if score < min_score:
                continue
            results.append(
                RetrievedChunk(
                    chunk=collection.chunks[index],
                    score=score,
                    semantic_score=float(semantic[index]),
                    lexical_score=float(lexical[index]),
                    rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break

        results = self._complete_siblings(results, collection, hybrid, semantic, lexical, min_score)

        logger.debug(
            "Retrieved %d/%d chunks for domain=%s query=%r",
            len(results),
            top_k,
            domain,
            query[:80],
        )
        return results

    @staticmethod
    def _parent_path(chunk: Chunk) -> tuple[str, ...]:
        """Heading path with the final heading removed."""
        return chunk.heading_path[:-1] if len(chunk.heading_path) > 1 else ()

    def _complete_siblings(
        self,
        results: list[RetrievedChunk],
        collection,
        hybrid: np.ndarray,
        semantic: np.ndarray,
        lexical: np.ndarray,
        min_score: float,
    ) -> list[RetrievedChunk]:
        """Add sibling subsections of the top result that ranking left behind.

        Only fires when the top result sits under a shared parent heading and a
        sibling clears the same relevance floor, so this cannot drag in
        unrelated text.
        """
        if not results:
            return results

        parent = self._parent_path(results[0].chunk)
        if not parent:
            return results

        already = {r.chunk.chunk_id for r in results}
        candidates: list[tuple[float, int]] = []
        for index, chunk in enumerate(collection.chunks):
            if chunk.chunk_id in already:
                continue
            if self._parent_path(chunk) != parent:
                continue
            score = float(hybrid[index])
            if score >= min_score:
                candidates.append((score, index))

        if not candidates:
            return results

        candidates.sort(reverse=True)
        for score, index in candidates[:SIBLING_EXPANSION_LIMIT]:
            results.append(
                RetrievedChunk(
                    chunk=collection.chunks[index],
                    score=score,
                    semantic_score=float(semantic[index]),
                    lexical_score=float(lexical[index]),
                    rank=len(results) + 1,
                )
            )

        # Re-rank so numbering stays dense and reflects score order.
        results.sort(key=lambda r: r.score, reverse=True)
        for position, result in enumerate(results, start=1):
            result.rank = position

        return results

    # ------------------------------------------------------------------ #

    def format_context(self, results: Sequence[RetrievedChunk]) -> str:
        """Render results as a numbered, citable context block for the LLM.

        The numbering is what the prompts instruct the model to cite, so the
        format here and the citation instruction in ``prompts.py`` must agree.
        """
        if not results:
            return "NO_RELEVANT_CONTEXT"

        blocks: list[str] = []
        for result in results:
            chunk = result.chunk
            header_bits = [f"[{result.rank}] {chunk.citation}"]
            if chunk.policy_version:
                header_bits.append(f"(policy v{chunk.policy_version})")
            if chunk.last_reviewed:
                header_bits.append(f"(last reviewed {chunk.last_reviewed})")
            blocks.append(f"{' '.join(header_bits)}\n{chunk.text}")

        return "\n\n---\n\n".join(blocks)

    def describe(self) -> str:
        stats = ", ".join(f"{d}={n}" for d, n in self.store.stats().items())
        return f"backend={self.store.backend} dim={self.store.dim} chunks[{stats}]"


def load_retriever(settings: Settings | None = None) -> Retriever:
    """Load the persisted index and wrap it in a :class:`Retriever`."""
    settings = settings or get_settings()

    store = VectorStore.load(
        settings.index_path, expected_backend=settings.embedding_backend
    )
    embedder = build_embedder(settings)
    embedder.load_state(store.embedder_state)
    return Retriever(store=store, embedder=embedder, settings=settings)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide cached retriever.

    The index is a few hundred vectors, but reloading it per question would
    still dominate latency in the Streamlit UI, which re-runs the script on
    every interaction.
    """
    return load_retriever()


def reset_retriever_cache() -> None:
    get_retriever.cache_clear()
