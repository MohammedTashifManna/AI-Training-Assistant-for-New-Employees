"""Embedding backends for the knowledge index.

Two interchangeable implementations sit behind one :class:`Embedder` interface:

:class:`GeminiEmbedder`
    Calls the Gemini embedding API. Used whenever an API key is configured.

:class:`TfidfEmbedder`
    A self-contained TF-IDF vectoriser with no network dependency. Used in
    offline mode so that the retrieval half of the system stays genuinely
    functional (and testable) without credentials.

Both return **L2-normalised** vectors, so a dot product is a cosine similarity.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Sequence

import numpy as np

from ..config import Settings
from .text_utils import tokenize

logger = logging.getLogger(__name__)

#: Documents per Gemini embedding request. The API accepts batches; this keeps
#: request sizes modest so a single oversized section cannot fail the whole run.
GEMINI_BATCH_SIZE = 32


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows untouched."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class Embedder(ABC):
    """Turns text into dense vectors."""

    #: Short identifier persisted alongside the index.
    backend: str = "abstract"

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    def fit(self, texts: Sequence[str]) -> None:
        """Learn any corpus statistics required before embedding.

        No-op for API-backed embedders.
        """

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed corpus passages. Returns an ``(n, dim)`` array."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query. Returns a ``(dim,)`` array."""

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state needed to embed queries after a reload."""
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""

    def term_weight(self, term: str) -> float:
        """How much a single (already stemmed) term should count.

        Exposes corpus-level term rarity so consumers other than the vectoriser
        can reuse it. Backends without such statistics return a flat weight.
        """
        return 1.0


class TfidfEmbedder(Embedder):
    """Exact TF-IDF vectoriser with sublinear term frequency.

    One dimension per vocabulary term, so there is no hashing and therefore no
    collision noise. That matters more than it might sound: with a hashed
    projection, an off-topic query could land in a bucket shared with a real
    term and manufacture a similarity score out of nothing, which in turn
    defeats the "no relevant context" threshold that stops the assistant
    inventing answers.

    Vectors are wide but extremely sparse, so the store persists them sparsely.
    This design suits a corpus of a few hundred passages; at a much larger scale
    the Gemini embedding backend is the right choice.
    """

    backend = "tfidf"

    def __init__(self, dim: int = 0) -> None:
        # ``dim`` is accepted for interface symmetry but is derived from the
        # fitted vocabulary; it is not a tunable for this backend.
        self._idf: dict[str, float] = {}
        self._vocabulary: dict[str, int] = {}

    @property
    def dim(self) -> int:
        return len(self._vocabulary)

    @property
    def _fitted(self) -> bool:
        return bool(self._vocabulary)

    # ----------------------------- fitting ----------------------------- #

    def fit(self, texts: Sequence[str]) -> None:
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(tokenize(text)))

        total_docs = max(1, len(texts))
        # Smoothed IDF, always positive, so a term appearing in every document
        # still contributes a little rather than vanishing entirely.
        self._idf = {
            term: math.log((total_docs + 1.0) / (freq + 1.0)) + 1.0
            for term, freq in document_frequency.items()
        }
        self._rebuild_vocabulary()
        logger.info(
            "TF-IDF fitted on %d passages, vocabulary size %d",
            total_docs,
            len(self._vocabulary),
        )

    def _rebuild_vocabulary(self) -> None:
        """Assign each term a stable column index.

        Sorting makes the mapping deterministic, so it can be rebuilt from the
        persisted IDF table alone without storing a second dictionary.
        """
        self._vocabulary = {term: i for i, term in enumerate(sorted(self._idf))}

    # ---------------------------- embedding ---------------------------- #

    def _vectorise(self, text: str, *, is_query: bool) -> np.ndarray:
        vector = np.zeros(len(self._vocabulary), dtype=np.float32)
        counts = Counter(tokenize(text, drop_query_noise=is_query))

        for term, count in counts.items():
            column = self._vocabulary.get(term)
            if column is None:
                # Term never seen in the corpus. It cannot match any document,
                # so it contributes nothing. Anything else would be noise.
                continue
            # Sublinear scaling damps a term repeated many times inside one
            # long policy table.
            vector[column] += (1.0 + math.log(count)) * self._idf[term]

        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([self._vectorise(t, is_query=False) for t in texts])
        return _l2_normalise(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "TfidfEmbedder must be fitted (or loaded from an index) "
                "before embedding a query."
            )
        vector = self._vectorise(text, is_query=True).reshape(1, -1)
        return _l2_normalise(vector)[0]

    # ------------------------- (de)serialisation ------------------------ #

    def state_dict(self) -> dict[str, Any]:
        return {"idf": self._idf}

    def load_state(self, state: dict[str, Any]) -> None:
        self._idf = {str(k): float(v) for k, v in (state.get("idf") or {}).items()}
        self._rebuild_vocabulary()

    def term_weight(self, term: str) -> float:
        """Inverse document frequency of ``term``.

        An unseen term is treated as maximally rare, which is the right default
        for a caller ranking text: a term absent from the corpus but present in
        both the question and a passage is a strong, specific match.
        """
        if term in self._idf:
            return self._idf[term]
        return max(self._idf.values(), default=1.0)


class GeminiEmbedder(Embedder):
    """Embeddings from the Gemini embedding API via ``google-genai``."""

    backend = "gemini"

    def __init__(self, model: str, dim: int, api_key: str = "") -> None:
        self._model = model
        self._dim = int(dim)
        self._api_key = api_key
        self._client: Any = None

    @property
    def dim(self) -> int:
        return self._dim

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai  # imported lazily so offline mode is free

            # An empty api_key lets the SDK fall back to its own resolution
            # order, which is what Vertex AI mode needs.
            self._client = (
                genai.Client(api_key=self._api_key) if self._api_key else genai.Client()
            )
        return self._client

    def _embed(self, texts: Sequence[str], task_type: str) -> np.ndarray:
        import time
        from google.genai import errors, types

        client = self._get_client()
        vectors: list[list[float]] = []

        for start in range(0, len(texts), GEMINI_BATCH_SIZE):
            batch = list(texts[start: start + GEMINI_BATCH_SIZE])

            # Rate limiting retry loop
            retries = 0
            max_retries = 5
            backoff = 10.0  # seconds to wait if rate-limited

            while True:
                try:
                    response = client.models.embed_content(
                        model=self._model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self._dim,
                        ),
                    )
                    embeddings = getattr(response, "embeddings", None) or []
                    if len(embeddings) != len(batch):
                        raise RuntimeError(
                            "Gemini embedding response size mismatch: "
                            f"asked for {len(batch)}, received {len(embeddings)}."
                        )
                    vectors.extend(list(item.values) for item in embeddings)

                    # Throttle calls to stay under the RPM limit on Free Tier
                    time.sleep(2.0)
                    break

                except errors.APIError as exc:
                    # Catch 429 Resource Exhausted / Quota errors
                    if exc.code == 429 or "RESOURCE_EXHAUSTED" in str(exc):
                        retries += 1
                        if retries > max_retries:
                            raise exc
                        logger.warning(
                            "Quota limit reached. Sleeping %.1fs before retry (%d/%d)...",
                            backoff,
                            retries,
                            max_retries,
                        )
                        time.sleep(backoff)
                        backoff *= 2  # Exponential backoff
                    else:
                        raise exc

        if not vectors:
            return np.zeros((0, self._dim), dtype=np.float32)

        return _l2_normalise(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def state_dict(self) -> dict[str, Any]:
        return {"model": self._model, "dim": self._dim}

    def load_state(self, state: dict[str, Any]) -> None:
        self._model = str(state.get("model", self._model))
        self._dim = int(state.get("dim", self._dim))


def build_embedder(settings: Settings) -> Embedder:
    """Return the embedder matching the active execution mode."""
    if settings.offline:
        # Width is derived from the fitted vocabulary, so embed_dim does not
        # apply to this backend.
        return TfidfEmbedder()
    return GeminiEmbedder(
        model=settings.embed_model,
        dim=settings.embed_dim,
        api_key=settings.api_key,
    )
