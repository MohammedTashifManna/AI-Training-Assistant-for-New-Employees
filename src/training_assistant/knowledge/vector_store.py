"""A small persistent vector store, partitioned by knowledge domain.

Why not an off-the-shelf vector database? The corpus is a few hundred passages,
and the routing design needs *strictly partitioned* collections so that an
``admin`` question can never retrieve a ``role`` passage. A dependency-free store
keeps the project reproducible on any machine and makes the retrieval mechanics
visible rather than hidden behind a service.

The store persists as a single JSON document containing:

* the embedding backend name and dimensionality (so incompatible vectors can
  never be silently mixed),
* the embedder's own state (the TF-IDF vocabulary and IDF weights),
* one collection per domain, each holding chunk metadata plus its vectors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .documents import Chunk
from .embeddings import Embedder

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

#: Vectors are written sparsely when at most this fraction of entries are
#: non-zero. TF-IDF vectors over an exact vocabulary sit far below this;
#: Gemini vectors are dense and fall back to the plain list form.
SPARSE_DENSITY_THRESHOLD = 0.3

#: Decimal places kept on disk. Well below the precision that affects ranking.
VECTOR_PRECISION = 6


def _encode_vectors(vectors: np.ndarray, dim: int) -> dict[str, Any]:
    """Serialise a vector matrix, choosing a sparse or dense representation."""
    if vectors.size == 0:
        return {"format": "dense", "rows": []}

    density = float(np.count_nonzero(vectors)) / float(vectors.size)
    rounded = np.round(vectors, VECTOR_PRECISION)

    if density > SPARSE_DENSITY_THRESHOLD:
        return {"format": "dense", "rows": rounded.tolist()}

    rows: list[list[list[float]]] = []
    for row in rounded:
        (indices,) = np.nonzero(row)
        rows.append([[int(i), float(row[i])] for i in indices])
    return {"format": "sparse", "dim": dim, "rows": rows}


def _decode_vectors(payload: Any, rows_expected: int, dim: int) -> np.ndarray:
    """Inverse of :func:`_encode_vectors`, tolerant of the plain-list form."""
    # Legacy / plain form: a bare list of lists.
    if isinstance(payload, list):
        array = np.asarray(payload, dtype=np.float32)
        return array if array.size else np.zeros((rows_expected, dim), np.float32)

    fmt = payload.get("format", "dense")
    rows = payload.get("rows") or []

    if fmt == "dense":
        array = np.asarray(rows, dtype=np.float32)
        return array if array.size else np.zeros((rows_expected, dim), np.float32)

    if fmt != "sparse":
        raise ValueError(f"Unknown vector encoding {fmt!r} in index.")

    width = int(payload.get("dim", dim))
    matrix = np.zeros((len(rows), width), dtype=np.float32)
    for row_index, entries in enumerate(rows):
        for column, value in entries:
            matrix[row_index, int(column)] = float(value)
    return matrix


@dataclass(slots=True)
class Collection:
    """Vectors and chunks for a single knowledge domain."""

    domain: str
    chunks: list[Chunk]
    vectors: np.ndarray  # shape (len(chunks), dim)

    def __len__(self) -> int:
        return len(self.chunks)


class VectorStore:
    """Domain-partitioned vector store with JSON persistence."""

    def __init__(self, backend: str, dim: int) -> None:
        self.backend = backend
        self.dim = dim
        self._collections: dict[str, Collection] = {}
        self._embedder_state: dict[str, Any] = {}

    # ------------------------------ building ----------------------------- #

    def add_domain(self, domain: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Attach a fully embedded domain to the store."""
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Domain {domain!r}: {len(chunks)} chunks but "
                f"{vectors.shape[0]} vectors."
            )
        if vectors.size and vectors.shape[1] != self.dim:
            raise ValueError(
                f"Domain {domain!r}: vectors have width {vectors.shape[1]}, "
                f"store expects {self.dim}."
            )
        self._collections[domain] = Collection(
            domain=domain,
            chunks=chunks,
            vectors=vectors.astype(np.float32, copy=False),
        )

    def set_embedder_state(self, state: dict[str, Any]) -> None:
        self._embedder_state = state

    @property
    def embedder_state(self) -> dict[str, Any]:
        """State needed to rebuild the embedder that produced these vectors."""
        return self._embedder_state

    # ------------------------------ reading ------------------------------ #

    @property
    def domains(self) -> list[str]:
        return sorted(self._collections)

    def collection(self, domain: str) -> Collection:
        try:
            return self._collections[domain]
        except KeyError as exc:
            raise KeyError(
                f"Domain {domain!r} is not in this index. "
                f"Available domains: {self.domains}. "
                "Rebuild with `python scripts/build_index.py`."
            ) from exc

    def has_domain(self, domain: str) -> bool:
        return domain in self._collections

    def total_chunks(self) -> int:
        return sum(len(c) for c in self._collections.values())

    def iter_chunks(self) -> Iterable[Chunk]:
        for collection in self._collections.values():
            yield from collection.chunks

    def stats(self) -> dict[str, int]:
        return {domain: len(c) for domain, c in sorted(self._collections.items())}

    # ---------------------------- persistence ---------------------------- #

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend,
            "dim": self.dim,
            "embedder_state": self._embedder_state,
            "collections": {
                domain: {
                    "chunks": [chunk.to_dict() for chunk in collection.chunks],
                    "vectors": _encode_vectors(collection.vectors, self.dim),
                }
                for domain, collection in self._collections.items()
            },
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        logger.info(
            "Saved index to %s (%d chunks across %d domains)",
            path,
            self.total_chunks(),
            len(self._collections),
        )

    @classmethod
    def load(cls, path: Path, *, expected_backend: str | None = None) -> "VectorStore":
        if not path.is_file():
            raise FileNotFoundError(
                f"No index found at {path}. "
                "Build it first with `python scripts/build_index.py`."
            )

        payload = json.loads(path.read_text(encoding="utf-8"))

        schema_version = int(payload.get("schema_version", 0))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Index at {path} uses schema version {schema_version}, "
                f"this build expects {SCHEMA_VERSION}. "
                "Rebuild with `python scripts/build_index.py`."
            )

        backend = str(payload["backend"])
        if expected_backend and backend != expected_backend:
            raise ValueError(
                f"Index at {path} was built with the {backend!r} embedding "
                f"backend but the current configuration uses {expected_backend!r}. "
                "Rebuild with `python scripts/build_index.py`."
            )

        store = cls(backend=backend, dim=int(payload["dim"]))
        store.set_embedder_state(payload.get("embedder_state") or {})

        for domain, raw in (payload.get("collections") or {}).items():
            chunks = [Chunk.from_dict(item) for item in raw["chunks"]]
            vectors = _decode_vectors(raw["vectors"], len(chunks), store.dim)
            store.add_domain(domain, chunks, vectors)

        logger.info(
            "Loaded index from %s (%d chunks, backend=%s)",
            path,
            store.total_chunks(),
            backend,
        )
        return store

    # ------------------------------ search ------------------------------- #

    def similarity(self, domain: str, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of ``query_vector`` against one domain.

        Both sides are L2-normalised at build time, so this is a plain dot
        product. Returns a ``(n_chunks,)`` array.
        """
        collection = self.collection(domain)
        if not len(collection):
            return np.zeros(0, dtype=np.float32)
        return collection.vectors @ query_vector.astype(np.float32)


def build_store(
    domain_chunks: dict[str, list[Chunk]], embedder: Embedder
) -> VectorStore:
    """Embed every domain and assemble a :class:`VectorStore`.

    The embedder is fitted on the **whole** corpus before any domain is
    embedded, so IDF weights are global. Fitting per domain would make the same
    passage score differently depending on which collection it landed in.
    """
    all_texts = [
        chunk.embedding_text
        for chunks in domain_chunks.values()
        for chunk in chunks
    ]
    embedder.fit(all_texts)

    store = VectorStore(backend=embedder.backend, dim=embedder.dim)
    for domain, chunks in domain_chunks.items():
        vectors = embedder.embed_documents([c.embedding_text for c in chunks])
        store.add_domain(domain, chunks, vectors)

    store.set_embedder_state(embedder.state_dict())
    return store
