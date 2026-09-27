"""Knowledge layer: document loading, chunking, embedding, storage, retrieval."""

from __future__ import annotations

from .documents import Chunk, load_all_chunks, load_document, load_domain_chunks
from .embeddings import Embedder, GeminiEmbedder, TfidfEmbedder, build_embedder
from .retriever import (
    Retriever,
    RetrievedChunk,
    get_retriever,
    load_retriever,
    reset_retriever_cache,
)
from .vector_store import Collection, VectorStore, build_store

__all__ = [
    "Chunk",
    "Collection",
    "Embedder",
    "GeminiEmbedder",
    "RetrievedChunk",
    "Retriever",
    "TfidfEmbedder",
    "VectorStore",
    "build_embedder",
    "build_store",
    "get_retriever",
    "load_all_chunks",
    "load_document",
    "load_domain_chunks",
    "load_retriever",
    "reset_retriever_cache",
]
