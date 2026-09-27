"""Central configuration for the AI Training Assistant.

Every tunable lives here so the rest of the codebase never reads ``os.environ``
directly. Configuration is resolved once, lazily, and cached.

Two execution modes are supported:

``live``
    A Gemini API key is available, so the ADK agents are backed by a real
    Gemini model and the vector index is built from Gemini embeddings.

``offline``
    No key is available (or offline mode is forced). The ADK agent graph is
    still exercised end to end, but the model behind each agent is the
    deterministic :class:`~training_assistant.models.offline_llm.OfflineDemoLlm`
    and the index is built from a local TF-IDF vectoriser. This keeps the
    project reproducible for reviewers who do not have a key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# .../src/training_assistant/config.py -> project root is three parents up.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
INDEX_DIR: Final[Path] = PROJECT_ROOT / "index"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"

#: Maps a knowledge domain to the folder holding its source documents.
#: The keys are the canonical route names used throughout the system.
DOMAIN_DIRS: Final[dict[str, Path]] = {
    "company": DATA_DIR / "company",
    "role": DATA_DIR / "roles",
    "admin": DATA_DIR / "admin",
}

#: Human-readable labels for each route, used in traces and in the UI.
DOMAIN_LABELS: Final[dict[str, str]] = {
    "company": "General company knowledge base",
    "role": "Role & team documents",
    "admin": "Policy & administration documents",
    "general": "Direct LLM response (no retrieval)",
}

EVAL_QUESTIONS_PATH: Final[Path] = DATA_DIR / "eval" / "synthetic_questions.json"

APP_NAME: Final[str] = "ai_training_assistant"

# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #

_TRUTHY = {"1", "true", "yes", "on", "y"}


def _load_dotenv_once() -> None:
    """Load ``.env`` from the project root if python-dotenv is installed.

    Failure is non-fatal: the project must remain runnable when configuration
    comes from real environment variables instead of a file.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    load_dotenv(env_path, override=False)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return _env_str(name, str(default)).lower() in _TRUTHY


def _resolve_api_key() -> str:
    """Return the first non-empty Gemini key from the accepted variable names."""
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Settings:
    """Immutable, fully-resolved runtime configuration."""

    api_key: str
    model: str
    embed_model: str
    embed_dim: int
    top_k: int
    semantic_weight: float
    min_score: float
    force_offline: bool
    use_vertex: bool

    # Derived, filled in __post_init__ style via field default + property below.
    domain_dirs: dict[str, Path] = field(default_factory=lambda: dict(DOMAIN_DIRS))

    # ---------------- derived properties ---------------- #

    @property
    def offline(self) -> bool:
        """True when the deterministic, network-free backend should be used."""
        if self.force_offline:
            return True
        # Vertex AI authenticates via ADC rather than an API key.
        if self.use_vertex:
            return False
        return not self.api_key

    @property
    def mode(self) -> str:
        """``'offline'`` or ``'live'`` - the human-facing mode name."""
        return "offline" if self.offline else "live"

    @property
    def lexical_weight(self) -> float:
        """Weight given to BM25 scoring in the hybrid retrieval score."""
        return max(0.0, 1.0 - self.semantic_weight)

    @property
    def index_path(self) -> Path:
        """Location of the persisted vector index for the current embedder."""
        return INDEX_DIR / f"knowledge_index_{self.embedding_backend}.json"

    @property
    def embedding_backend(self) -> str:
        """Identifier for the active embedding strategy.

        The index filename embeds this value so that switching between offline
        and live embeddings can never silently mix incompatible vectors.
        """
        return "tfidf" if self.offline else "gemini"

    def describe(self) -> str:
        """One-line summary suitable for logging and the UI sidebar."""
        backend = "OfflineDemoLlm (deterministic)" if self.offline else self.model
        return (
            f"mode={self.mode} | model={backend} | "
            f"embeddings={self.embedding_backend} | top_k={self.top_k}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Resolve and cache :class:`Settings` from the environment."""
    _load_dotenv_once()

    api_key = _resolve_api_key()
    use_vertex = _env_bool("GOOGLE_GENAI_USE_VERTEXAI", False)

    # ADK / google-genai read GOOGLE_API_KEY, so mirror whichever name was set.
    if api_key and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = api_key

    semantic_weight = _env_float("TRAINING_ASSISTANT_SEMANTIC_WEIGHT", 0.65)
    semantic_weight = min(max(semantic_weight, 0.0), 1.0)

    return Settings(
        api_key=api_key,
        model=_env_str("TRAINING_ASSISTANT_MODEL", "gemini-3.5-flash"),
        embed_model=_env_str("TRAINING_ASSISTANT_EMBED_MODEL", "gemini-embedding-001"),
        embed_dim=max(64, _env_int("TRAINING_ASSISTANT_EMBED_DIM", 768)),
        top_k=max(1, _env_int("TRAINING_ASSISTANT_TOP_K", 4)),
        semantic_weight=semantic_weight,
        min_score=_env_float("TRAINING_ASSISTANT_MIN_SCORE", 0.05),
        force_offline=_env_bool("TRAINING_ASSISTANT_FORCE_OFFLINE", False),
        use_vertex=use_vertex,
    )


def reset_settings_cache() -> None:
    """Clear the cached settings.

    Used by the evaluation harness and the UI when environment variables are
    changed inside a running process.
    """
    get_settings.cache_clear()
