"""Load the synthetic knowledge base and split it into retrievable chunks.

The corpus is Markdown with a small YAML-ish front-matter block. Chunking is
**heading aware**: each chunk corresponds to one Markdown section and carries its
heading path, so a citation can point a new employee at
"Leave and Time Off Policy > Carry-forward, encashment and expiry" rather than at
an anonymous blob of text.

Long sections are split further on paragraph boundaries, and Markdown tables are
kept intact because splitting a table mid-row destroys its meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..config import DOMAIN_DIRS

# Sections longer than this (in characters) are split on paragraph boundaries.
MAX_CHUNK_CHARS = 1_600

# Chunks shorter than this are merged into the following chunk, to avoid
# indexing fragments like a lone heading.
MIN_CHUNK_CHARS = 120

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")


@dataclass(slots=True)
class Chunk:
    """One retrievable passage of the knowledge base."""

    chunk_id: str
    domain: str
    doc_id: str
    doc_title: str
    heading_path: tuple[str, ...]
    text: str
    source_path: str
    owner: str = ""
    last_reviewed: str = ""
    policy_version: str = ""

    @property
    def citation(self) -> str:
        """Short human-readable source label, used in answers and in the UI."""
        if self.heading_path:
            return f"{self.doc_title} > {' > '.join(self.heading_path)}"
        return self.doc_title

    @property
    def embedding_text(self) -> str:
        """Text handed to the embedder.

        The document title and heading path are prepended so that a section
        called "Notice and approval" is still findable by a query about *leave*
        notice, even though the section body never repeats the word "leave".
        """
        header = " > ".join((self.doc_title, *self.heading_path))
        return f"{header}\n\n{self.text}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "domain": self.domain,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "heading_path": list(self.heading_path),
            "text": self.text,
            "source_path": self.source_path,
            "owner": self.owner,
            "last_reviewed": self.last_reviewed,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Chunk":
        return cls(
            chunk_id=payload["chunk_id"],
            domain=payload["domain"],
            doc_id=payload["doc_id"],
            doc_title=payload["doc_title"],
            heading_path=tuple(payload.get("heading_path") or ()),
            text=payload["text"],
            source_path=payload["source_path"],
            owner=payload.get("owner", ""),
            last_reviewed=payload.get("last_reviewed", ""),
            policy_version=payload.get("policy_version", ""),
        )


@dataclass(slots=True)
class _ParsedDocument:
    metadata: dict[str, str] = field(default_factory=dict)
    body: str = ""


def _parse_front_matter(raw: str) -> _ParsedDocument:
    """Extract the leading ``---`` block as flat ``key: value`` metadata.

    A deliberately minimal parser: the corpus only uses scalar values, so
    depending on PyYAML would add a dependency for no benefit.
    """
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return _ParsedDocument(metadata={}, body=raw)

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip().strip("'\"")

    return _ParsedDocument(metadata=metadata, body=raw[match.end():])


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, but never split inside a Markdown table."""
    blocks: list[str] = []
    buffer: list[str] = []
    in_table = False

    for line in text.splitlines():
        is_table_line = line.lstrip().startswith("|")

        if is_table_line:
            in_table = True
            buffer.append(line)
            continue

        if not line.strip():
            # A blank line ends the current block (table or paragraph).
            if buffer:
                blocks.append("\n".join(buffer).strip())
                buffer = []
            in_table = False
            continue

        if in_table:
            # Non-table line directly after a table: close the table block.
            blocks.append("\n".join(buffer).strip())
            buffer = [line]
            in_table = False
            continue

        buffer.append(line)

    if buffer:
        blocks.append("\n".join(buffer).strip())

    return [block for block in blocks if block]


def _pack_sections(paragraphs: list[str], limit: int) -> list[str]:
    """Greedily pack paragraphs into pieces of at most ``limit`` characters."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0

    for paragraph in paragraphs:
        para_len = len(paragraph)
        if current and size + para_len > limit:
            pieces.append("\n\n".join(current))
            current = [paragraph]
            size = para_len
        else:
            current.append(paragraph)
            size += para_len + 2

    if current:
        pieces.append("\n\n".join(current))

    return pieces


def _iter_sections(body: str) -> Iterable[tuple[tuple[str, ...], str]]:
    """Yield ``(heading_path, section_text)`` pairs for a Markdown body.

    ``heading_path`` excludes the document's single H1 title, which is captured
    separately as the document title.
    """
    # heading_stack[level] = heading text at that level
    heading_stack: dict[int, str] = {}
    current_path: tuple[str, ...] = ()
    buffer: list[str] = []

    def flush() -> Iterable[tuple[tuple[str, ...], str]]:
        text = "\n".join(buffer).strip()
        if text:
            yield current_path, text

    for line in body.splitlines():
        heading_match = _HEADING_RE.match(line)
        if not heading_match:
            buffer.append(line)
            continue

        yield from flush()
        buffer = []

        level = len(heading_match.group(1))
        title = heading_match.group(2).strip()

        heading_stack[level] = title
        # Drop any deeper headings now out of scope.
        for deeper in [lvl for lvl in heading_stack if lvl > level]:
            del heading_stack[deeper]

        # Level 1 is the document title, so it is not part of the path.
        current_path = tuple(
            heading_stack[lvl] for lvl in sorted(heading_stack) if lvl > 1
        )

    yield from flush()


def _merge_tiny_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Fold very short chunks into the next chunk of the same document."""
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    pending: Chunk | None = None

    for chunk in chunks:
        if pending is not None:
            if pending.doc_id == chunk.doc_id:
                chunk.text = f"{pending.text}\n\n{chunk.text}".strip()
                # Keep the shallower heading path so the citation stays useful.
                if len(pending.heading_path) < len(chunk.heading_path):
                    chunk.heading_path = pending.heading_path
            else:
                merged.append(pending)
            pending = None

        if len(chunk.text) < MIN_CHUNK_CHARS:
            pending = chunk
            continue

        merged.append(chunk)

    if pending is not None:
        # Nothing followed it, so attach to the previous chunk if we can.
        if merged and merged[-1].doc_id == pending.doc_id:
            merged[-1].text = f"{merged[-1].text}\n\n{pending.text}".strip()
        else:
            merged.append(pending)

    # Re-number so chunk ids stay dense and stable for a given corpus.
    for position, chunk in enumerate(merged):
        chunk.chunk_id = f"{chunk.doc_id}::{position:03d}"

    return merged


def load_document(path: Path, domain: str) -> list[Chunk]:
    """Load a single Markdown file into chunks."""
    raw = path.read_text(encoding="utf-8")
    parsed = _parse_front_matter(raw)
    metadata = parsed.metadata

    doc_id = metadata.get("doc_id") or path.stem
    doc_title = metadata.get("title") or path.stem.replace("_", " ").title()

    chunks: list[Chunk] = []
    for heading_path, section_text in _iter_sections(parsed.body):
        paragraphs = _split_paragraphs(section_text)
        if not paragraphs:
            continue

        for piece in _pack_sections(paragraphs, MAX_CHUNK_CHARS):
            chunks.append(
                Chunk(
                    chunk_id="",  # assigned after merging
                    domain=domain,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    heading_path=heading_path,
                    text=piece,
                    source_path=path.as_posix(),
                    owner=metadata.get("owner", ""),
                    last_reviewed=metadata.get("last_reviewed", ""),
                    policy_version=metadata.get("policy_version", ""),
                )
            )

    return _merge_tiny_chunks(chunks)


def load_domain_chunks(domain: str) -> list[Chunk]:
    """Load every chunk for one knowledge domain (``company``/``role``/``admin``)."""
    try:
        directory = DOMAIN_DIRS[domain]
    except KeyError as exc:  # pragma: no cover - guarded by callers
        raise ValueError(
            f"Unknown domain {domain!r}. Expected one of {sorted(DOMAIN_DIRS)}."
        ) from exc

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Knowledge directory for domain {domain!r} not found: {directory}"
        )

    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(load_document(path, domain))
    return chunks


def load_all_chunks() -> dict[str, list[Chunk]]:
    """Load every domain, keyed by domain name."""
    return {domain: load_domain_chunks(domain) for domain in DOMAIN_DIRS}
