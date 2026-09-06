"""Filesystem walking + text chunking.

The indexer walks `settings.root_dir`, identifies text files, splits each
into overlapping chunks, and emits a stream of `Chunk` records.

Chunking strategy: paragraph-aware, character-budgeted. We split on blank
lines, then greedily pack paragraphs into a chunk until the running length
exceeds `chunk_size`. The previous chunk's tail (its last `overlap`
characters) is repeated at the start of the next chunk so context isn't lost
across boundaries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from .config import Settings
from .security import is_text_file, safe_resolve


@dataclass(slots=True, frozen=True)
class Chunk:
    """A single piece of a file, ready to be embedded and indexed."""

    chunk_id: str
    file_path: str  # absolute, validated path
    rel_path: str  # path relative to root_dir
    start: int  # char offset where chunk text begins in file
    end: int  # char offset where chunk text ends (exclusive)
    text: str

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass(slots=True, frozen=True)
class FileMeta:
    """Lightweight metadata about an indexed file."""

    abs_path: Path
    rel_path: str
    size: int
    mtime_ns: int
    sha256: str


def discover_files(settings: Settings) -> list[FileMeta]:
    """Walk `settings.root_dir`, returning text files only.

    Symlinks that escape root are skipped. Files larger than
    `settings.max_file_bytes` are skipped. Non-text files are skipped.
    Output is sorted by relative path for determinism.
    """
    root = settings.root_dir.resolve()
    out: list[FileMeta] = []
    ignore_patterns = settings.ignore_globs

    def _is_ignored(p: Path) -> bool:
        rel = str(p.relative_to(root))
        return any(fnmatch(rel, pat) for pat in ignore_patterns)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _is_ignored(path):
            continue
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if safe_resolve(root, resolved) != resolved:
            # symlink that escapes root
            continue
        try:
            st = resolved.stat()
        except OSError:
            continue
        if st.st_size > settings.max_file_bytes:
            continue
        if not is_text_file(resolved):
            continue
        try:
            with resolved.open("rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            continue
        out.append(
            FileMeta(
                abs_path=resolved,
                rel_path=str(resolved.relative_to(root)),
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
                sha256=digest,
            )
        )
    return out


def chunk_text(
    text: str, *, chunk_size: int, overlap: int
) -> list[tuple[int, int, str]]:
    """Split `text` into overlapping chunks.

    Returns `(char_start, char_end, body)` triples. `char_start` is the
    character offset of the first character of `body` in the original text;
    `char_end` is one past the last character.

    Implementation: split into paragraphs on blank lines, then greedily pack
    them. When a chunk is emitted we drop its leading `overlap` characters
    (which become the tail of the next chunk's leading paragraph, but
    handled transparently via the loop's `_flush`).
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    paragraphs = _split_paragraphs(text)
    # Build a (start_offset, paragraph_text) list
    starts: list[int] = []
    running = 0
    for para in paragraphs:
        starts.append(running)
        running += len(para) + 2  # "\n\n" joiner

    chunks: list[tuple[int, int, str]] = []
    cursor = 0
    current_parts: list[str] = []
    current_len = 0

    def _flush(end_cursor: int) -> None:
        nonlocal current_parts, current_len
        if not current_parts:
            return
        joined = "\n\n".join(current_parts).strip()
        if joined:
            chunks.append((cursor, end_cursor, joined))
        current_parts = []
        current_len = 0

    for para, start in zip(paragraphs, starts):
        end = start + len(para)
        if current_len + len(para) + 2 > chunk_size and current_parts:
            cursor = max(start - overlap, 0)
            _flush(start)
        current_parts.append(para)
        current_len += len(para) + 2
        if current_len >= chunk_size:
            _flush(end)
            cursor = end - overlap

    _flush(len(text))
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; preserve non-empty paragraphs verbatim."""
    out: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if buf:
                out.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        out.append("\n".join(buf))
    return out


def chunk_file(file_meta: FileMeta, settings: Settings) -> list[Chunk]:
    """Read `file_meta.abs_path` and return its chunks."""
    try:
        text = file_meta.abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    raw_chunks = chunk_text(
        text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    out: list[Chunk] = []
    for start, end, body in raw_chunks:
        cid = _chunk_id(file_meta, start, end, body)
        out.append(
            Chunk(
                chunk_id=cid,
                file_path=str(file_meta.abs_path),
                rel_path=file_meta.rel_path,
                start=start,
                end=end,
                text=body,
            )
        )
    return out


def _chunk_id(file_meta: FileMeta, start: int, end: int, body: str) -> str:
    """Stable, content-aware chunk id."""
    h = hashlib.sha256()
    h.update(file_meta.sha256.encode())
    h.update(b"|")
    h.update(str(start).encode())
    h.update(b"|")
    h.update(str(end).encode())
    h.update(b"|")
    h.update(body[:64].encode("utf-8", errors="replace"))
    return h.hexdigest()[:24]
