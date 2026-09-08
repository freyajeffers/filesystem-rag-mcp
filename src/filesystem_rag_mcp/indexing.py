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
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .converter import convert_file_to_markdown
from .security import is_indexable_file, safe_resolve
from .semantic_chunker import semantic_chunk_text


@dataclass(slots=True, frozen=True)
class Chunk:
    """A single piece of a file, ready to be embedded and indexed.

    Kept as `@dataclass(slots=True, frozen=True)` rather than Pydantic because
    a single indexing pass constructs thousands of these (one per chunk
    across every file in the root tree) and Pydantic's runtime validation
    adds measurable overhead in that hot loop. The trade-off is that this
    type doesn't get free JSON Schema generation; callers that need to
    emit a chunk externally should wrap it (e.g. `ChunkSchema(**dataclasses.asdict(c))`)
    before serializing.
    """

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
    """Lightweight metadata about an indexed file.

    Same hot-path rationale as `Chunk`: `discover_files()` instantiates
    one of these per candidate file during walk, and there is no API
    surface that would benefit from Pydantic validation.
    """

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
        if not is_indexable_file(resolved, allow_binary=settings.index_binary_files):
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


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
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
    """Read and convert `file_meta.abs_path` to Markdown, then return its semantic chunks."""
    try:
        text = convert_file_to_markdown(file_meta.abs_path)
    except Exception:
        try:
            text = file_meta.abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
    raw_chunks = semantic_chunk_text(
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


class ReindexResult(BaseModel):
    """Summary of a full reindex pass.

    Returned by `indexing.reindex()` once per indexing call (not per chunk),
    so Pydantic validation overhead is negligible. Promoted to BaseModel
    because the `index` CLI subcommand dumps this directly to JSON and
    the same shape is also returned to MCP callers as part of
    `refresh_index`.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    files_indexed: int = Field(
        description="Number of files that produced at least one chunk",
        ge=0,
    )
    chunks_indexed: int = Field(
        description="Total number of chunks upserted into the stores",
        ge=0,
    )
    chunks_evicted: int = Field(
        description=(
            "Stale chunks removed because their file was deleted or replaced "
            "since the previous index"
        ),
        ge=0,
    )
    full_rebuild: bool = Field(description="True if the indexes were cleared before this pass")
    vector_index: bool = Field(description="True if vector embeddings were (re)computed this pass")


def reindex(
    settings: Settings,
    ft: Any,  # FullTextStore (avoid import cycle)
    vec: Any,  # VectorStore
    *,
    full_rebuild: bool = False,
    vector_index: bool = True,
    detect_file_type: Any = None,
) -> ReindexResult:
    """Walk `settings.root_dir` and refresh both full-text and vector indexes.

    This is the shared indexing pipeline used by the MCP server's background
    indexing tasks and by the `index` CLI subcommand. It is intentionally
    blocking/synchronous so that CLI invocations don't need a running event
    loop.

    If `full_rebuild` is True, the existing indexes are cleared before any
    new chunks are added. Stale chunks (whose `rel_path` is no longer
    present in the discovered file set) are always evicted at the end.

    When `vector_index` is True, chunks are also inserted into the vector
    store. When False, only the full-text index is updated (the "quick"
    pass used to make text queries ready before embeddings complete).
    """
    files = discover_files(settings)

    # On full rebuild, clear selected indexes first.
    if full_rebuild:
        for cid in list(ft.all_chunk_ids()):
            ft.delete_by_chunk_id(cid)
        if vector_index:
            for cid in list(vec.all_chunk_ids()):
                vec.delete_by_chunk_id(cid)

    new_chunk_ids: set[str] = set()
    indexed_files = 0
    indexed_chunks = 0
    for fm in files:
        chunks = chunk_file(fm, settings)
        if not chunks:
            continue
        ft.upsert(chunks)
        if vector_index:
            if not settings.index_binary_vectors and detect_file_type is not None:
                type_info = detect_file_type(fm.abs_path)
                if type_info.is_text or type_info.is_convertible:
                    vec.upsert(chunks)
            else:
                vec.upsert(chunks)
        new_chunk_ids.update(c.chunk_id for c in chunks)
        indexed_files += 1
        indexed_chunks += len(chunks)

    evicted = 0
    all_text_ids = set(ft.all_chunk_ids())
    for stale_id in all_text_ids - new_chunk_ids:
        ft.delete_by_chunk_id(stale_id)
        evicted += 1
    if vector_index:
        all_vec_ids = set(vec.all_chunk_ids())
        for stale_id in all_vec_ids - new_chunk_ids:
            vec.delete_by_chunk_id(stale_id)
            evicted += 1

    return ReindexResult(
        files_indexed=indexed_files,
        chunks_indexed=indexed_chunks,
        chunks_evicted=evicted,
        full_rebuild=full_rebuild,
        vector_index=vector_index,
    )
