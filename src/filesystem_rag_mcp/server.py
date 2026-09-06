"""The MCP server: tools, resources, prompts.

Exposes a single tool, `search`, plus a `refresh_index` admin tool. Both
are guarded by path-validating argument handlers. Resources provide
introspection into the index state.

Protocol version advertised: MCP 2026-07-28 (see __init__.py for the
date-correction note).
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.types import ToolAnnotations

from .config import Settings
from .fulltext import FullTextStore
from .indexing import Chunk, chunk_file, discover_files
from .logging_setup import get_logger
from .search import SearchEngine, SearchHit
from .fetcher import fetch_csv_data, fetch_json_data, fetch_lines, fetch_sqlite_query
from .tree import list_directory
from .watcher import DirectoryWatcher
from .converter import convert_file_to_markdown
from .detector import detect_file_type
from .errors import (
    chunk_not_found_error,
    conversion_error,
    file_not_found_error,
    file_read_error,
    invalid_parameter_error,
    not_a_file_error,
    path_traversal_error,
    search_error,
)
from .security import PathSecurityError, safe_resolve
from .vector import Embedder, VectorStore

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def build_server(settings: Settings, *, auth_provider: Any | None = None) -> MCPServer:
    """Construct the MCP server.

    `auth_provider` is an `MCPFileRAGAuthProvider` instance. If None and
    `settings.auth_required` is True, a provider is built with the current
    settings. If auth is not required, None is passed to MCPServer.
    """
    # Validate the root directory early so we fail fast on misconfig.
    settings.root_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    if auth_provider is None and settings.auth_required:
        from .oauth import MCPFileRAGAuthProvider

        auth_provider = MCPFileRAGAuthProvider(settings)

    auth_settings = (
        AuthSettings(
            issuer_url=settings.oauth_issuer,  # type: ignore[arg-type]
            resource_server_url=settings.oauth_issuer,  # type: ignore[arg-type]
            client_registration_options=ClientRegistrationOptions(
                enabled=settings.oauth_allow_dynamic_registration,
                valid_scopes=["fs.rag.read", "fs.rag.admin"],
                default_scopes=["fs.rag.read"],
            ),
            required_scopes=["fs.rag.read"],
        )
        if auth_provider is not None
        else None
    )

    server = MCPServer(
        name="filesystem-rag-mcp",
        title="Filesystem RAG",
        version="0.1.0",
        instructions=(
            "Indexes a local filesystem directory and exposes hybrid "
            "(full-text + vector) search via the `search` tool. Use "
            "`refresh_index` to (re)build the index, `get_chunk` to fetch a "
            "specific chunk by id, and `read_file` to retrieve a file's "
            "contents. All paths are validated against the configured root."
        ),
        auth_server_provider=auth_provider,
        auth=auth_settings,
    )

    # Bind dependencies; capture via closures.
    state = _ServerState(settings)
    state.start_background_indexing()

    @server.tool(
        name="search",
        description=(
            "Hybrid full-text + vector search over the indexed filesystem. "
            "Returns the top-k most relevant chunks with their file path, "
            "character offset range, and a snippet of text. Does NOT block on background indexing "
            "by default; immediately searches what is currently available and reports index completeness. "
            "Set `wait_for_indexing=True` to explicitly wait until thorough indexing finishes."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def search_tool(
        query: str,
        top_k: int | None = None,
        alpha: float | None = None,
        path_glob: str | None = None,
        rerank: bool = False,
        wait_for_indexing: bool = False,
    ) -> dict[str, Any]:
        return await state.search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            path_glob=path_glob,
            rerank=rerank,
            wait_for_indexing=wait_for_indexing,
        )

    @server.tool(
        name="get_index_status",
        description=(
            "Inspect the live indexing state: background quick/thorough task status, "
            "indexed text and vector chunk counts, and completion timestamps. "
            "Set `wait=True` and optional `timeout_seconds` to synchronously await index completion."
        ),
    )
    async def get_index_status_tool(
        wait: bool = False,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        return await state.get_indexing_status(wait=wait, timeout_seconds=timeout_seconds)

    @server.tool(
        name="refresh_index",
        description=(
            "Walk the configured root directory, (re-)chunk every text file, "
            "and (re-)embed + index all chunks. Safe to call repeatedly; "
            "stale chunks are evicted. May be slow on large corpora."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
    )
    async def refresh_index_tool(
        full_rebuild: bool = False,
    ) -> dict[str, Any]:
        return await state.refresh_index(full_rebuild=full_rebuild)

    @server.tool(
        name="get_chunk",
        description="Fetch a single chunk by its id. Returns the chunk text and metadata.",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def get_chunk_tool(chunk_id: str) -> dict[str, Any]:
        return await state.get_chunk(chunk_id)

    @server.tool(
        name="read_file",
        description=(
            "Read a file's contents from the configured root. The path is "
            "validated against the root; symlinks that escape are refused."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def read_file_tool(
        rel_path: str,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        return await state.read_file(rel_path=rel_path, max_bytes=max_bytes)

    @server.tool(
        name="read_file_markdown",
        description=(
            "Convert any document (PDF, DOCX, PPTX, XLSX, HTML, IPYNB, CSV, RTF, "
            "JSON, YAML, text, code) to clean Markdown. Validated against root."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def read_file_markdown_tool(
        rel_path: str,
    ) -> dict[str, Any]:
        return await state.read_file_markdown(rel_path=rel_path)

    @server.tool(
        name="download_file_raw",
        description=(
            "Download/read a raw file as base64-encoded bytes with MIME type. "
            "Suitable for downloading binary files, images, PDFs, etc. Validated against root."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def download_file_raw_tool(
        rel_path: str,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        return await state.download_file_raw(rel_path=rel_path, max_bytes=max_bytes)

    @server.tool(
        name="list_directory",
        description=(
            "Explore the sandboxed filesystem tree. Returns directory/file metadata including "
            "size, detected MIME/type, conversion support, and relative path. Supports depth and glob filtering."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def list_directory_tool(
        rel_path: str = "",
        max_depth: int = 2,
        pattern: str | None = None,
        include_files: bool = True,
        include_dirs: bool = True,
        limit: int = 150,
    ) -> dict[str, Any]:
        return state.list_directory(
            rel_path=rel_path,
            max_depth=max_depth,
            pattern=pattern,
            include_files=include_files,
            include_dirs=include_dirs,
            limit=limit,
        )

    @server.tool(
        name="fetch_targeted_data",
        description=(
            "Execute targeted data fetching against structured files. Supports:\n"
            "- SQLite: SQL query via `query` parameter (e.g. 'SELECT * FROM users WHERE role=\"Admin\"')\n"
            "- JSON/JSONL: path expression via `query` parameter (e.g. 'users[0].email' or 'config.db')\n"
            "- CSV/TSV: column filtering, row offsets/limits, and value match filters\n"
            "- Text/Code: line range extraction via `start_line` and `end_line` parameters."
        ),
    )
    async def fetch_targeted_data_tool(
        rel_path: str,
        query: str | None = None,
        columns: list[str] | None = None,
        row_offset: int = 0,
        row_limit: int = 50,
        filter_col: str | None = None,
        filter_value: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        return await state.fetch_targeted(
            rel_path=rel_path,
            query=query,
            columns=columns,
            row_offset=row_offset,
            row_limit=row_limit,
            filter_col=filter_col,
            filter_value=filter_value,
            start_line=start_line,
            end_line=end_line,
        )

    # Resources — index status and metadata
    @server.resource(
        name="index_status",
        uri="fsrag://status",
        description="Current index status: counts, last refresh time, root directory.",
        mime_type="application/json",
    )
    async def index_status_resource() -> str:
        return json.dumps(await state.status(), indent=2)

    @server.resource(
        name="index_settings",
        uri="fsrag://settings",
        description="Effective server settings (non-sensitive).",
        mime_type="application/json",
    )
    async def index_settings_resource() -> str:
        return json.dumps(state.public_settings(), indent=2)

    # Prompts
    @server.prompt(
        name="rag_answer",
        description="Build a grounded answer from the indexed filesystem.",
    )
    async def rag_answer_prompt(query: str, top_k: int = 5) -> str:
        return state.rag_answer_prompt(query=query, top_k=top_k)

    return server


# ---------------------------------------------------------------------------
# Server-side state (lazy)
# ---------------------------------------------------------------------------


class _ServerState:
    """Holds the (lazily-built) stores and runs the tool implementations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ft: FullTextStore | None = None
        self._embedder: Embedder | None = None
        self._vec: VectorStore | None = None
        self._engine: SearchEngine | None = None
        self._last_refresh_at: float | None = None
        self._background_quick_task: asyncio.Task[Any] | None = None
        self._background_thorough_task: asyncio.Task[Any] | None = None
        self._index_lock = asyncio.Lock()
        self._quick_index_ready = asyncio.Event()
        self._thorough_index_ready = asyncio.Event()
        self._watcher = DirectoryWatcher(
            Path(settings.root_dir).resolve(),
            self._run_watcher_refresh,
        )

    def start_background_indexing(self) -> None:
        """Trigger fast quick indexing and thorough deep indexing concurrently on server startup."""
        try:
            loop = asyncio.get_running_loop()
            self._background_quick_task = loop.create_task(self._run_quick_index())
            self._background_thorough_task = loop.create_task(self._run_thorough_index())
            self._watcher.start()
        except RuntimeError:
            # If no running event loop yet (e.g. synchronous init), wait for first async call
            pass

    async def _run_watcher_refresh(self) -> None:
        """Incrementally refresh both indexes after a debounced filesystem change."""
        log.info("watcher_incremental_refresh_start")
        async with self._index_lock:
            await self._refresh_index_internal(full_rebuild=False, vector_index=True)
        self._quick_index_ready.set()
        self._thorough_index_ready.set()
        log.info("watcher_incremental_refresh_complete")

    async def _run_quick_index(self) -> None:
        """Runs the quick full-text initial indexing in the background for fast query readiness."""
        try:
            log.info("initial_quick_index_start", root=str(self.settings.root_dir))
            async with self._index_lock:
                await self._refresh_index_internal(full_rebuild=False, vector_index=False)
            log.info("initial_quick_index_complete", root=str(self.settings.root_dir))
        except Exception as exc:
            log.warning("initial_quick_index_failed", error=str(exc))
        finally:
            self._quick_index_ready.set()

    async def _run_thorough_index(self) -> None:
        """Runs thorough indexing (full vector embeddings + deep scan) alongside quick index."""
        try:
            # Allow quick index to claim the lock first if both start simultaneously
            await asyncio.sleep(0.05)
            # Ensure quick index completes before proceeding with thorough vector pass
            await self._quick_index_ready.wait()
            log.info("thorough_deep_index_start", root=str(self.settings.root_dir))
            async with self._index_lock:
                await self._refresh_index_internal(full_rebuild=False, vector_index=True)
            log.info("thorough_deep_index_complete", root=str(self.settings.root_dir))
        except Exception as exc:
            log.warning("thorough_deep_index_failed", error=str(exc))
        finally:
            self._thorough_index_ready.set()

    async def ensure_indexed_synchronously(self, require_thorough: bool = False) -> None:
        """Ensures indexing is completed synchronously before proceeding with complex tools."""
        target_ready = self._thorough_index_ready if require_thorough else self._quick_index_ready
        target_task = self._background_thorough_task if require_thorough else self._background_quick_task

        if target_task and not target_task.done():
            log.info("awaiting_indexing_for_tool", require_thorough=require_thorough)
            await target_ready.wait()
        elif not target_ready.is_set():
            async with self._index_lock:
                if not target_ready.is_set():
                    await self._refresh_index_internal(full_rebuild=False, vector_index=require_thorough)
                    target_ready.set()

    # ---- lazy initialization ------------------------------------------

    def _ensure(self) -> None:
        if self._engine is not None:
            return
        self._ft = FullTextStore(self.settings)
        self._embedder = Embedder(self.settings)
        self._vec = VectorStore(self.settings, self._embedder)
        self._engine = SearchEngine(self.settings, self._ft, self._vec)

    # ---- tool handlers -------------------------------------------------

    async def search(
        self,
        *,
        query: str,
        top_k: int | None,
        alpha: float | None,
        path_glob: str | None = None,
        rerank: bool = False,
        wait_for_indexing: bool = False,
    ) -> dict[str, Any]:
        if not query or not query.strip():
            return invalid_parameter_error(
                "query",
                query,
                "Query string cannot be empty",
                "Provide a non-empty search string with relevant keywords or questions.",
            )
        if top_k is not None and (top_k <= 0 or top_k > 500):
            return invalid_parameter_error(
                "top_k",
                top_k,
                "top_k must be between 1 and 500",
                "Specify a top_k integer between 1 and 500, e.g., top_k=10.",
            )
        if alpha is not None and not (0.0 <= alpha <= 1.0):
            return invalid_parameter_error(
                "alpha",
                alpha,
                "alpha must be between 0.0 and 1.0",
                "Specify alpha in range [0.0, 1.0], where 0.0 is pure vector and 1.0 is pure BM25 full-text.",
            )
        try:
            effective_alpha = alpha if alpha is not None else self.settings.hybrid_alpha

            # Check if caller wants to wait for thorough indexing
            if wait_for_indexing:
                needs_vector = effective_alpha > 0.0
                await self.ensure_indexed_synchronously(require_thorough=needs_vector)

            # Do NOT block if wait_for_indexing is False: use what is currently available
            quick_ready = self._quick_index_ready.is_set()
            thorough_ready = self._thorough_index_ready.is_set()
            is_indexing = (self._background_quick_task and not self._background_quick_task.done()) or (
                self._background_thorough_task and not self._background_thorough_task.done()
            )

            self._ensure()
            assert self._engine is not None

            # Fallback to pure text search if vector index is still compiling and alpha > 0
            applied_alpha = effective_alpha
            indexing_note = None
            if effective_alpha > 0.0 and not thorough_ready:
                applied_alpha = 0.0  # Search with available text index immediately
                indexing_note = (
                    "Background thorough vector indexing is currently in progress. "
                    "Results were generated using available full-text indexing without blocking. "
                    "Pass `wait_for_indexing=True` or check `get_index_status` if semantic vector ranking is required."
                )

            hits = self._engine.search(
                query,
                top_k=top_k,
                alpha=applied_alpha,
                path_glob=path_glob,
                rerank=rerank,
            )
            return {
                "success": True,
                "query": query,
                "top_k": top_k or self.settings.default_top_k,
                "alpha": applied_alpha,
                "requested_alpha": effective_alpha,
                "path_glob": path_glob,
                "rerank_requested": rerank,
                "results": [_hit_to_dict(h, query=query) for h in hits],
                "index_state": {
                    "indexing_in_progress": bool(is_indexing),
                    "quick_index_ready": quick_ready,
                    "thorough_index_ready": thorough_ready,
                    "notice": indexing_note,
                },
            }
        except Exception as exc:
            return search_error(query, "hybrid", str(exc))

    async def _refresh_index_internal(
        self, *, full_rebuild: bool, vector_index: bool = True
    ) -> dict[str, Any]:
        self._ensure()
        assert self._ft is not None and self._vec is not None
        log.info(
            "refresh_index_start",
            full_rebuild=full_rebuild,
            vector_index=vector_index,
            root=str(self.settings.root_dir),
        )
        files = discover_files(self.settings)
        # On full rebuild, clear selected indexes first.
        if full_rebuild:
            for cid in list(self._ft.all_chunk_ids()):
                self._ft.delete_by_chunk_id(cid)
            if vector_index:
                for cid in list(self._vec.all_chunk_ids()):
                    self._vec.delete_by_chunk_id(cid)

        # Index files: chunk -> upsert into stores
        new_chunk_ids: set[str] = set()
        new_rel_paths: set[str] = set()
        indexed_files = 0
        indexed_chunks = 0
        for fm in files:
            chunks = chunk_file(fm, self.settings)
            if not chunks:
                continue
            self._ft.upsert(chunks)
            if vector_index:
                self._vec.upsert(chunks)
            new_chunk_ids.update(c.chunk_id for c in chunks)
            new_rel_paths.add(fm.rel_path)
            indexed_files += 1
            indexed_chunks += len(chunks)

        # Evict stale chunks whose rel_path is no longer present
        all_text_ids = self._ft.all_chunk_ids()
        for stale_id in all_text_ids - new_chunk_ids:
            self._ft.delete_by_chunk_id(stale_id)

        if vector_index:
            all_vec_ids = self._vec.all_chunk_ids()
            for stale_id in all_vec_ids - new_chunk_ids:
                self._vec.delete_by_chunk_id(stale_id)

        self._last_refresh_at = _now()
        log.info(
            "refresh_index_done",
            files=indexed_files,
            chunks=indexed_chunks,
            full_rebuild=full_rebuild,
            vector_index=vector_index,
        )
        return {
            "full_rebuild": full_rebuild,
            "vector_index": vector_index,
            "files_indexed": indexed_files,
            "chunks_indexed": indexed_chunks,
            "root": str(self.settings.root_dir),
            "completed_at": self._last_refresh_at,
        }

    async def refresh_index(self, *, full_rebuild: bool) -> dict[str, Any]:
        async with self._index_lock:
            res = await self._refresh_index_internal(full_rebuild=full_rebuild, vector_index=True)
            self._quick_index_ready.set()
            self._thorough_index_ready.set()
            return res

    async def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        await self.ensure_indexed_synchronously()
        self._ensure()
        assert self._engine is not None
        for hit in self._engine.ft.search(f"chunk_id:{chunk_id}", top_k=1):
            if hit.chunk_id == chunk_id:
                return _hit_to_dict_full(hit.chunk_id, hit.rel_path, hit.file_path,
                                          hit.start, hit.end, hit.text)
        # Fallback: vector store via document lookup
        assert self._vec is not None
        result = self._vec._collection.get(ids=[chunk_id], include=["documents", "metadatas"])  # noqa: SLF001
        if result["ids"]:
            meta = result["metadatas"][0] or {}
            return {
                "chunk_id": chunk_id,
                "rel_path": str(meta.get("rel_path", "")),
                "file_path": str(meta.get("file_path", "")),
                "start": int(meta.get("start", 0)),
                "end": int(meta.get("end", 0)),
                "text": str(result["documents"][0]) if result["documents"] else "",
            }
        return chunk_not_found_error(chunk_id)

    async def read_file(
        self, *, rel_path: str, max_bytes: int | None
    ) -> dict[str, Any]:
        root = self.settings.root_dir.resolve()
        try:
            resolved = safe_resolve(root, rel_path)
        except PathSecurityError as exc:
            return path_traversal_error(rel_path, str(root), str(exc))
        if not resolved.exists():
            return file_not_found_error(rel_path, str(root))
        if not resolved.is_file():
            return not_a_file_error(rel_path)
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            return file_read_error(rel_path, str(exc))
        if max_bytes is not None and len(data) > max_bytes:
            data = data[:max_bytes]
            truncated = True
        else:
            truncated = False
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        return {
            "rel_path": str(resolved.relative_to(root)),
            "abs_path": str(resolved),
            "size_bytes": len(data),
            "truncated": truncated,
            "text": text,
        }

    async def read_file_markdown(self, *, rel_path: str) -> dict[str, Any]:
        """Convert any supported file type to clean Markdown."""
        root = self.settings.root_dir.resolve()
        try:
            resolved = safe_resolve(root, rel_path)
        except PathSecurityError as exc:
            return path_traversal_error(rel_path, str(root), str(exc))
        if not resolved.exists():
            return file_not_found_error(rel_path, str(root))
        if not resolved.is_file():
            return not_a_file_error(rel_path)
        try:
            markdown = convert_file_to_markdown(resolved)
            return {
                "rel_path": str(resolved.relative_to(root)),
                "abs_path": str(resolved),
                "markdown": markdown,
                "length_chars": len(markdown),
            }
        except Exception as exc:
            type_info = detect_file_type(resolved)
            return conversion_error(rel_path, type_info.label, str(exc))

    async def download_file_raw(
        self, *, rel_path: str, max_bytes: int | None
    ) -> dict[str, Any]:
        """Download raw file as base64 with MIME type."""
        root = self.settings.root_dir.resolve()
        try:
            resolved = safe_resolve(root, rel_path)
        except PathSecurityError as exc:
            return path_traversal_error(rel_path, str(root), str(exc))
        if not resolved.exists():
            return file_not_found_error(rel_path, str(root))
        if not resolved.is_file():
            return not_a_file_error(rel_path)
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            return file_read_error(rel_path, str(exc))

        total_size = len(data)
        truncated = False
        if max_bytes is not None and total_size > max_bytes:
            data = data[:max_bytes]
            truncated = True

        type_info = detect_file_type(resolved)
        mime_type = type_info.mime_type or "application/octet-stream"

        encoded = base64.b64encode(data).decode("ascii")
        return {
            "rel_path": str(resolved.relative_to(root)),
            "abs_path": str(resolved),
            "mime_type": mime_type,
            "detected_label": type_info.label,
            "total_size_bytes": total_size,
            "returned_size_bytes": len(data),
            "truncated": truncated,
            "base64_data": encoded,
        }

    def list_directory(
        self,
        *,
        rel_path: str = "",
        max_depth: int = 2,
        pattern: str | None = None,
        include_files: bool = True,
        include_dirs: bool = True,
        limit: int = 150,
    ) -> dict[str, Any]:
        if not 0 <= max_depth <= 10:
            return invalid_parameter_error(
                "max_depth", max_depth, "max_depth must be between 0 and 10", "Use a depth from 0 to 10."
            )
        if not 1 <= limit <= 1000:
            return invalid_parameter_error(
                "limit", limit, "limit must be between 1 and 1000", "Use a limit from 1 to 1000."
            )
        return list_directory(
            Path(self.settings.root_dir).resolve(),
            rel_path,
            max_depth=max_depth,
            pattern=pattern,
            include_files=include_files,
            include_dirs=include_dirs,
            limit=limit,
        )

    async def fetch_targeted(
        self,
        *,
        rel_path: str,
        query: str | None = None,
        columns: list[str] | None = None,
        row_offset: int = 0,
        row_limit: int = 50,
        filter_col: str | None = None,
        filter_value: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        root = Path(self.settings.root_dir).resolve()
        try:
            resolved = safe_resolve(root, rel_path)
        except PathSecurityError as exc:
            return path_traversal_error(rel_path, str(root), str(exc))

        if not resolved.exists():
            return file_not_found_error(rel_path, str(root))
        if not resolved.is_file():
            return not_a_file_error(rel_path)

        type_info = detect_file_type(resolved)
        label = type_info.label.lower()
        ext = resolved.suffix.lower()

        # Line-based fetching
        if start_line is not None or end_line is not None:
            s_line = start_line if start_line is not None else 1
            e_line = end_line if end_line is not None else (s_line + 50)
            return fetch_lines(resolved, s_line, e_line)

        # SQLite
        if ext in (".sqlite", ".sqlite3", ".db") or label == "sqlite":
            sql_query = query or "SELECT name FROM sqlite_master WHERE type='table'"
            return fetch_sqlite_query(resolved, sql=sql_query, limit=row_limit)

        # JSON / JSONL
        if label in ("json", "jsonl") or ext in (".json", ".jsonl"):
            return fetch_json_data(resolved, path_expr=query, max_items=row_limit)

        # CSV / TSV
        if label in ("csv", "tsv") or ext in (".csv", ".tsv"):
            return fetch_csv_data(
                resolved,
                columns=columns,
                row_offset=row_offset,
                row_limit=row_limit,
                filter_col=filter_col,
                filter_value=filter_value,
            )

        # Default fallback for text files: lines
        return fetch_lines(resolved, start_line=1, end_line=row_limit)

    async def get_indexing_status(self, wait: bool = False, timeout_seconds: float = 30.0) -> dict[str, Any]:
        """Check live background indexing state, with optional caller wait."""
        if wait:
            try:
                await asyncio.wait_for(self._thorough_index_ready.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                pass

        quick_ready = self._quick_index_ready.is_set()
        thorough_ready = self._thorough_index_ready.is_set()
        quick_running = bool(self._background_quick_task and not self._background_quick_task.done())
        thorough_running = bool(self._background_thorough_task and not self._background_thorough_task.done())

        text_count = vector_count = 0
        try:
            self._ensure()
            assert self._ft is not None and self._vec is not None
            text_count = self._ft.count()
            vector_count = self._vec.count()
        except Exception:
            pass

        return {
            "root": str(self.settings.root_dir),
            "indexing_in_progress": quick_running or thorough_running,
            "quick_index": {
                "in_progress": quick_running,
                "ready": quick_ready,
                "text_chunk_count": text_count,
            },
            "thorough_index": {
                "in_progress": thorough_running,
                "ready": thorough_ready,
                "vector_chunk_count": vector_count,
            },
            "last_refresh_at": self._last_refresh_at,
        }

    async def status(self) -> dict[str, Any]:
        text_count = vector_count = 0
        try:
            self._ensure()
            assert self._ft is not None and self._vec is not None
            text_count = self._ft.count()
            vector_count = self._vec.count()
        except Exception:  # noqa: BLE001 - status should never crash
            pass
        return {
            "root": str(self.settings.root_dir),
            "data_dir": str(self.settings.data_dir),
            "text_chunk_count": text_count,
            "vector_chunk_count": vector_count,
            "last_refresh_at": self._last_refresh_at,
            "embedding_model": self.settings.embedding_model,
            "embedding_dim": self.settings.embedding_dim,
        }

    def public_settings(self) -> dict[str, Any]:
        """Settings safe to expose — no secrets."""
        s = self.settings
        return {
            "root_dir": str(s.root_dir),
            "data_dir": str(s.data_dir),
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "embedding_model": s.embedding_model,
            "embedding_dim": s.embedding_dim,
            "default_top_k": s.default_top_k,
            "hybrid_alpha": s.hybrid_alpha,
            "auth_required": s.auth_required,
            "oauth_allow_dynamic_registration": s.oauth_allow_dynamic_registration,
            "protocol_version": "2026-07-28",
        }

    def rag_answer_prompt(self, *, query: str, top_k: int) -> str:
        return (
            "You are a knowledgeable assistant grounded in the user's local "
            "filesystem corpus, indexed by the filesystem-rag-mcp server.\n\n"
            f"User question: {query}\n\n"
            "Procedure:\n"
            f"1. Call the `search` tool with `query={query!r}` and `top_k={top_k}`.\n"
            "2. Read each returned chunk's `text` and `rel_path`.\n"
            "3. Compose a concise answer that cites the source paths inline as "
            "`[rel_path:start-end]`.\n"
            "4. If the corpus does not contain the answer, say so plainly.\n"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snippet(text: str, query: str = "", limit: int = 300) -> str:
    """Generate a high-signal snippet centered around query terms if present."""
    import re

    cleaned = " ".join(text.split())
    if not query or not query.strip():
        return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "…"

    # Find the earliest matching term
    terms = [re.escape(t) for t in query.strip().split() if len(t) > 2]
    if not terms:
        return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "…"

    pattern = re.compile(r"(" + "|".join(terms) + r")", re.IGNORECASE)
    match = pattern.search(cleaned)
    if not match:
        return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "…"

    start_idx = max(0, match.start() - 100)
    end_idx = min(len(cleaned), start_idx + limit)
    snippet = cleaned[start_idx:end_idx].strip()

    prefix = "… " if start_idx > 0 else ""
    suffix = " …" if end_idx < len(cleaned) else ""
    return f"{prefix}{snippet}{suffix}"


def _hit_to_dict_full(
    chunk_id: str, rel_path: str, file_path: str, start: int, end: int, text: str
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "rel_path": rel_path,
        "file_path": file_path,
        "start": start,
        "end": end,
        "text": text,
    }


def _hit_to_dict(hit: SearchHit, query: str = "") -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "rel_path": hit.rel_path,
        "start": hit.start,
        "end": hit.end,
        "score": round(hit.score, 6),
        "sources": list(hit.sources),
        "snippet": _snippet(hit.text, query=query),
    }


def _now() -> float:
    import time

    return time.time()


# ---------------------------------------------------------------------------
# Chunk helper re-export
# ---------------------------------------------------------------------------


__all__ = ["build_server", "Chunk"]
