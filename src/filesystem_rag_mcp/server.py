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
import contextlib
import json
from pathlib import Path
from typing import Any

from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .config import Settings
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
from .fetcher import fetch_csv_data, fetch_json_data, fetch_lines, fetch_sqlite_query
from .fulltext import FullTextStore
from .git_ops import git_search
from .graph import CorpusGraphBuilder
from .grep import grep_search
from .indexing import Chunk, chunk_file, discover_files
from .logging_setup import get_logger
from .orchestrator import ContextOrchestrator
from .query_cache import QueryCache
from .search import SearchEngine, SearchHit
from .security import PathSecurityError, safe_resolve
from .symbols import search_symbols
from .tree import list_directory
from .vector import Embedder, VectorStore
from .watcher import DirectoryWatcher

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

    import atexit

    atexit.register(state.stop)

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
        fuzzy: bool = False,
        wait_for_indexing: bool = False,
    ) -> dict[str, Any]:
        return await state.search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            path_glob=path_glob,
            rerank=rerank,
            fuzzy=fuzzy,
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

    @server.tool(
        name="grep_search",
        description=(
            "Exact regex or substring search across files in the workspace. "
            "Returns matching lines with context lines, line numbers, and file paths. "
            "Supports path_glob filtering (e.g. 'src/**/*.py') and case sensitivity."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def grep_search_tool(
        pattern: str,
        path_glob: str | None = None,
        max_matches: int = 100,
        case_sensitive: bool = False,
        context_lines: int = 2,
        sub_dir: str = "",
    ) -> dict[str, Any]:
        return state.grep(
            pattern=pattern,
            path_glob=path_glob,
            max_matches=max_matches,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            sub_dir=sub_dir,
        )

    @server.tool(
        name="read_files_batch",
        description=(
            "Inspect multiple files concurrently in a single roundtrip. "
            "Returns a mapping of relative paths to contents or structured error objects. "
            "Supports Markdown conversion (`as_markdown=True`) or raw text."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def read_files_batch_tool(
        rel_paths: list[str],
        as_markdown: bool = True,
        max_bytes_per_file: int | None = None,
    ) -> dict[str, Any]:
        return await state.read_files_batch(
            rel_paths=rel_paths,
            as_markdown=as_markdown,
            max_bytes_per_file=max_bytes_per_file,
        )

    @server.tool(
        name="refresh_file",
        description=(
            "Incrementally (re-)chunk and re-index a single specific file into full-text and vector stores "
            "in <50ms without walking the rest of the workspace."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
    )
    async def refresh_file_tool(
        rel_path: str,
    ) -> dict[str, Any]:
        return await state.refresh_file(rel_path=rel_path)

    @server.tool(
        name="get_chunk_context",
        description=(
            "Fetch contextual neighbor chunks surrounding a chunk_id from the same file. "
            "Expands awareness of the document before and after a search hit."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def get_chunk_context_tool(
        chunk_id: str,
        before_chunks: int = 1,
        after_chunks: int = 1,
    ) -> dict[str, Any]:
        return await state.get_chunk_context(
            chunk_id=chunk_id,
            before_chunks=before_chunks,
            after_chunks=after_chunks,
        )

    @server.tool(
        name="deep_search",
        description=(
            "Multi-hop search decomposing complex research questions across multiple topics/files. "
            "Runs parallel subquery searches, clusters findings, and merges deduplicated chunks."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def deep_search_tool(
        query: str,
        sub_queries: list[str] | None = None,
        top_k_per_subquery: int = 5,
        alpha: float = 0.5,
        rerank: bool = True,
    ) -> dict[str, Any]:
        return await state.deep_search(
            query=query,
            sub_queries=sub_queries,
            top_k_per_subquery=top_k_per_subquery,
            alpha=alpha,
            rerank=rerank,
        )

    @server.tool(
        name="pack_context",
        description=(
            "Assemble a clean, token-bounded Markdown prompt context pack from search hits. "
            "Groups contiguous chunks by file, dedupes, and trims to strict token budgets (e.g. 4000 tokens)."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def pack_context_tool(
        query: str,
        max_tokens: int = 4000,
        alpha: float = 0.5,
        rerank: bool = True,
        path_glob: str | None = None,
    ) -> dict[str, Any]:
        return await state.pack_context(
            query=query,
            max_tokens=max_tokens,
            alpha=alpha,
            rerank=rerank,
            path_glob=path_glob,
        )

    @server.tool(
        name="get_corpus_graph",
        description=(
            "Construct an architectural dependency and reference topology graph of the workspace. "
            "Detects central architectural hubs (highest in-degree), file dependencies, and orphan files."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def get_corpus_graph_tool(
        sub_dir: str = "",
        max_files: int = 500,
    ) -> dict[str, Any]:
        return state.get_corpus_graph(sub_dir=sub_dir, max_files=max_files)

    @server.tool(
        name="git_search",
        description=(
            "Inspect git commit logs, recent changes, commit diffs, or line blame across repository files. "
            "Modes: 'commits', 'recent_changes', 'diff', 'blame'."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def git_search_tool(
        mode: str = "commits",
        query: str | None = None,
        rel_path: str | None = None,
        limit: int = 20,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        return state.git_search(
            mode=mode,
            query=query,
            rel_path=rel_path,
            limit=limit,
            line_start=line_start,
            line_end=line_end,
        )

    @server.tool(
        name="search_symbols",
        description=(
            "Search for function and class declarations/definitions across Python, JS/TS, and generic code. "
            "Returns symbol names, lines, parameters, and docstrings without full text scanning."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def search_symbols_tool(
        name: str = "",
        symbol_type: str = "all",
        path_glob: str | None = None,
        max_matches: int = 100,
        sub_dir: str = "",
        workspace: str = "default",
    ) -> dict[str, Any]:
        return state.search_symbols(
            name=name,
            symbol_type=symbol_type,
            path_glob=path_glob,
            max_matches=max_matches,
            sub_dir=sub_dir,
            workspace=workspace,
        )

    @server.tool(
        name="add_workspace",
        description="Register an additional workspace directory for multi-root monorepos or polyrepos.",
        annotations=ToolAnnotations(readOnlyHint=False),
    )
    async def add_workspace_tool(name: str, path: str) -> dict[str, Any]:
        return await state.add_workspace(name=name, path=path)

    @server.tool(
        name="list_workspaces",
        description="List all registered workspace roots and their status.",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def list_workspaces_tool() -> dict[str, Any]:
        return state.list_workspaces()

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
        self._query_cache = QueryCache(max_entries=1000, ttl_seconds=300.0)
        self._workspaces: dict[str, Path] = {
            "default": Path(settings.root_dir).resolve(),
        }
        self._workspaces_lock = asyncio.Lock()
        self._watcher = DirectoryWatcher(
            Path(settings.root_dir).resolve(),
            self._run_watcher_refresh,
        )

    def stop(self) -> None:
        """Explicitly shut down background tasks and watcher."""
        if self._watcher:
            self._watcher.stop()
        if self._background_quick_task and not self._background_quick_task.done():
            self._background_quick_task.cancel()
        if self._background_thorough_task and not self._background_thorough_task.done():
            self._background_thorough_task.cancel()
        log.info("server_state_stopped")

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
        target_task = (
            self._background_thorough_task if require_thorough else self._background_quick_task
        )

        if target_task and not target_task.done():
            log.info("awaiting_indexing_for_tool", require_thorough=require_thorough)
            await target_ready.wait()
        elif not target_ready.is_set():
            async with self._index_lock:
                if not target_ready.is_set():
                    await self._refresh_index_internal(
                        full_rebuild=False, vector_index=require_thorough
                    )
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
        fuzzy: bool = False,
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
            is_indexing = (
                self._background_quick_task and not self._background_quick_task.done()
            ) or (self._background_thorough_task and not self._background_thorough_task.done())

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

            # Check query cache first if indexing is not active
            cached = self._query_cache.get(
                query=query,
                top_k=top_k,
                alpha=applied_alpha,
                path_glob=path_glob,
                rerank=rerank,
                fuzzy=fuzzy,
            )
            if cached is not None:
                cached_res = dict(cached)
                cached_res["cached"] = True
                return cached_res

            hits = self._engine.search(
                query,
                top_k=top_k,
                alpha=applied_alpha,
                path_glob=path_glob,
                rerank=rerank,
                fuzzy=fuzzy,
            )
            response = {
                "success": True,
                "query": query,
                "cached": False,
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
            self._query_cache.set(
                response,
                query=query,
                top_k=top_k,
                alpha=applied_alpha,
                path_glob=path_glob,
                rerank=rerank,
                fuzzy=fuzzy,
            )
            return response
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
                # If index_binary_vectors is False, skip non-text fallback binary files in vector store
                if not self.settings.index_binary_vectors:
                    type_info = detect_file_type(fm.abs_path)
                    if not type_info.is_text and not type_info.is_convertible:
                        pass
                    else:
                        self._vec.upsert(chunks)
                else:
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
        self._query_cache.invalidate()
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
                return _hit_to_dict_full(
                    hit.chunk_id, hit.rel_path, hit.file_path, hit.start, hit.end, hit.text
                )
        # Fallback: vector store via document lookup
        assert self._vec is not None
        result = self._vec._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
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

    async def read_file(self, *, rel_path: str, max_bytes: int | None) -> dict[str, Any]:
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
            markdown = convert_file_to_markdown(
                resolved, max_bytes=self.settings.max_convert_file_bytes
            )
            return {
                "rel_path": str(resolved.relative_to(root)),
                "abs_path": str(resolved),
                "markdown": markdown,
                "length_chars": len(markdown),
            }
        except Exception as exc:
            type_info = detect_file_type(resolved)
            return conversion_error(rel_path, type_info.label, str(exc))

    async def download_file_raw(self, *, rel_path: str, max_bytes: int | None) -> dict[str, Any]:
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
                "max_depth",
                max_depth,
                "max_depth must be between 0 and 10",
                "Use a depth from 0 to 10.",
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

    def grep(
        self,
        *,
        pattern: str,
        path_glob: str | None = None,
        max_matches: int = 100,
        case_sensitive: bool = False,
        context_lines: int = 2,
        sub_dir: str = "",
    ) -> dict[str, Any]:
        root = Path(self.settings.root_dir).resolve()
        return grep_search(
            root,
            pattern=pattern,
            path_glob=path_glob,
            max_matches=max_matches,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            sub_dir=sub_dir,
        )

    async def read_files_batch(
        self,
        *,
        rel_paths: list[str],
        as_markdown: bool = True,
        max_bytes_per_file: int | None = None,
    ) -> dict[str, Any]:
        if not rel_paths:
            return invalid_parameter_error(
                "rel_paths",
                rel_paths,
                "rel_paths list cannot be empty",
                "Provide a list of relative file paths to inspect, e.g. ['src/app.py', 'README.md'].",
            )
        if len(rel_paths) > 50:
            return invalid_parameter_error(
                "rel_paths",
                len(rel_paths),
                "rel_paths exceeds batch limit of 50 files",
                "Request at most 50 files per batch call.",
            )

        async def _inspect_one(p: str) -> tuple[str, dict[str, Any]]:
            if as_markdown:
                res = await self.read_file_markdown(rel_path=p)
            else:
                res = await self.read_file(rel_path=p, max_bytes=max_bytes_per_file)
            return p, res

        results = await asyncio.gather(*[_inspect_one(p) for p in rel_paths])
        return {
            "success": True,
            "count": len(results),
            "files": dict(results),
        }

    async def refresh_file(self, *, rel_path: str) -> dict[str, Any]:
        root = Path(self.settings.root_dir).resolve()
        try:
            resolved = safe_resolve(root, rel_path)
        except PathSecurityError as exc:
            return path_traversal_error(rel_path, str(root), str(exc))

        if not resolved.exists():
            return file_not_found_error(rel_path, str(root))
        if not resolved.is_file():
            return not_a_file_error(rel_path)

        import hashlib

        from .detector import detect_file_type
        from .indexing import FileMeta, chunk_file
        from .security import is_indexable_file

        if not is_indexable_file(resolved, allow_binary=self.settings.index_binary_files):
            return {
                "success": False,
                "rel_path": rel_path,
                "message": "File is not indexable based on current settings",
            }

        st = resolved.stat()
        h = hashlib.sha256()
        with resolved.open("rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        fm = FileMeta(
            rel_path=str(resolved.relative_to(root)),
            abs_path=resolved,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            sha256=h.hexdigest(),
        )

        chunks = chunk_file(fm, self.settings)

        self._ensure()
        assert self._ft is not None and self._vec is not None

        # Clean existing chunks for this rel_path
        self._ft.delete_by_rel_path(fm.rel_path)
        self._vec.delete_by_rel_path(fm.rel_path)

        if chunks:
            self._ft.upsert(chunks)
            if self.settings.index_binary_vectors:
                self._vec.upsert(chunks)
            else:
                type_info = detect_file_type(resolved)
                if type_info.is_text or type_info.is_convertible:
                    self._vec.upsert(chunks)

        return {
            "success": True,
            "rel_path": fm.rel_path,
            "chunks_indexed": len(chunks),
            "size_bytes": fm.size,
        }

    async def get_chunk_context(
        self, *, chunk_id: str, before_chunks: int = 1, after_chunks: int = 1
    ) -> dict[str, Any]:
        center = await self.get_chunk(chunk_id)
        if not center.get("rel_path"):
            return center

        rel_path = center["rel_path"]

        # Retrieve all chunks for this relative path
        self._ensure()
        assert self._ft is not None
        all_hits = self._ft.search(f'rel_path:"{rel_path}"', top_k=200)

        # Filter to exact file and sort by start offset
        file_chunks = [h for h in all_hits if h.rel_path == rel_path]
        file_chunks.sort(key=lambda x: x.start)

        center_idx = -1
        for idx, ch in enumerate(file_chunks):
            if ch.chunk_id == chunk_id:
                center_idx = idx
                break

        if center_idx == -1:
            return {
                "chunk_id": chunk_id,
                "center": center,
                "before": [],
                "after": [],
            }

        start_b = max(0, center_idx - before_chunks)
        end_a = min(len(file_chunks), center_idx + after_chunks + 1)

        before = [
            _hit_to_dict_full(c.chunk_id, c.rel_path, c.file_path, c.start, c.end, c.text)
            for c in file_chunks[start_b:center_idx]
        ]
        after = [
            _hit_to_dict_full(c.chunk_id, c.rel_path, c.file_path, c.start, c.end, c.text)
            for c in file_chunks[center_idx + 1 : end_a]
        ]

        return {
            "success": True,
            "chunk_id": chunk_id,
            "rel_path": rel_path,
            "center": center,
            "before": before,
            "after": after,
            "total_file_chunks": len(file_chunks),
        }

    async def deep_search(
        self,
        *,
        query: str,
        sub_queries: list[str] | None = None,
        top_k_per_subquery: int = 5,
        alpha: float = 0.5,
        rerank: bool = True,
    ) -> dict[str, Any]:
        self._ensure()
        assert self._engine is not None
        orchestrator = ContextOrchestrator(self._engine)
        return orchestrator.deep_search(
            query=query,
            sub_queries=sub_queries,
            top_k_per_subquery=top_k_per_subquery,
            alpha=alpha,
            rerank=rerank,
        )

    async def pack_context(
        self,
        *,
        query: str,
        max_tokens: int = 4000,
        alpha: float = 0.5,
        rerank: bool = True,
        path_glob: str | None = None,
    ) -> dict[str, Any]:
        self._ensure()
        assert self._engine is not None
        orchestrator = ContextOrchestrator(self._engine)
        return orchestrator.pack_context(
            query=query,
            max_tokens=max_tokens,
            alpha=alpha,
            rerank=rerank,
            path_glob=path_glob,
        )

    def get_corpus_graph(self, *, sub_dir: str = "", max_files: int = 500) -> dict[str, Any]:
        builder = CorpusGraphBuilder(self.settings.root_dir)
        return builder.build_graph(sub_dir=sub_dir, max_files=max_files)

    def git_search(
        self,
        *,
        mode: str = "commits",
        query: str | None = None,
        rel_path: str | None = None,
        limit: int = 20,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        return git_search(
            self.settings.root_dir,
            mode=mode,
            query=query,
            rel_path=rel_path,
            limit=limit,
            line_start=line_start,
            line_end=line_end,
        )

    def search_symbols(
        self,
        *,
        name: str = "",
        symbol_type: str = "all",
        path_glob: str | None = None,
        max_matches: int = 100,
        sub_dir: str = "",
        workspace: str = "default",
    ) -> dict[str, Any]:
        ws_root = self._workspaces.get(workspace)
        if not ws_root:
            from .errors import invalid_parameter_error

            return invalid_parameter_error(
                "workspace",
                workspace,
                f"Workspace '{workspace}' is not registered. Available: {list(self._workspaces.keys())}",
                "Select a registered workspace from `list_workspaces()` or add one via `add_workspace()`.",
            )
        return search_symbols(
            ws_root,
            name=name,
            symbol_type=symbol_type,
            path_glob=path_glob,
            max_matches=max_matches,
            sub_dir=sub_dir,
        )

    async def add_workspace(self, *, name: str, path: str) -> dict[str, Any]:
        p = Path(path).resolve()
        if not p.exists() or not p.is_dir():
            from .errors import path_traversal_error

            return path_traversal_error(
                path,
                "Workspace path does not exist or is not a directory.",
                "Ensure target workspace path exists and is an accessible directory.",
            )
        clean_name = name.strip()
        if not clean_name:
            from .errors import invalid_parameter_error

            return invalid_parameter_error(
                "name", name, "Workspace name cannot be empty.", "Provide a valid name."
            )
        async with self._workspaces_lock:
            self._workspaces[clean_name] = p
            all_ws = list(self._workspaces.keys())
        return {
            "success": True,
            "workspace": clean_name,
            "path": str(p),
            "all_workspaces": all_ws,
        }

    def list_workspaces(self) -> dict[str, Any]:
        return {
            "success": True,
            "workspaces": {
                name: {"path": str(path), "exists": path.exists()}
                for name, path in self._workspaces.items()
            },
        }

    async def get_indexing_status(
        self, wait: bool = False, timeout_seconds: float = 30.0
    ) -> dict[str, Any]:
        """Check live background indexing state, with optional caller wait."""
        if wait:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._thorough_index_ready.wait(), timeout=timeout_seconds)

        quick_ready = self._quick_index_ready.is_set()
        thorough_ready = self._thorough_index_ready.is_set()
        quick_running = bool(self._background_quick_task and not self._background_quick_task.done())
        thorough_running = bool(
            self._background_thorough_task and not self._background_thorough_task.done()
        )

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
        except Exception:
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


__all__ = ["Chunk", "build_server"]
