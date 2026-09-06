"""The MCP server: tools, resources, prompts.

Exposes a single tool, `search`, plus a `refresh_index` admin tool. Both
are guarded by path-validating argument handlers. Resources provide
introspection into the index state.

Protocol version advertised: MCP 2026-07-28 (see __init__.py for the
date-correction note).
"""

from __future__ import annotations

import json
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

    @server.tool(
        name="search",
        description=(
            "Hybrid full-text + vector search over the indexed filesystem. "
            "Returns the top-k most relevant chunks with their file path, "
            "character offset range, and a snippet of text."
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def search_tool(
        query: str,
        top_k: int | None = None,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        return await state.search(query=query, top_k=top_k, alpha=alpha)

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
        self, *, query: str, top_k: int | None, alpha: float | None
    ) -> dict[str, Any]:
        self._ensure()
        if not query or not query.strip():
            return {"query": query, "results": []}
        assert self._engine is not None
        hits = self._engine.search(query, top_k=top_k, alpha=alpha)
        return {
            "query": query,
            "top_k": top_k or self.settings.default_top_k,
            "alpha": alpha if alpha is not None else self.settings.hybrid_alpha,
            "results": [_hit_to_dict(h) for h in hits],
        }

    async def refresh_index(self, *, full_rebuild: bool) -> dict[str, Any]:
        self._ensure()
        assert self._ft is not None and self._vec is not None
        log.info("refresh_index_start", full_rebuild=full_rebuild, root=str(self.settings.root_dir))
        files = discover_files(self.settings)
        # On full rebuild, clear both indexes first.
        if full_rebuild:
            for cid in list(self._vec.all_chunk_ids()):
                self._vec.delete_by_chunk_id(cid)
            for cid in list(self._ft.all_chunk_ids()):
                self._ft.delete_by_chunk_id(cid)
        # Index files: chunk -> upsert into both stores
        new_chunk_ids: set[str] = set()
        new_rel_paths: set[str] = set()
        indexed_files = 0
        indexed_chunks = 0
        for fm in files:
            chunks = chunk_file(fm, self.settings)
            if not chunks:
                continue
            self._ft.upsert(chunks)
            self._vec.upsert(chunks)
            new_chunk_ids.update(c.chunk_id for c in chunks)
            new_rel_paths.add(fm.rel_path)
            indexed_files += 1
            indexed_chunks += len(chunks)
        # Evict stale chunks whose rel_path is no longer present (or whose
        # chunk_id no longer matches the current set).
        all_text_ids = self._ft.all_chunk_ids()
        all_vec_ids = self._vec.all_chunk_ids()
        for stale_id in all_text_ids - new_chunk_ids:
            self._ft.delete_by_chunk_id(stale_id)
        for stale_id in all_vec_ids - new_chunk_ids:
            self._vec.delete_by_chunk_id(stale_id)
        self._last_refresh_at = _now()
        log.info(
            "refresh_index_done",
            files=indexed_files,
            chunks=indexed_chunks,
            full_rebuild=full_rebuild,
        )
        return {
            "full_rebuild": full_rebuild,
            "files_indexed": indexed_files,
            "chunks_indexed": indexed_chunks,
            "root": str(self.settings.root_dir),
            "completed_at": self._last_refresh_at,
        }

    async def get_chunk(self, chunk_id: str) -> dict[str, Any]:
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
        return {"error": f"chunk {chunk_id!r} not found"}

    async def read_file(
        self, *, rel_path: str, max_bytes: int | None
    ) -> dict[str, Any]:
        try:
            root = self.settings.root_dir.resolve()
            resolved = safe_resolve(root, rel_path)
        except PathSecurityError as exc:
            return {"error": str(exc)}
        if not resolved.exists() or not resolved.is_file():
            return {"error": f"{rel_path!r} does not exist or is not a file"}
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            return {"error": f"read failed: {exc}"}
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


def _hit_to_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "rel_path": hit.rel_path,
        "start": hit.start,
        "end": hit.end,
        "score": round(hit.score, 6),
        "sources": list(hit.sources),
        "snippet": _snippet(hit.text),
    }


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


def _snippet(text: str, limit: int = 240) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _now() -> float:
    import time

    return time.time()


# ---------------------------------------------------------------------------
# Chunk helper re-export
# ---------------------------------------------------------------------------


__all__ = ["build_server", "Chunk"]
