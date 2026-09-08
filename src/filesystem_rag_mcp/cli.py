"""Command-line entrypoint for filesystem-rag-mcp.

Usage (legacy flag-based server launcher — unchanged for backward compatibility):
  filesystem-rag-mcp --transport stdio --root-dir /path/to/docs
  filesystem-rag-mcp --transport http --host 127.0.0.1 --port 8000 --root-dir /path/to/docs

Subcommand form (new in this release):
  filesystem-rag-mcp doctor [--json] [--root-dir DIR] [--data-dir DIR]
  filesystem-rag-mcp search  QUERY [-- top-k N] [-- alpha F] [--glob PAT] [--fuzzy] [--rerank] [--json]
  filesystem-rag-mcp stats   [--root-dir DIR] [--data-dir DIR] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from . import __version__
from .config import Settings
from .logging_setup import configure_logging, get_logger
from .server import build_server

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------

_SUBCOMMANDS = ("doctor", "search", "stats", "index")


def _argv_uses_subcommand(argv: list[str]) -> str | None:
    """Return the subcommand name if the first positional arg is recognized.

    `argv` here is the program-relative arg list (e.g. ``['doctor']`` or
    ``['--transport', 'stdio']``). We inspect the *first* non-flag token so
    the legacy flag-only path (e.g. ``--transport``) is never accidentally
    parsed as a subcommand. This avoids restructuring the existing
    top-level parser.
    """
    if not argv:
        return None
    if argv[0].startswith("-"):
        return None
    return argv[0] if argv[0] in _SUBCOMMANDS else None


def _build_subcommand_parser(prog: str) -> argparse.ArgumentParser:
    """Build a fresh parser dedicated to the new subcommand surface."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="filesystem-rag-mcp maintenance subcommands",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub_p = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # doctor ---------------------------------------------------------------
    p_doc = sub_p.add_parser(
        "doctor",
        help="Run environment, dependency, and health diagnostics and exit",
    )
    p_doc.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit JSON instead of formatted text"
    )
    p_doc.add_argument("--root-dir", type=Path, default=Path(os.environ.get("FSRAG_ROOT_DIR", ".")))
    p_doc.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("FSRAG_DATA_DIR", ".fsrag"))
    )

    # search ---------------------------------------------------------------
    p_search = sub_p.add_parser(
        "search",
        help="Run a hybrid full-text + vector search over the on-disk index",
    )
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--top-k", type=int, default=None)
    p_search.add_argument(
        "--alpha", type=float, default=None, help="Hybrid weight: 0=text-only, 1=vector-only"
    )
    p_search.add_argument("--glob", dest="path_glob", default=None)
    p_search.add_argument("--fuzzy", action="store_true")
    p_search.add_argument("--rerank", action="store_true")
    p_search.add_argument("--json", action="store_true", dest="as_json")
    p_search.add_argument(
        "--root-dir", type=Path, default=Path(os.environ.get("FSRAG_ROOT_DIR", "."))
    )
    p_search.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("FSRAG_DATA_DIR", ".fsrag"))
    )

    # stats ----------------------------------------------------------------
    p_stats = sub_p.add_parser("stats", help="Print full-text and vector index statistics and exit")
    p_stats.add_argument("--json", action="store_true", dest="as_json")
    p_stats.add_argument(
        "--root-dir", type=Path, default=Path(os.environ.get("FSRAG_ROOT_DIR", "."))
    )
    p_stats.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("FSRAG_DATA_DIR", ".fsrag"))
    )

    # index ----------------------------------------------------------------
    p_index = sub_p.add_parser(
        "index",
        help="Build (or refresh) the on-disk full-text and vector indexes synchronously",
    )
    p_index.add_argument(
        "--thorough",
        action="store_true",
        help="Run the thorough pass (includes vector embeddings); default is the quick BM25 pass",
    )
    p_index.add_argument(
        "--root-dir", type=Path, default=Path(os.environ.get("FSRAG_ROOT_DIR", "."))
    )
    p_index.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("FSRAG_DATA_DIR", ".fsrag"))
    )
    p_index.add_argument("--json", action="store_true", dest="as_json")

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _run_doctor(args: argparse.Namespace) -> int:
    from .doctor import print_doctor_report

    settings = Settings(
        root_dir=args.root_dir.resolve(),
        data_dir=args.data_dir.resolve(),
    )
    return print_doctor_report(settings, as_json=getattr(args, "as_json", False))


def _run_search(args: argparse.Namespace) -> int:
    from .fulltext import FullTextStore
    from .search import SearchEngine
    from .vector import Embedder, VectorStore

    settings = Settings(
        root_dir=args.root_dir.resolve(),
        data_dir=args.data_dir.resolve(),
    )

    ft = FullTextStore(settings)
    embedder = Embedder(settings)
    vec = VectorStore(settings, embedder)
    engine = SearchEngine(settings, ft, vec)

    hits = engine.search(
        args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        path_glob=args.path_glob,
        rerank=args.rerank,
        fuzzy=args.fuzzy,
    )

    if getattr(args, "as_json", False):
        print(
            json.dumps(
                [
                    {
                        "chunk_id": h.chunk_id,
                        "rel_path": h.rel_path,
                        "score": h.score,
                        "sources": list(h.sources),
                        "snippet": h.text[:240],
                    }
                    for h in hits
                ],
                indent=2,
            )
        )
    else:
        if not hits:
            print(f"(no hits for: {args.query!r})")
            return 1
        for i, h in enumerate(hits, start=1):
            tag = "+".join(h.sources) if h.sources else "?"
            snippet = h.text.replace("\n", " ")[:200]
            print(f"[{i:>2}] {h.score:.4f}  {h.rel_path}  ({tag})")
            print(f"     {snippet}")

    return 0


def _run_stats(args: argparse.Namespace) -> int:
    from .fulltext import FullTextStore
    from .vector import Embedder, VectorStore

    settings = Settings(
        root_dir=args.root_dir.resolve(),
        data_dir=args.data_dir.resolve(),
    )

    ft = FullTextStore(settings)
    embedder = Embedder(settings)
    vec = VectorStore(settings, embedder)

    data = {
        "root_dir": str(settings.root_dir),
        "data_dir": str(settings.data_dir),
        "embedding_model": settings.embedding_model,
        "fulltext_chunks": ft.count(),
        "vector_chunks": vec.count(),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "hybrid_alpha": settings.hybrid_alpha,
        "default_top_k": settings.default_top_k,
    }

    if getattr(args, "as_json", False):
        print(json.dumps(data, indent=2))
    else:
        print("=" * 60)
        print("  filesystem-rag-mcp Index Statistics")
        print("=" * 60)
        print(f"  Root directory:        {data['root_dir']}")
        print(f"  Data directory:        {data['data_dir']}")
        print(f"  Embedding model:       {data['embedding_model']}")
        print(f"  Full-text chunks:      {data['fulltext_chunks']}")
        print(f"  Vector chunks:         {data['vector_chunks']}")
        print(f"  Chunk size / overlap:  {data['chunk_size']} / {data['chunk_overlap']}")
        print(f"  Hybrid alpha:          {data['hybrid_alpha']}")
        print(f"  Default top-k:         {data['default_top_k']}")
        print("=" * 60)

    return 0


def _run_index_async(args: argparse.Namespace) -> int:
    """Run a synchronous indexing pass using the shared `indexing.reindex` helper.

    The `index` subcommand is intentionally blocking and side-effecting: it
    populates the on-disk full-text and vector indexes that subsequent
    `search` invocations will read from. We do *not* go through the server's
    `_ServerState` (which is private) — instead we construct the same
    storage primitives the server uses and call the shared helper.
    """
    from .detector import detect_file_type
    from .fulltext import FullTextStore
    from .indexing import reindex as _reindex
    from .vector import Embedder, VectorStore

    settings = Settings(
        root_dir=args.root_dir.resolve(),
        data_dir=args.data_dir.resolve(),
    )

    settings.root_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    thorough = bool(getattr(args, "thorough", False))

    ft = FullTextStore(settings)
    embedder = Embedder(settings)
    vec = VectorStore(settings, embedder)

    log.info(
        "subcommand_index_start",
        thorough=thorough,
        root_dir=str(settings.root_dir),
        embedding_available=embedder.is_available(),
    )

    result = _reindex(
        settings,
        ft,
        vec,
        full_rebuild=False,
        vector_index=thorough,
        detect_file_type=detect_file_type,
    )

    payload = {
        "ok": True,
        "root_dir": str(settings.root_dir),
        "data_dir": str(settings.data_dir),
        "mode": "thorough" if thorough else "quick",
        "files_indexed": result.files_indexed,
        "chunks_indexed": result.chunks_indexed,
        "chunks_evicted": result.chunks_evicted,
        "vector_index_built": result.vector_index,
        "embedding_available": embedder.is_available(),
    }
    if getattr(args, "as_json", False):
        print(json.dumps(payload, indent=2))
    else:
        print("Indexing complete.")
        print(f"  root:               {payload['root_dir']}")
        print(f"  data:               {payload['data_dir']}")
        print(f"  mode:               {payload['mode']}")
        print(f"  files indexed:      {payload['files_indexed']}")
        print(f"  chunks indexed:     {payload['chunks_indexed']}")
        print(f"  chunks evicted:     {payload['chunks_evicted']}")
        if not payload["embedding_available"]:
            print(
                "  (vector index not built: embedding model unavailable; rerun with network access)"
            )
    return 0


def _run_index(args: argparse.Namespace) -> int:
    return _run_index_async(args)


# ---------------------------------------------------------------------------
# Legacy argparse (unchanged)
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="filesystem-rag-mcp",
        description="Local filesystem RAG MCP server (vector + full-text search, OAuth 2.1)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run environment, dependency, and health diagnostics and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostics in structured JSON format (used with --doctor)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http", "sse"],
        default=os.environ.get("FSRAG_TRANSPORT", "stdio"),
        help="Transport to run: 'stdio' or 'http' (streamable-http)",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(os.environ.get("FSRAG_ROOT_DIR", ".")),
        help="Root directory to index and search (default: current directory)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("FSRAG_DATA_DIR", ".fsrag")),
        help="Directory to store indexes, embeddings, and oauth db (default: .fsrag)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("FSRAG_HOST", "127.0.0.1"),
        help="Host to bind for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FSRAG_PORT", "8000")),
        help="Port to bind for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("FSRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help="SentenceTransformer model name or HF path",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        default=os.environ.get("FSRAG_AUTH_REQUIRED", "true").lower() in {"0", "false", "no"},
        help="Disable OAuth 2.1 requirement on HTTP transport",
    )
    parser.add_argument(
        "--no-dcr",
        action="store_true",
        default=os.environ.get("FSRAG_ALLOW_DCR", "true").lower() in {"0", "false", "no"},
        help="Disable OAuth 2.1 Dynamic Client Registration",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.environ.get("FSRAG_LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=os.environ.get("FSRAG_OFFLINE", "false").lower() in {"1", "true", "yes"},
        help="Run in offline mode (do not download embedding model from HuggingFace)",
    )
    parser.add_argument(
        "--oauth-issuer",
        default=os.environ.get("FSRAG_OAUTH_ISSUER"),
        help="OAuth 2.1 issuer URL (defaults to http://<host>:<port>)",
    )
    parser.add_argument(
        "--create-client",
        metavar="NAME",
        help="Pre-generate an OAuth client (client_id + client_secret), persist to database, print credentials, and exit",
    )
    parser.add_argument(
        "--list-clients",
        action="store_true",
        help="List all pre-generated / registered OAuth clients and exit",
    )
    parser.add_argument(
        "--config-snippet",
        choices=["claude", "zed", "hermes", "all"],
        help="Print ready-to-paste MCP client configuration JSON and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    argv_list = sys.argv[1:] if argv is None else argv

    # ---- Subcommand dispatch (new) ------------------------------------
    sub = _argv_uses_subcommand(argv_list)
    if sub is not None:
        # Subcommands emit JSON to stdout for `--json` consumers, so we route
        # all structlog status lines to stderr at WARNING-or-above. The legacy
        # server-launcher path (below) calls configure_logging(args.log_level)
        # with whatever the user requested.
        configure_logging("WARNING")
        sub_parser = _build_subcommand_parser("filesystem-rag-mcp")
        args = sub_parser.parse_args(argv_list)
        if args.command == "doctor":
            sys.exit(_run_doctor(args))
        if args.command == "search":
            sys.exit(_run_search(args))
        if args.command == "stats":
            sys.exit(_run_stats(args))
        if args.command == "index":
            sys.exit(_run_index(args))
        sys.exit(f"Unknown subcommand: {args.command}")

    # ---- Legacy flag-based dispatch -----------------------------------
    args = parse_args(argv)

    if args.doctor:
        from .doctor import print_doctor_report

        settings = Settings(
            root_dir=args.root_dir.resolve(),
            data_dir=args.data_dir.resolve(),
            embedding_model=args.embedding_model,
        )
        sys.exit(print_doctor_report(settings, as_json=args.json))

    if args.create_client:
        from .oauth import MCPFileRAGAuthProvider

        settings = Settings(
            root_dir=args.root_dir.resolve(),
            data_dir=args.data_dir.resolve(),
        )
        provider = MCPFileRAGAuthProvider(settings)
        client = provider.create_pregenerated_client(client_name=args.create_client)
        print(
            json.dumps(
                {
                    "client_id": client.client_id,
                    "client_secret": client.client_secret,
                    "client_name": client.client_name,
                    "scope": client.scope,
                    "redirect_uris": [str(u) for u in (client.redirect_uris or [])],
                },
                indent=2,
            )
        )
        return

    if args.list_clients:
        from .oauth import MCPFileRAGAuthProvider

        settings = Settings(
            root_dir=args.root_dir.resolve(),
            data_dir=args.data_dir.resolve(),
        )
        provider = MCPFileRAGAuthProvider(settings)
        clients = provider.store.list_clients()
        print(
            json.dumps(
                [
                    {
                        "client_id": c.client_id,
                        "client_name": c.client_name,
                        "scope": c.scope,
                        "redirect_uris": [str(u) for u in (c.redirect_uris or [])],
                    }
                    for c in clients
                ],
                indent=2,
            )
        )
        return

    if args.config_snippet:
        _print_config_snippet(args.config_snippet, args)
        return

    configure_logging(args.log_level)

    transport_choice = args.transport
    if transport_choice == "http":
        transport_choice = "streamable-http"

    auth_required = not args.no_auth and transport_choice != "stdio"
    allow_dcr = not args.no_dcr

    issuer_url = args.oauth_issuer or f"http://{args.host}:{args.port}"

    settings = Settings(
        root_dir=args.root_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        embedding_model=args.embedding_model,
        auth_required=auth_required,
        oauth_allow_dynamic_registration=allow_dcr,
        oauth_issuer=issuer_url,
        offline_mode=args.offline,
    )

    log.info(
        "starting_server",
        transport=transport_choice,
        root_dir=str(settings.root_dir),
        data_dir=str(settings.data_dir),
        auth_required=settings.auth_required,
    )

    server = build_server(settings)

    if transport_choice == "stdio":
        asyncio.run(server.run_stdio_async())
    elif transport_choice == "streamable-http":
        asyncio.run(
            server.run_streamable_http_async(
                host=args.host,
                port=args.port,
            )
        )
    elif transport_choice == "sse":
        asyncio.run(
            server.run_sse_async(
                host=args.host,
                port=args.port,
            )
        )
    else:
        sys.exit(f"Unknown transport: {args.transport}")


def _print_config_snippet(target: str, args: argparse.Namespace) -> None:
    import json
    import sys

    root = str(args.root_dir.resolve())
    exec_path = sys.executable

    claude_cfg = {
        "mcpServers": {
            "filesystem-rag": {
                "command": exec_path,
                "args": [
                    "-m",
                    "filesystem_rag_mcp.cli",
                    "--transport",
                    "stdio",
                    "--root-dir",
                    root,
                ],
            }
        }
    }

    zed_cfg = {
        "context_servers": [
            {
                "id": "filesystem-rag",
                "executable": exec_path,
                "args": [
                    "-m",
                    "filesystem_rag_mcp.cli",
                    "--transport",
                    "stdio",
                    "--root-dir",
                    root,
                ],
            }
        ]
    }

    hermes_cfg = {
        "mcp_servers": {
            "filesystem-rag": {
                "command": exec_path,
                "args": [
                    "-m",
                    "filesystem_rag_mcp.cli",
                    "--transport",
                    "stdio",
                    "--root-dir",
                    root,
                ],
            }
        }
    }

    if target in ("claude", "all"):
        print("\n# --- Claude Desktop (claude_desktop_config.json) ---")
        print(json.dumps(claude_cfg, indent=2))
    if target in ("zed", "all"):
        print("\n# --- Zed Editor (settings.json context_servers) ---")
        print(json.dumps(zed_cfg, indent=2))
    if target in ("hermes", "all"):
        print("\n# --- Hermes / Antigravity Agent Configuration ---")
        print(json.dumps(hermes_cfg, indent=2))


if __name__ == "__main__":
    main()
