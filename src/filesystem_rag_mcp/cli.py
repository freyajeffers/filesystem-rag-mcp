"""Command-line entrypoint for filesystem-rag-mcp.

Usage:
  filesystem-rag-mcp --transport stdio --root-dir /path/to/docs
  filesystem-rag-mcp --transport http --host 127.0.0.1 --port 8000 --root-dir /path/to/docs
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .config import Settings
from .logging_setup import configure_logging, get_logger
from .server import build_server

log = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="filesystem-rag-mcp",
        description="Local filesystem RAG MCP server (vector + full-text search, OAuth 2.1)",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    transport_choice = args.transport
    if transport_choice == "http":
        transport_choice = "streamable-http"

    auth_required = not args.no_auth and transport_choice != "stdio"
    allow_dcr = not args.no_dcr

    settings = Settings(
        root_dir=args.root_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        embedding_model=args.embedding_model,
        auth_required=auth_required,
        oauth_allow_dynamic_registration=allow_dcr,
        oauth_issuer=f"http://{args.host}:{args.port}",
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


if __name__ == "__main__":
    main()
