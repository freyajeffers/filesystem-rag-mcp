"""filesystem-rag-mcp: Local filesystem RAG MCP server.

A Model Context Protocol server that indexes a local filesystem directory and
exposes vector (semantic) and full-text search over the indexed corpus.

Implements MCP specification 2026-07-28 (the most current stable revision;
see README.md for the date-correction note). Supports the Streamable HTTP
and stdio transports, and OAuth 2.1 with Dynamic Client Registration (RFC 7591)
as a permitted fallback alongside the newer Client ID Metadata Documents
mechanism preferred by 2026-07-28.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Freya Jeffers"
__license__ = "MIT"

# The protocol version this server is built against. The 2026-07-28 release
# is the current stable spec; the server advertises this in the
# `MCP-Protocol-Version` header and in `serverInfo.protocolVersion`.
PROTOCOL_VERSION = "2026-07-28"

# Earlier revisions we still understand (for backward-compatible clients).
SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2026-07-28",
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
)
