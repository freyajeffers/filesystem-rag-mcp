"""Configuration for the filesystem-rag-mcp server.

Settings are loaded from environment variables (with sensible defaults so the
server runs out of the box for local development). Production deployments
should set explicit values for `FS_RAG_INDEX_DIR`, `FS_RAG_DATA_DIR`,
`FS_RAG_OAUTH_SECRET`, and `FS_RAG_TRANSPORT`.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Transport(str, Enum):
    """Transport the server listens on."""

    STDIO = "stdio"
    HTTP = "http"


class Settings(BaseModel):
    """Server settings; populated from environment variables.

    The mapping is one-to-one with `FS_RAG_*` variables. Booleans parse
    naturally from `"true"`/`"false"`; paths are expanded.
    """

    # ---- transport ----------------------------------------------------------
    transport: Transport = Field(
        default=Transport.STDIO,
        description="Transport to listen on: 'stdio' or 'http'.",
    )
    http_host: str = Field(
        default="127.0.0.1",
        description="Bind address for HTTP transport. Use 127.0.0.1 unless on a "
        "trusted network — the spec requires validating Origin headers and "
        "binding to localhost when running locally.",
    )
    http_port: int = Field(default=8765, ge=1, le=65535)
    http_path: str = Field(
        default="/mcp",
        description="Path component of the MCP endpoint exposed over HTTP.",
    )

    # ---- filesystem root ----------------------------------------------------
    root_dir: Path = Field(
        default=Path(os.path.expanduser("~/Documents")),
        description="Root directory the server is permitted to read and index.",
    )

    # ---- storage ------------------------------------------------------------
    data_dir: Path = Field(
        default=Path(".filesystem-rag-mcp"),
        description="Where the vector store, full-text index, and OAuth client "
        "registry live. Created on startup if missing.",
    )

    # ---- indexing -----------------------------------------------------------
    chunk_size: int = Field(default=800, ge=64, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    max_file_bytes: int = Field(default=2_000_000, ge=1_024)
    """Files larger than this are skipped (binary files are auto-skipped
    regardless of size)."""
    follow_symlinks: bool = Field(default=False)
    ignore_globs: tuple[str, ...] = Field(
        default=(
            "**/.git/**",
            "**/__pycache__/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/.filesystem-rag-mcp/**",
            "**/.DS_Store",
        )
    )

    # ---- embeddings ---------------------------------------------------------
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model id; the default is small, "
        "fast, and gives 384-dim vectors.",
    )
    embedding_dim: int = Field(default=384, ge=1)

    # ---- search -------------------------------------------------------------
    default_top_k: int = Field(default=10, ge=1, le=100)
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight given to vector score vs BM25 score when combining "
        "the two for hybrid search. 0.0 = full-text only, 1.0 = vectors only.",
    )

    # ---- OAuth / auth -------------------------------------------------------
    auth_required: bool = Field(
        default=False,
        description="When True, the HTTP transport requires a Bearer token. "
        "Ignored on stdio (stdio always uses env-supplied credentials).",
    )
    oauth_issuer: str = Field(
        default="http://127.0.0.1:8765",
        description="Canonical issuer URL used in AS metadata and JWT iss claim.",
    )
    oauth_secret: str = Field(
        default="dev-only-secret-please-override-with-at-least-32-bytes-key!",
        min_length=32,
        description="HMAC secret for signing access tokens (minimum 32 bytes for HS256).",
    )
    oauth_access_token_ttl_seconds: int = Field(default=3600, ge=60, le=86_400)
    oauth_refresh_token_ttl_seconds: int = Field(default=86_400 * 30, ge=3_600)
    oauth_require_pkce: bool = Field(
        default=True,
        description="If True, all authorization requests require PKCE (S256).",
    )
    oauth_allow_dynamic_registration: bool = Field(default=True)

    # ---- operational & performance guards -----------------------------------
    offline_mode: bool = Field(
        default=False,
        description="When True, disable outbound HuggingFace network requests and "
        "gracefully degrade to full-text search if local embedding weights are not cached.",
    )
    max_convert_file_bytes: int = Field(
        default=50_000_000,
        ge=10_000,
        description="Files larger than this limit are truncated or converted via head-slice to prevent OOM.",
    )
    index_binary_files: bool = Field(
        default=False,
        description="If False, raw binary files that fall back to hexdumps are excluded from "
        "indexing. When True, they are converted via universal hexdump/string extraction.",
    )
    index_binary_vectors: bool = Field(
        default=False,
        description="If False, raw binary files that fall back to hexdumps are indexed in "
        "fulltext search (if index_binary_files=True) but skipped during dense vector embedding.",
    )

    # ---- observability ------------------------------------------------------
    log_level: str = Field(default="INFO")

    @field_validator("root_dir", "data_dir", mode="before")
    @classmethod
    def _expand_path(cls, v: object) -> Path:
        return Path(os.path.expanduser(str(v))).resolve()

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        """Build a Settings instance from `FS_RAG_*` environment variables.

        Only variables that are actually set are honored; everything else
        keeps its default. This lets the server run with zero config in dev.
        """
        env = env if env is not None else dict(os.environ)
        prefix = "FS_RAG_"
        mapping: dict[str, object] = {}
        for key, value in env.items():
            if not key.startswith(prefix):
                continue
            short = key[len(prefix) :]
            mapping[_env_key_to_field(short)] = _coerce(short, value)
        return cls(**mapping)


def _env_key_to_field(short: str) -> str:
    """Map `FS_RAG_HTTP_HOST` (after prefix strip -> `HTTP_HOST`) -> `http_host`."""
    return short.lower()


def _coerce(short: str, raw: str) -> object:
    """Coerce an env-string into the right Python type for the field."""
    if short == "TRANSPORT":
        return raw.strip().lower()
    if short in {
        "HTTP_HOST", "HTTP_PATH", "ROOT_DIR", "DATA_DIR",
        "EMBEDDING_MODEL", "OAUTH_ISSUER", "OAUTH_SECRET", "LOG_LEVEL",
    }:
        return raw
    if short in {
        "HTTP_PORT", "CHUNK_SIZE", "CHUNK_OVERLAP", "MAX_FILE_BYTES",
        "EMBEDDING_DIM", "DEFAULT_TOP_K",
        "OAUTH_ACCESS_TOKEN_TTL_SECONDS", "OAUTH_REFRESH_TOKEN_TTL_SECONDS",
        "MAX_CONVERT_FILE_BYTES",
    }:
        return int(raw)
    if short == "HYBRID_ALPHA":
        return float(raw)
    if short in {
        "FOLLOW_SYMLINKS", "OAUTH_REQUIRE_PKCE",
        "OAUTH_ALLOW_DYNAMIC_REGISTRATION", "AUTH_REQUIRED",
        "OFFLINE_MODE", "INDEX_BINARY_FILES", "INDEX_BINARY_VECTORS",
    }:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if short == "IGNORE_GLOBS":
        # Comma-separated list of glob patterns
        return tuple(p.strip() for p in raw.split(",") if p.strip())
    return raw


settings: Settings = Settings.from_env()
