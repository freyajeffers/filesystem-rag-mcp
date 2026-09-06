# filesystem-rag-mcp

Local filesystem RAG MCP server combining vector/semantic search and full-text (BM25) search over files. Compliant with MCP spec 2026-07-28 / 2026-07-29, supporting both `stdio` and `http` (Streamable HTTP) transports, along with OAuth 2.1 authentication and Dynamic Client Registration (RFC 7591).

## Features

- **MCP Protocol Conformance**: Built against MCP specification 2026-07-28 (`mcp[cli]>=1.21.0`), supporting tool calling, resources (`fs://stats`, `fs://config`), and prompts (`rag_query`).
- **Dual Transports**:
  - `stdio`: Standard input/output transport for local desktop assistants and CLI hosts (Claude Desktop, Hermes, etc.).
  - `http`: Modern Streamable HTTP transport for remote and web deployments.
- **Hybrid Search Architecture**:
  - **Full-Text**: Fast embedded BM25 search via Whoosh with field boosting and prefix matching.
  - **Vector / Semantic**: Embedded ChromaDB with sentence-transformers embedding generation.
  - **Reciprocal Rank Fusion (RRF)**: Merges sparse full-text and dense semantic scores without manual hyperparameter tuning.
- **OAuth 2.1 & Dynamic Client Registration (RFC 7591)**:
  - Supports RFC 7591 DCR (`/register`) to dynamically onboard MCP clients.
  - PKCE S256 code challenge verification.
  - RFC 8414 Authorization Server Metadata (`/.well-known/oauth-authorization-server`).
  - Protected Resource Metadata (`/.well-known/oauth-protected-resource`).
  - Strict Bearer token verification using cryptographic JWTs.
- **Path Security**:
  - Directory traversal prevention (`..`, path symlink escapes).
  - Configurable inclusion/exclusion glob patterns.

## Installation

Using `uv`:
```bash
git clone https://github.com/freyajeffers/filesystem-rag-mcp.git
cd filesystem-rag-mcp
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Running the Server

### Stdio Transport (Default)
Ideal for desktop clients like Claude Desktop:
```bash
filesystem-rag-mcp --transport stdio --root-dir /path/to/my/documents
```

### Streamable HTTP Transport with OAuth 2.1
```bash
filesystem-rag-mcp --transport http --host 127.0.0.1 --port 8000 --root-dir /path/to/my/documents
```

### Streamable HTTP Transport without Auth (Dev Mode)
```bash
filesystem-rag-mcp --transport http --no-auth --host 127.0.0.1 --port 8000 --root-dir /path/to/my/documents
```

## Available MCP Tools

- `search`:
  - Query parameters:
    - `query` (string, required): Natural language search query or keywords.
    - `mode` (string, default: "hybrid"): `hybrid`, `fulltext`, or `semantic`.
    - `top_k` (integer, default: 10): Maximum number of search hits.
    - `path_filter` (string, optional): Glob pattern or subfolder constraint (e.g. `src/**/*.py`).
- `refresh_index`:
  - Force re-indexing of documents and chunk caching.

## Testing

Run tests with `pytest`:
```bash
pytest
```

## License

MIT
