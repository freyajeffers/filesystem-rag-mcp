# filesystem-rag-mcp

Local filesystem RAG MCP server combining vector/semantic search and full-text (BM25) search over files. Compliant with MCP spec 2026-07-28 / 2026-07-29, supporting both `stdio` and `http` (Streamable HTTP) transports, along with OAuth 2.1 authentication and Dynamic Client Registration (RFC 7591).

## Features

- **MCP Protocol Conformance**: Built against MCP specification 2026-07-28 (`mcp[cli]>=1.21.0`), supporting tool calling, resources (`fs://stats`, `fs://config`), and prompts (`rag_query`).
- **Universal File to Markdown Conversion**:
  - Converts virtually **any** file type to clean, informative Markdown on demand:
    - **Documents & Office**: PDF, Word (DOCX/DOC), PowerPoint (PPTX/PPT), Excel (XLSX/XLS), RTF, EPUB.
    - **Notebooks & Code**: Jupyter Notebooks (`.ipynb`) with inputs/outputs/markdown, Python, JavaScript, TypeScript, Rust, Go, C/C++, Java, Shell, etc.
    - **Structured Data**: JSON, JSONL, YAML, TOML, XML, CSV, TSV, SQL.
    - **Databases**: SQLite (`.sqlite`, `.db`, `.sqlite3`) with schema extraction and row previews.
    - **Archives**: ZIP, TAR, TGZ manifests and directory listings.
    - **Media & Audio**: MP3, WAV, FLAC, OGG, M4A with metadata tags (ID3, Vorbis) and audio stream properties.
    - **Images**: Dimensions, format, color mode, and EXIF camera metadata.
    - **Emails**: `.eml` and RFC 822 messages with headers, body parts, and attachment lists.
    - **Binary & Firmware**: Formatted hexdump summaries with embedded printable ASCII string extraction.
- **Deep Content-Type Detection**:
  - Integrates **Google Magika AI** and magic byte inspection so files are classified and converted accurately regardless of extension or missing extensions.
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
    - `alpha` (float, default: 0.5): Weighting between full-text (0.0) and vector (1.0).
    - `wait_for_indexing` (boolean, default: false): If `false`, immediately executes searches using whatever index is currently available without blocking caller; if `true`, waits for background thorough indexing to complete.
  - Returns ranked search hits with match scores, source indexes, contextual snippets, and an `index_state` object notifying the caller of background indexing progress.
- `get_index_status`:
  - Query parameters:
    - `wait` (boolean, default: false): If `true`, synchronously waits for background indexing to finish before returning.
    - `timeout_seconds` (float, default: 30.0): Maximum duration to wait.
  - Returns live indexing status, indicating whether quick or thorough indexes are running or ready, plus chunk counts and timestamps.
- `fetch_targeted_data`:
  - Fine-grained, targeted data extraction from structured and tabular files:
    - **SQLite**: Execute read-only SQL queries via `query` parameter (e.g. `SELECT id, name FROM users WHERE active=1`).
    - **JSON / JSONL**: Query paths via `query` parameter (e.g. `users[0].address.city` or `config.database`).
    - **CSV / TSV**: Select specific columns, apply row offsets/limits, and value match filters.
    - **Text / Code**: Extract exact line ranges via `start_line` and `end_line`.
- `read_file_markdown`:
  - Automatically converts diverse document and data formats (PDF, DOCX, PPTX, XLSX, HTML, IPYNB, CSV, RTF, JSON, YAML, TOML, XML, code) to clean Markdown.
  - Returns `{ "rel_path", "abs_path", "markdown", "length_chars" }`.
- `download_file_raw`:
  - Downloads binary or text files as base64 with auto-detected MIME type and size headers.
  - Returns `{ "rel_path", "abs_path", "mime_type", "total_size_bytes", "returned_size_bytes", "truncated", "base64_data" }`.
- `read_file`:
  - Read plain text files with optional byte truncation.
- `get_chunk`:
  - Inspect full chunk contents by ID.
- `refresh_index`:
  - Force re-indexing of documents and chunk caching.

## Testing

Run tests with `pytest`:
```bash
pytest
```

## License

MIT
