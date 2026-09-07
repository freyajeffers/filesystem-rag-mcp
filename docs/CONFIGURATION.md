# Configuration Reference

All settings can be specified via command-line arguments or environment variables (`FS_RAG_*` or `FSRAG_*`).

---

## Environment Variables & Options

| Option | Environment Variable | Default | Description |
|---|---|---|---|
| `--root-dir` | `FS_RAG_ROOT_DIR` / `FSRAG_ROOT_DIR` | `.` | Root directory path to index, monitor, and search |
| `--data-dir` | `FS_RAG_DATA_DIR` / `FSRAG_DATA_DIR` | `.fsrag` | Storage location for Whoosh indexes, Chroma vector DB, and OAuth SQLite state |
| `--transport` | `FS_RAG_TRANSPORT` / `FSRAG_TRANSPORT` | `stdio` | Transport protocol: `stdio`, `http` (Streamable HTTP), or `sse` |
| `--host` | `FS_RAG_HOST` / `FSRAG_HOST` | `127.0.0.1` | Network interface to bind for HTTP/SSE transport |
| `--port` | `FS_RAG_PORT` / `FSRAG_PORT` | `8000` | TCP port for HTTP/SSE transport |
| `--log-level` | `FS_RAG_LOG_LEVEL` / `FSRAG_LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--embedding-model` | `FS_RAG_EMBEDDING_MODEL` / `FSRAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name or HuggingFace path |
| `--offline` | `FS_RAG_OFFLINE_MODE` / `FSRAG_OFFLINE_MODE` | `false` | When true, disables outbound network calls to HuggingFace |
| `--no-auth` | `FS_RAG_AUTH_REQUIRED` / `FSRAG_AUTH_REQUIRED` | `false` | When set, disables OAuth 2.1 authentication on HTTP transport |
| `--no-dcr` | `FS_RAG_ALLOW_DCR` / `FSRAG_ALLOW_DCR` | `false` | When set, disables Dynamic Client Registration (RFC 7591) |

---

## Performance & Tuning Variables

| Environment Variable | Default | Description |
|---|---|---|
| `FS_RAG_CHUNK_SIZE` | `500` | Target chunk token/word length for semantic chunking |
| `FS_RAG_CHUNK_OVERLAP` | `50` | Overlap between sequential chunks |
| `FS_RAG_MAX_FILE_SIZE_MB` | `50` | Maximum file size in MB to index into vector store |
| `FS_RAG_MAX_TEXT_SIZE_CHARS` | `2000000` | Maximum characters extracted per document |
| `FS_RAG_WATCH_DEBOUNCE_MS` | `500` | Debounce window in milliseconds for file change watcher |
