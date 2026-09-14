# Local Filesystem RAG MCP Server — Session Context & Handoff Brief

## 1. Executive Summary & Scope
The **Local Filesystem RAG MCP Server** (`filesystem-rag-mcp`) provides offline-first, sub-75ms hybrid neural and lexical search over local codebases and personal Markdown vaults via the Model Context Protocol (MCP 2026-07-28 / 2026-07-29). It supports both `stdio` and Streamable `http` transports with OAuth 2.1 RFC 7591 Dynamic Client Registration.

All 5 development phases from the architectural implementation plan have been converted to Markdown, integrated into `docs/`, fully implemented in the Python codebase, validated with unit/integration test suites, and pushed to `origin/main`.

---

## 2. Five-Phase Implementation Status

| Phase | Title | Core Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Workload Profiles & Domain Specialization** | Polymorphic `notes` vs `codebase` profiles, automatic root path resolution (`~/.local/share/notes`), editor artifact filtering (`.*.swp`, `.*.swo`, `.*.un~`, `*~`, `*.tmp`, `.goutputstream-*`), `search_notes` tool, profile pre-flight diagnostics in `doctor.py`. | **Complete** |
| **Phase 2** | **Hardware-Conscious Inference & Thread Pinning** | Markdown AST heading lineage extraction (`H1 > H2 > H3`) saved in chunk metadata, Whoosh schema, and Vector metadata; CPU thread limits (`intra_op=6`, `inter_op=1` on AMD Ryzen 5 5600XT architecture); eager embedding warmup; batched vector upserts. | **Complete** |
| **Phase 3** | **Service Daemonization & Systemd Management** | Asynchronous debounced `watcher.py` daemon with signal handling (`SIGINT`, `SIGTERM`); `filesystem-rag-mcp service [install\|start\|stop\|restart\|status]` CLI subcommands generating user units. | **Complete** |
| **Phase 4** | **Retrieval Quality Benchmarking & Quality Gates** | 25-query ground-truth benchmark harness (`benchmark.py` + `tests/data/evaluation_dataset.json`) evaluating 4 modes (Lexical, Vector, Hybrid RRF, Hybrid + Cross-Encoder); `benchmark` CLI subcommand; CI integration. | **Complete** |
| **Phase 5** | **Automated Storage Maintenance & Runbooks** | Database vacuuming (`incremental_vacuum`), WAL checkpointing (`TRUNCATE`), Whoosh index segment compaction, integrity auditing (`optimize_storage`); `optimize_database` MCP tool; weekly systemd maintenance timer. | **Complete** |

---

## 3. Git Status & Commit History

- **Repository Root**: `/home/freya/Projects/filesystem-rag-mcp`
- **Branch**: `main` (synchronized with `origin/main`)
- **Working Tree**: Clean (`nothing to commit, working tree clean`)

### Logged & Pushed Commits
1. `7ce0fba` — `docs: integrate architectural implementation plans for phases 1-5 and developer guidelines`
2. `ea8f8db` — `feat(config): add workload profiles and editor artifact exclusion`
3. `f5297fa` — `feat(search): add hierarchical heading breadcrumbs and hardware-conscious inference`
4. `3f279ce` — `feat(service): enhance background watcher daemon and service lifecycle CLI`
5. `fdb19e5` — `feat(maintenance): add storage optimization and compaction engine`
6. `0c239ae` — `feat(benchmark): add retrieval quality evaluation harness and regression gates`
7. `f6ba82e` — `feat(server): expose profile-aware tools, diagnostics, and update CI workflows`

---

## 4. Architectural & Codebase Map

```
filesystem-rag-mcp/
├── .github/workflows/
│   └── ci.yml                     # Multi-version CI matrix with benchmark & doctor smoke tests
├── docs/
│   ├── ARCHITECTURE.md            # High-level architecture and subsystem design
│   ├── CONFIGURATION.md           # Configuration options and environment variables
│   ├── DEVELOPER_GUIDELINES.md    # Agent and developer engineering conventions
│   ├── HIGH_LEVEL_PROJECT_OVERVIEW.md # Mission, constraints, and operational goals
│   ├── INSTRUCTIONS.md            # Setup, CLI usage, and operational runbooks
│   ├── MASTER_IMPLEMENTATION_PLAN.md  # 5-phase consolidated roadmap
│   ├── PHASE_1_WORKLOAD_PROFILES.md
│   ├── PHASE_2_HARDWARE_INFERENCE.md
│   ├── PHASE_3_SERVICE_DAEMONIZATION.md
│   ├── PHASE_4_RETRIEVAL_BENCHMARKING.md
│   ├── PHASE_5_STORAGE_MAINTENANCE.md
│   └── QUICKSTART.md
├── src/filesystem_rag_mcp/
│   ├── benchmark.py               # 4-mode MRR@5 & Hit@1 automated evaluation harness
│   ├── cli.py                     # Unified CLI: doctor, search, stats, index, service, benchmark, optimize
│   ├── config.py                  # Settings, WorkloadProfile enum, ProfileConfig presets
│   ├── doctor.py                  # Environment, dependency, profile, and WAL diagnostics
│   ├── fulltext.py                # Whoosh schema with breadcrumbs, BM25 scoring, index compaction
│   ├── indexing.py                # Profile-aware file discovery, hash caching, chunk orchestration
│   ├── maintenance.py             # SQLite vacuum, WAL truncate, Whoosh segment merge
│   ├── search.py                  # RRF fusion engine (k=60), FlashRank cross-encoder reranker
│   ├── security.py                # Path sandboxing, traversal guards, is_editor_artifact()
│   ├── semantic_chunker.py        # AST code preservation & Markdown H1-H6 breadcrumb tracking
│   ├── server.py                  # FastMCP server, OAuth 2.1, 20+ search and admin MCP tools
│   ├── vector.py                  # ChromaDB store, CPU thread tuning, eager model warmup
│   └── watcher.py                 # Async debounced file watcher daemon with signal traps
└── tests/
    ├── data/
    │   └── evaluation_dataset.json # 25-query ground-truth evaluation benchmark
    ├── test_benchmark_runner.py
    ├── test_editor_artifact_filtering.py
    ├── test_heading_breadcrumbs.py
    ├── test_service_cli.py
    ├── test_storage_maintenance.py
    ├── test_workload_profiles.py
    └── ... (26 test files total, 83 test cases)
```

---

## 5. Tool & CLI Capabilities

### MCP Tools Exposed
- **Retrieval**: `search`, `search_notes` (notes-profile optimized), `grep_search`, `deep_search`, `pack_context`, `get_chunk_context`, `get_chunk`.
- **Navigation & Inspection**: `list_directory`, `read_file`, `read_files_batch`, `read_file_markdown`, `download_file_raw`, `fetch_targeted_data`.
- **Code Intelligence**: `search_symbols`, `find_symbol_references`, `get_corpus_graph`, `git_search`.
- **Mutation & Indexing**: `patch_file`, `refresh_file`, `index_file`, `reindex_directory`, `refresh_index`.
- **Administration & Diagnostics**: `ping`, `get_index_status`, `get_index_stats`, `optimize_database`, `add_workspace`, `list_workspaces`.

### CLI Subcommands
```bash
filesystem-rag-mcp doctor [--json] [--profile notes|codebase]
filesystem-rag-mcp search "<query>" [--mode hybrid|fulltext|semantic] [--profile notes|codebase]
filesystem-rag-mcp stats [--data-dir PATH] [--json]
filesystem-rag-mcp index [--root-dir PATH] [--data-dir PATH] [--profile notes|codebase]
filesystem-rag-mcp service [install|start|stop|restart|status] [--profile notes|codebase]
filesystem-rag-mcp benchmark [--dataset PATH] [--json]
filesystem-rag-mcp optimize [--data-dir PATH] [--json]
```

---

## 6. Verification Evidence

- **Pytest Suite**: 83/83 tests passing in ~87s (`uv run pytest -q`).
- **Static Typing**: 0 errors across all 28 source files (`uv run mypy src`).
- **Linter & Formatting**: 0 issues, fully formatted with Ruff (`uv run ruff check .` / `uv run ruff format --check .`).
- **Smoke Tests**: Validated CLI commands for `doctor`, `index`, `stats`, `search`, `optimize`, `benchmark`, and `service`.

---

## 7. Configuration & Environment Variables

The server resolves configuration using three prefixes with descending precedence: `FSRAG_*` > `FILESYSTEM_RAG_*` > `MCP_RAG_*`.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `FSRAG_PROFILE` | `codebase` | Workload profile preset: `codebase` or `notes`. |
| `FSRAG_ROOT_DIR` | Current Dir (or `~/.local/share/notes` for notes profile) | Root directory monitored and indexed. |
| `FSRAG_DATA_DIR` | `~/.local/share/filesystem-rag-mcp` | Storage directory for ChromaDB, Whoosh, and metadata. |
| `FSRAG_OFFLINE_MODE` | `0` | Disable outbound HuggingFace requests; use cached models only. |
| `FSRAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Dense vector embedding model. |
| `FSRAG_RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | FlashRank neural cross-encoder reranker model. |
| `FSRAG_CHUNK_SIZE` | `800` (codebase) / `600` (notes) | Target character length per chunk. |
| `FSRAG_CHUNK_OVERLAP`| `150` (codebase) / `100` (notes) | Character overlap between consecutive chunks. |

---

## 8. Invariants & Handoff Guidance for the Next Agent

1. **Backwards Compatibility**: All legacy MCP endpoints (`search`, `read_file`, `ping`, etc.) must remain signature-compatible. Additive parameters (`breadcrumbs`, `profile`) must retain functional defaults.
2. **Editor Artifact Invariant**: Never allow temporary files (`*.swp`, `*.swo`, `*.un~`, `*~`, `.goutputstream-*`, `*.tmp`) into Whoosh or ChromaDB indexes. Both `indexing.py` and `watcher.py` enforce `is_editor_artifact()`.
3. **Execution Safety**: The terminal tool caps foreground execution at 600 seconds. For tests or long runs, keep timeout $\le 600$.
4. **Offline Resilience**: Embedder and Reranker gracefully degrade if weights are unavailable or if running in air-gapped environments without failing server initialization.
