# Master Phased Implementation Plan: Local Filesystem RAG MCP Server

## Architecture Roadmap & Phased Execution

### 1. Architectural Scope & Mission
The Local Filesystem RAG MCP Server provides low-latency, offline semantic and lexical intelligence over personal notes and software codebases via Model Context Protocol (MCP).

### 2. Consolidated 5-Phase Roadmap

#### Phase 1: Workload Profiles & Domain Specialization
- Implement `PersonalNotesProfile` and `GeneralCodebaseProfile` polymorphic abstractions.
- Configure default path resolution (`~/.local/share/notes` for notes profile).
- Filter editor temporary files (`.*.swp`, `.*.swo`, `.*.swx`, `.*.un~`, `*~`, `*.tmp`, `.goutputstream-*`).
- Preserve Markdown AST heading breadcrumbs (H1-H6) in chunk metadata.
- Extend diagnostics in `doctor.py` to validate active profile, directory permissions, and exclusion rules.

#### Phase 2: Hardware-Conscious Inference & Adaptive Thread Pinning
- Configure execution provider options: intra_op_num_threads = 6, inter_op_num_threads = 1 on host Ryzen 5 5600XT architecture, with dynamic physical core fallback.
- Perform startup eager initialization and synthetic warm-up pass.
- Calibrate 3-stage retrieval funnel: Stage 1 (top 50 BM25 + top 50 Vector via RRF k=60) -> Stage 2 (top 20 neural rerank cut with sigmoid score) -> Stage 3 (top 5 projection).
- Integrate query caching and bounded batch sizes during ingestion.

#### Phase 3: Service Daemonization & Systemd Lifecycle Management
- Provide systemd user service unit generator for `~/.config/systemd/user/mcp-rag-watcher.service`.
- Implement CLI subcommands: `service install`, `start`, `stop`, `restart`, `status`.
- Route telemetry to journald / stderr.
- Ensure non-blocking concurrency and graceful signal trapping (SIGINT, SIGTERM).

#### Phase 4: Retrieval Quality Benchmarking & Regression Gates
- Curate 25-query ground-truth benchmark in `tests/data/evaluation_dataset.json`.
- Provide automated evaluation runner across 4 modes (Lexical, Vector, Hybrid RRF, Hybrid + Reranker) tracking MRR@5 and Hit@1.
- CLI subcommand: `benchmark` and CI workflow integration.
- Enforce component and round-trip latency SLAs (<75ms average).

#### Phase 5: Automated Storage Maintenance & Operational Runbooks
- Implement storage compaction, incremental vacuuming, and index segment optimization.
- Expose administrative MCP tool: `optimize_database`.
- Create scheduled systemd maintenance timer `mcp-rag-maintenance.timer`.
- Provide complete disaster recovery runbooks.
