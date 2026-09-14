# Developer & Coding Agent Guidelines

## Local Filesystem RAG MCP Server

### 1. Purpose & Agent Persona

This document establishes operational directives, architectural invariants, and development protocols for any autonomous coding agent, AI assistant, or software engineer contributing to the filesystem_rag_mcp codebase.

Prioritize correctness over cleverness, maintain non-blocking concurrency, respect host hardware constraints, write comprehensive tests before landing code, and leave no architectural debt.

### 2. Core Architectural Invariants (Non-Negotiable Constraints)

1. **Zero Feature Degradation**: The existing codebase contains mature capabilities for multi-format conversion, code symbol extraction, git operations, and grep tools. Enhancements for the personal notes profile MUST NOT break, remove, or degrade any existing general-purpose codebase capabilities.

2. **Strict Thread Pool Discipline**: Never instantiate unmanaged ONNX Runtime sessions, PyTorch thread pools, or thread pools that default to logical thread counts. Intra-op threads must be bounded by physical core counts (max 6 on the host 5600XT architecture) and inter-op threads must remain 1.

3. **Non-Blocking Storage & Concurrency**:
   - Write operations must utilize immediate transaction semantics to prevent deadlocks.
   - Long-running CPU computation or file I/O must never occur inside open database transactions.
   - Store configurations must support non-blocking concurrent reads during active indexing.

4. **Editor Artifact Exclusion**: Any file matching editor temporary patterns (`.*.swp`, `.*.swo`, `.*.swx`, `.*.un~`, `*~`, `*.tmp`, `.*.tmp`, `.goutputstream-*`) must be rejected at the earliest possible filter boundary before entering ingestion queues.

5. **Strict Path Traversal & Sandboxing Security**: All path operations must resolve symbolic links, normalize paths, and verify that target paths remain strictly within authorized profile roots. Direct un-sanitized path operations are prohibited.

6. **Stdio Purity for MCP Transport**: The FastMCP server communicates with client applications via standard input and standard output (stdio) using JSON-RPC frames. Never emit debug print statements, unformatted logs, or progress bars to stdout. All logging must route strictly to stderr or systemd journald.

### 3. Test-Driven Development (TDD) Protocol

Every modification, bug fix, or feature addition must adhere to a strict verification protocol:
- **Define Unit & Integration Tests First**: Before implementing changes, write test cases in the `tests/` directory that capture expected behavior and edge cases.
- **Mock Expensive Subsystems**: In unit tests, mock ONNX Runtime inference sessions and heavy model weights to ensure test execution remains instantaneous. Use integration tests with synthetic sample files to verify end-to-end pipelines.
- **Validate Non-Blocking Concurrency**: Concurrency features must be tested with simulated simultaneous read and write operations.
- **Run Pre-Flight Regression Gates**: Execute the full test suite and benchmark runner before committing changes.

### 4. Inter-Module Contracts & Interface Rules

- **Strict Typing**: All public module functions, method parameters, and return types must be fully typed using Python type hints, conforming to `py.typed` standards.
- **Data Class & Pydantic Boundaries**: Use explicit typed data structures (such as Pydantic models or standard dataclasses) for all data moving across module boundaries.
- **Decoupled Dependencies**: Storage, chunker, retriever, and server components remain cleanly separated.

### 5. Performance SLAs & Latency Budgets

All code changes must operate within the following component latency budgets:
- **FTS5 / Lexical Search**: < 5 ms (for 50 candidate chunks).
- **Dense Vector Cosine Search**: < 15 ms (for 50 candidate chunks).
- **Cross-Encoder Reranker**: < 45 ms (for 20 candidate pairs on CPU).
- **RRF Fusion & Response Formatting**: < 10 ms.
- **Total End-to-End Query Round-Trip**: < 75 ms (with 95th percentile < 100 ms).
