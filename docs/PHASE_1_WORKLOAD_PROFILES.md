# Phase 1: Workload Profiles & Domain Specialization

## 1. Architectural Scope
Phase 1 introduces a unified profile abstraction allowing filesystem_rag_mcp to tailor scanning rules, file format filters, editor artifact exclusions, and chunking heuristics based on target domain.

## 2. Workload Profiles
- **Personal Notes Profile (`notes`)**:
  - Target path defaults to `~/.local/share/notes`.
  - Whitelist: Markdown files (`*.md`, `*.markdown`).
  - AST Breadcrumbs: Enforces heading breadcrumb retention (H1-H6) in chunk metadata.
  - Disables code symbol extraction and complex binary conversion.
- **General Codebase Profile (`codebase`)**:
  - Broad multi-format support (code, config, docs).
  - Enables tree-sitter symbols, git inspection, and grep.

## 3. Editor Artifact & Temporary File Filtering
- Swap files: `.*.swp`, `.*.swo`, `.*.swx`
- Undo files: `.*.un~`
- Backup / Temp files: `*~`, `*.tmp`, `.*.tmp`, `.goutputstream-*`
- Hidden directories: `.git/`, `.obsidian/`, `.trash/`, `.fsrag/`

## 4. Heading Breadcrumbs
Preserve heading hierarchy (e.g., `Guide > Architecture > Storage`) in chunk metadata and fulltext index fields.

## 5. Diagnostics
Extend `doctor.py` to validate active profile root existence, permissions, and exclusion filters.
