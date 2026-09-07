# Architecture Overview

`filesystem-rag-mcp` is an embedded, zero-configuration Model Context Protocol (MCP) server providing high-performance hybrid retrieval across local directories.

---

## 1. System Pipeline

```
[ Local Filesystem ]
        │
        ▼
[ Live Watcher (watchfiles) ] ── (Incremental Debounce)
        │
        ▼
[ Deep Content Detector ] (Google Magika + MIME magic bytes)
        │
        ▼
[ Universal Format Converters ] (PDF, Word, PPT, Excel, RTF, Notebook, SQLite, etc.)
        │
        ├──► [ Semantic Chunker ] (Structure-aware Python & Markdown splitters)
        │            │
        │            ├──► [ ChromaDB HNSW Vector Store ] (SentenceTransformers dense vectors)
        │            │
        │            └──► [ Whoosh BM25 Full-Text Index ] (Sparse keyword/token index)
        │
        ▼
[ Hybrid Search Engine (RRF) ] (Reciprocal Rank Fusion)
        │
        ├── (Optional) ──► [ FlashRank Neural Reranker ] (Cross-encoder precision)
        │
        ▼
[ MCP Protocol Handlers ] (Tools, Resources, Prompts)
        │
        ├──► `stdio` (Local desktop agents: Claude Desktop, Hermes)
        └──► `streamable-http` / `sse` (Remote web clients with OAuth 2.1 + PKCE)
```

---

## 2. Core Subsystems

### Universal Format Converters (`converter.py`)
Extracts clean Markdown from heterogeneous documents:
- **Office**: PyMuPDF (PDF/EPUB), python-docx (DOCX), python-pptx (PPTX), openpyxl (XLSX), striprtf (RTF).
- **Code & Notebooks**: nbformat with cell input/output rendering for `.ipynb`.
- **Databases**: SQLite table schema extraction and row preview sampling.
- **Fallbacks**: Microsoft MarkItDown integration and hexdump representation for binaries.

### Hybrid Retrieval & Ranking (`search.py`)
- **Sparse**: Whoosh BM25 with prefix query expansion and fuzzy fallback.
- **Dense**: Chroma vector similarity using normalized inner-product distance.
- **Fusion**: Reciprocal Rank Fusion ($RRF = \sum \frac{w_i}{k + r_i}$) balancing lexical precision and semantic recall without arbitrary coefficient scaling.
- **Reranker**: Optional neural re-scoring with FlashRank.

### Security & Path Sandboxing (`security.py`)
- Traversal validation (`safe_resolve`) ensuring paths cannot escape `root_dir` through symlink dereferences or parent directory tokens (`..`).
