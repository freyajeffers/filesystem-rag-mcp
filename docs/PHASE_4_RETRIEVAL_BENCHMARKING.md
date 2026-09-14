# Phase 4: Retrieval Quality Benchmarking & Regression Gates

## 1. Architectural Scope
Phase 4 provides an automated evaluation harness and quantitative regression gates to track search quality metrics (MRR@5 and Hit@1) and latency budgets.

## 2. Benchmark Dataset
A curated dataset of 25+ queries covering:
1. Exact Keyword (FTS5 / BM25)
2. Conceptual Semantic (Vector Cosine)
3. Code Syntax & Symbol
4. Hierarchical Heading & Breadcrumbs

## 3. Evaluation Modes
1. Pure Lexical (BM25)
2. Pure Dense Vector
3. Hybrid RRF ($k=60$)
4. Two-Stage Hybrid + Cross-Encoder Reranking

## 4. Latency SLA Budgets
- Lexical search: < 5 ms
- Vector search: < 15 ms
- Cross-encoder rerank: < 45 ms
- Fusion & formatting: < 10 ms
- End-to-end average: < 75 ms (p95 < 100 ms)

## 5. CLI & CI Integration
- CLI command: `filesystem-rag-mcp benchmark`
- Automated CI testing workflow.
