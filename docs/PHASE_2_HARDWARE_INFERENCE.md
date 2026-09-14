# Phase 2: Hardware-Conscious Inference & Adaptive Thread Pinning

## 1. Architectural Scope
Phase 2 configures the machine learning inference layer to maximize embedding and reranking performance on the host hardware (AMD Ryzen 5 5600XT with 6 Zen 3 cores / 12 threads) while maintaining graceful fallback.

## 2. CPU Topology & Execution Provider Options
- `intra_op_num_threads = 6` (pinned to physical cores on host).
- `inter_op_num_threads = 1` (prevents operator contention).
- Full graph optimization enabled.
- Adaptive fallback dynamically detects physical cores on foreign architectures.

## 3. Startup Warm-Up Routine
- Eager model initialization at startup.
- Synthetic inference pass with dummy inputs to compile graphs and prime CPU caches before handling user queries.

## 4. Three-Stage Retrieval Funnel
- **Stage 1 (Broad Recall)**: Top 50 BM25 + top 50 dense vector cosine matches fused via Reciprocal Rank Fusion ($k=60$).
- **Stage 2 (Cross-Encoder Reranking)**: Top 20 candidates from RRF pool scored by cross-encoder model with sigmoid probability scaling.
- **Stage 3 (Final Projection)**: Top 5 projected candidates returned with relevance scores, chunk IDs, and breadcrumbs.

## 5. Query Cache Integration
Leverage in-memory query cache for query embeddings and reranking results for sub-millisecond repeated lookups.
