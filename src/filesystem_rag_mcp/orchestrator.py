"""Multi-hop search and token-bounded context packing."""

from __future__ import annotations

import asyncio
from typing import Any

from .search import SearchEngine, SearchHit


class ContextOrchestrator:
    """Provides multi-hop query decomposition and token-compacted context bundling."""

    def __init__(self, engine: SearchEngine) -> None:
        self.engine = engine

    def deep_search(
        self,
        query: str,
        *,
        sub_queries: list[str] | None = None,
        top_k_per_subquery: int = 5,
        alpha: float = 0.5,
        rerank: bool = True,
    ) -> dict[str, Any]:
        queries = [query]
        if sub_queries:
            for sq in sub_queries:
                if sq.strip() and sq.strip() not in queries:
                    queries.append(sq.strip())
        else:
            # Simple heuristic subquery generation if not explicitly provided
            words = query.strip().split()
            if len(words) > 6:
                queries.append(" ".join(words[: len(words) // 2]))
                queries.append(" ".join(words[len(words) // 2 :]))

        combined_hits: dict[str, SearchHit] = {}
        hits_by_query: dict[str, list[dict[str, Any]]] = {}

        for q in queries:
            q_hits = self.engine.search(
                q,
                top_k=top_k_per_subquery,
                alpha=alpha,
                rerank=rerank,
            )
            hits_by_query[q] = [
                {
                    "chunk_id": h.chunk_id,
                    "rel_path": h.rel_path,
                    "score": round(h.score, 4),
                    "snippet": h.text[:200].replace("\n", " "),
                }
                for h in q_hits
            ]
            for h in q_hits:
                if h.chunk_id not in combined_hits or h.score > combined_hits[h.chunk_id].score:
                    combined_hits[h.chunk_id] = h

        # Sort all deduplicated hits by score descending
        sorted_hits = sorted(combined_hits.values(), key=lambda x: x.score, reverse=True)

        return {
            "success": True,
            "primary_query": query,
            "sub_queries": queries,
            "total_unique_chunks": len(sorted_hits),
            "hits_by_subquery": hits_by_query,
            "merged_chunks": [
                {
                    "chunk_id": h.chunk_id,
                    "rel_path": h.rel_path,
                    "score": round(h.score, 4),
                    "text": h.text,
                }
                for h in sorted_hits
            ],
        }

    def pack_context(
        self,
        query: str,
        *,
        max_tokens: int = 4000,
        alpha: float = 0.5,
        rerank: bool = True,
        path_glob: str | None = None,
    ) -> dict[str, Any]:
        """Pack search hits into a structured, token-bounded Markdown prompt."""
        # Estimate: ~4 characters per token
        char_limit = max_tokens * 4

        hits = self.engine.search(
            query,
            top_k=25,
            alpha=alpha,
            rerank=rerank,
            path_glob=path_glob,
        )

        # Group contiguous chunks by rel_path to synthesize clean file intervals
        grouped: dict[str, list[SearchHit]] = {}
        for h in hits:
            grouped.setdefault(h.rel_path, []).append(h)

        total_chars = 0
        used_chunks: list[str] = []
        packed_blocks: list[str] = []

        packed_blocks.append(f"# Context Bundle for Query: {query}\n")
        total_chars += len(packed_blocks[0])

        for rel_path, f_hits in grouped.items():
            f_hits.sort(key=lambda x: x.start)
            file_header = f"\n## File: {rel_path}\n"
            if total_chars + len(file_header) > char_limit:
                break
            packed_blocks.append(file_header)
            total_chars += len(file_header)

            for ch in f_hits:
                code_fence = f"```\n{ch.text.strip()}\n```\n"
                if total_chars + len(code_fence) > char_limit:
                    truncated_length = char_limit - total_chars
                    if truncated_length > 100:
                        packed_blocks.append(f"```\n{ch.text[:truncated_length]}...\n```\n")
                        used_chunks.append(ch.chunk_id)
                    break
                packed_blocks.append(code_fence)
                total_chars += len(code_fence)
                used_chunks.append(ch.chunk_id)

            if total_chars >= char_limit:
                break

        markdown_bundle = "".join(packed_blocks)

        return {
            "success": True,
            "query": query,
            "estimated_tokens": len(markdown_bundle) // 4,
            "max_tokens": max_tokens,
            "chunks_included": len(used_chunks),
            "unique_files_included": len({h.rel_path for h in hits if h.chunk_id in set(used_chunks)}),
            "markdown": markdown_bundle,
        }
