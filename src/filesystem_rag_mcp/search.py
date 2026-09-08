"""Hybrid Search Engine with Path/Glob Filtering and Neural Reranking.

Combines BM25 full-text search and vector similarity with Reciprocal Rank Fusion (RRF),
supports fnmatch/glob path constraints, and optional cross-encoder neural reranking
using FlashRank.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .fulltext import FullTextStore, TextHit
from .logging_setup import get_logger
from .vector import VectorHit, VectorStore

log = get_logger("search")

_ranker_instance: Any = None


def _get_ranker() -> Any:
    global _ranker_instance
    if _ranker_instance is None:
        try:
            from flashrank import Ranker

            _ranker_instance = Ranker()
        except Exception as err:
            log.warning("flashrank_init_failed", error=str(err))
            _ranker_instance = False
    return _ranker_instance if _ranker_instance is not False else None


class SearchHit(BaseModel):
    """A single hybrid (text + vector) search result returned to MCP callers.

    Frozen so downstream consumers can't mutate a returned hit. The
    `sources` field carries provenance tags ("text", "vector", or both)
    so the caller can see which index paths contributed to the score.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Stable content-derived chunk identifier")
    rel_path: str = Field(description="Path relative to the indexed root")
    file_path: str = Field(description="Absolute filesystem path")
    start: int = Field(description="Char offset where the chunk begins in the source file")
    end: int = Field(description="Char offset one past the chunk's last char")
    text: str = Field(description="Chunk text used for embedding / matching")
    score: float = Field(description="Hybrid or reranked score; higher is better")
    sources: tuple[str, ...] = Field(
        description=(
            "Provenance tags naming which indexes contributed to this hit; "
            "subset of {'text', 'vector'}"
        )
    )


class SearchEngine:
    """Hybrid search over the full-text and vector indexes with reranking and path filtering."""

    RRF_K = 60.0  # standard RRF constant

    def __init__(
        self,
        settings: Settings,
        ft: FullTextStore,
        vec: VectorStore,
    ) -> None:
        self.settings = settings
        self.ft = ft
        self.vec = vec

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        alpha: float | None = None,
        path_glob: str | None = None,
        rerank: bool = False,
        fuzzy: bool = False,
    ) -> list[SearchHit]:
        k = top_k if top_k is not None else self.settings.default_top_k
        a = alpha if alpha is not None else self.settings.hybrid_alpha
        # Over-fetch for fusion, path filtering, and reranking
        fetch = max(k * 4, 50)

        text_hits: list[TextHit] = self.ft.search(query, fetch, fuzzy=fuzzy) if a < 1.0 else []
        vec_hits: list[VectorHit] = self.vec.search(query, fetch) if a > 0.0 else []

        # Apply path_glob filter if requested
        if path_glob and path_glob.strip():
            pat = path_glob.strip().lstrip("/")
            import re
            from pathlib import PurePath

            def _matches(p: str) -> bool:
                cleaned = p.lstrip("/")
                if PurePath(cleaned).match(pat) or fnmatch.fnmatch(cleaned, pat):
                    return True
                if "**" in pat:
                    reg = re.escape(pat).replace(r"\*\*/", "(.+/)?").replace(r"\*", "[^/]*")
                    if re.fullmatch(reg, cleaned):
                        return True
                return False

            text_hits = [h for h in text_hits if _matches(h.rel_path)]
            vec_hits = [h for h in vec_hits if _matches(h.rel_path)]

        scores: dict[str, float] = {}
        meta: dict[str, _HitMeta] = {}
        sources: dict[str, list[str]] = {}

        for rank, h in enumerate(text_hits, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + (1 - a) / (self.RRF_K + rank)
            meta[h.chunk_id] = _HitMeta(
                rel_path=h.rel_path,
                file_path=h.file_path,
                start=h.start,
                end=h.end,
                text=h.text,
            )
            sources.setdefault(h.chunk_id, []).append("text")

        for rank, vh in enumerate(vec_hits, start=1):
            scores[vh.chunk_id] = scores.get(vh.chunk_id, 0.0) + a / (self.RRF_K + rank)
            meta[vh.chunk_id] = _HitMeta(
                rel_path=vh.rel_path,
                file_path=vh.file_path,
                start=vh.start,
                end=vh.end,
                text=vh.text,
            )
            sources.setdefault(vh.chunk_id, []).append("vector")

        if not scores:
            return []

        # Top candidate pool
        pool_size = max(k * 2, 20)
        top_candidates = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:pool_size]

        # Neural reranking with FlashRank if requested
        if rerank:
            ranker = _get_ranker()
            if ranker is not None:
                try:
                    from flashrank import RerankRequest

                    passages = [{"id": cid, "text": meta[cid].text} for cid, _ in top_candidates]
                    rerank_req = RerankRequest(query=query, passages=passages)
                    reranked = ranker.rerank(rerank_req)

                    out: list[SearchHit] = []
                    for item in reranked[:k]:
                        cid = item["id"]
                        m = meta[cid]
                        out.append(
                            SearchHit(
                                chunk_id=cid,
                                rel_path=m.rel_path,
                                file_path=m.file_path,
                                start=m.start,
                                end=m.end,
                                text=m.text,
                                score=float(item.get("score", 0.0)),
                                sources=(*sources.get(cid, []), "rerank"),
                            )
                        )
                    return out
                except Exception as exc:
                    log.warning("reranking_execution_failed", error=str(exc))

        # Standard RRF ordering
        ranked = top_candidates[:k]
        out_standard: list[SearchHit] = []
        for cid, score in ranked:
            m = meta[cid]
            out_standard.append(
                SearchHit(
                    chunk_id=cid,
                    rel_path=m.rel_path,
                    file_path=m.file_path,
                    start=m.start,
                    end=m.end,
                    text=m.text,
                    score=score,
                    sources=tuple(sources.get(cid, [])),
                )
            )
        return out_standard


@dataclass(slots=True, frozen=True)
class _HitMeta:
    rel_path: str
    file_path: str
    start: int
    end: int
    text: str
