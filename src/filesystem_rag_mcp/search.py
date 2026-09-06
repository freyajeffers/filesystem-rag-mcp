"""Hybrid search engine combining full-text and vector results.

Combines hits from `FullTextStore` and `VectorStore` using reciprocal rank
fusion (RRF). The `alpha` setting can shift weight from one side to the
other: `alpha=0.0` is pure full-text, `alpha=1.0` is pure vector, anything
in between is hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .fulltext import FullTextStore, TextHit
from .vector import VectorHit, VectorStore


@dataclass(slots=True, frozen=True)
class SearchHit:
    chunk_id: str
    rel_path: str
    file_path: str
    start: int
    end: int
    text: str
    score: float  # hybrid score, higher is better
    sources: tuple[str, ...]  # ("text",) or ("vector",) or both


class SearchEngine:
    """Hybrid search over the full-text and vector indexes."""

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
    ) -> list[SearchHit]:
        k = top_k if top_k is not None else self.settings.default_top_k
        a = alpha if alpha is not None else self.settings.hybrid_alpha
        fetch = max(k * 3, 30)  # over-fetch for fusion

        text_hits: list[TextHit] = self.ft.search(query, fetch) if a < 1.0 else []
        vec_hits: list[VectorHit] = (
            self.vec.search(query, fetch) if a > 0.0 else []
        )

        scores: dict[str, float] = {}
        meta: dict[str, _HitMeta] = {}
        sources: dict[str, list[str]] = {}

        for rank, h in enumerate(text_hits, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + (1 - a) / (
                self.RRF_K + rank
            )
            meta[h.chunk_id] = _HitMeta(
                rel_path=h.rel_path,
                file_path=h.file_path,
                start=h.start,
                end=h.end,
                text=h.text,
            )
            sources.setdefault(h.chunk_id, []).append("text")

        for rank, h in enumerate(vec_hits, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + a / (
                self.RRF_K + rank
            )
            meta[h.chunk_id] = _HitMeta(
                rel_path=h.rel_path,
                file_path=h.file_path,
                start=h.start,
                end=h.end,
                text=h.text,
            )
            sources.setdefault(h.chunk_id, []).append("vector")

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: list[SearchHit] = []
        for cid, score in ranked:
            m = meta[cid]
            out.append(
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
        return out


@dataclass(slots=True, frozen=True)
class _HitMeta:
    rel_path: str
    file_path: str
    start: int
    end: int
    text: str
