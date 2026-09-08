"""Full-text search over chunked documents.

We use Whoosh as the embedded full-text engine. The schema indexes the chunk
text (analyzed with a stemmer for English), the relative path, the chunk id,
and the parent file's SHA-256 (so we can invalidate stale chunks on re-index).
The store is durable: the index lives under `<data_dir>/whoosh/`.

Score normalization: Whoosh returns unbounded positive scores. We
normalize per-query via min-max scaling into [0, 1] so they can be combined
with cosine-similarity vector scores (which live in [-1, 1], typically
[0, 1] for normalized embeddings).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, NUMERIC, TEXT, Schema
from whoosh.qparser import FuzzyTermPlugin, MultifieldParser, OrGroup
from whoosh.query import Query
from whoosh.searching import Hit

from .config import Settings
from .indexing import Chunk

SCHEMA = Schema(
    chunk_id=ID(stored=True, unique=True),
    rel_path=ID(stored=True),
    file_path=ID(stored=True),
    text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    start=NUMERIC(stored=True),
    end=NUMERIC(stored=True),
)


class TextHit(BaseModel):
    """A single full-text search result.

    Structurally identical to `vector.VectorHit` so the search engine can
    merge them without conversion. Pydantic gives us free `model_dump()`
    for the merged `SearchHit` JSON payload and field-level schema
    generation for MCP tool descriptors.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Stable content-derived chunk identifier")
    rel_path: str = Field(description="Path relative to the indexed root")
    file_path: str = Field(description="Absolute filesystem path")
    start: int = Field(description="Char offset where the chunk begins in the source file")
    end: int = Field(description="Char offset one past the chunk's last char")
    text: str = Field(description="Chunk text matched by the query")
    score: float = Field(description="Normalized score in [0, 1]")


class FullTextStore:
    """Persistent Whoosh index wrapper."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index_dir = settings.data_dir / "whoosh"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not index.exists_in(str(self.index_dir)):
            self._ix = index.create_in(str(self.index_dir), SCHEMA)
        else:
            self._ix = index.open_dir(str(self.index_dir))

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        writer = self._ix.writer()
        try:
            for chunk in chunks:
                writer.update_document(
                    chunk_id=chunk.chunk_id,
                    rel_path=chunk.rel_path,
                    file_path=chunk.file_path,
                    text=chunk.text,
                    start=chunk.start,
                    end=chunk.end,
                )
        finally:
            writer.commit()

    def delete_by_rel_path(self, rel_path: str) -> None:
        writer = self._ix.writer()
        try:
            writer.delete_by_term("rel_path", rel_path)
        finally:
            writer.commit()

    def delete_by_chunk_id(self, chunk_id: str) -> None:
        writer = self._ix.writer()
        try:
            writer.delete_by_term("chunk_id", chunk_id)
        finally:
            writer.commit()

    def all_chunk_ids(self) -> set[str]:
        ids: set[str] = set()
        with self._ix.searcher() as s:
            for item in s.iter_docs():
                # iter_docs yields (docnum, stored_fields) tuples in modern Whoosh
                if isinstance(item, tuple):
                    stored = (
                        item[1]
                        if len(item) > 1 and isinstance(item[1], dict)
                        else s.stored_fields(item[0])
                    )
                else:
                    stored = s.stored_fields(item)
                cid = stored.get("chunk_id")
                if cid:
                    ids.add(cid)
        return ids

    def search(self, query: str, top_k: int, fuzzy: bool = False) -> list[TextHit]:
        if not query.strip():
            return []
        with self._ix.searcher() as s:
            parser = MultifieldParser(["text", "rel_path"], schema=self._ix.schema, group=OrGroup)
            if fuzzy:
                parser.add_plugin(FuzzyTermPlugin())
                # Append ~1 fuzzy operator to single terms if not already present
                terms = [
                    f"{t}~1" if not any(c in t for c in "~*?^") and len(t) > 3 else t
                    for t in query.strip().split()
                ]
                effective_query = " ".join(terms)
            else:
                effective_query = query

            try:
                q: Query = parser.parse(effective_query)
                raw_hits = s.search(q, limit=top_k)
                hits = [_normalize_hit(s, h) for h in raw_hits]
            except Exception:
                hits = []

            # If normal search yielded nothing and fuzzy wasn't requested, retry with fuzzy expansion
            if not hits and not fuzzy:
                try:
                    parser.add_plugin(FuzzyTermPlugin())
                    terms = [
                        f"{t}~1" if not any(c in t for c in "~*?^") and len(t) > 3 else t
                        for t in query.strip().split()
                    ]
                    q = parser.parse(" ".join(terms))
                    raw_hits = s.search(q, limit=top_k)
                    hits = [_normalize_hit(s, h) for h in raw_hits]
                except Exception:
                    pass

            return hits

    def count(self) -> int:
        with self._ix.searcher() as s:
            return int(s.doc_count())


def _normalize_hit(searcher: object, hit: Hit) -> TextHit:
    """Convert a raw Whoosh hit to a `TextHit` with a [0, 1] score.

    Whoosh scores are unbounded; we use a sigmoid-style squash so the
    highest hit maps to ~1.0 and lower hits fall off quickly. This works
    well in practice for hybrid combination with cosine similarity.
    """
    raw = hit.score
    # squash: 1 - exp(-x / k). k tuned so that a "great" hit (~5.0) -> ~0.86
    k = 5.0
    norm = 1.0 - 2.71828 ** (-(raw / k))
    norm = max(0.0, min(1.0, norm))
    return TextHit(
        chunk_id=hit["chunk_id"],
        rel_path=hit["rel_path"],
        file_path=hit["file_path"],
        start=int(hit["start"]),
        end=int(hit["end"]),
        text=hit["text"],
        score=norm,
    )
