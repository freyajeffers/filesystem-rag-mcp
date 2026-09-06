"""Vector search over chunked documents.

We use Chroma as the embedded vector store. Chroma persists to disk under
`<data_dir>/chroma/` and handles HNSW indexing internally.

Embeddings are computed with `sentence-transformers` using a small local
model (`all-MiniLM-L6-v2` by default). The same model is used for both
indexing and querying; the embedding dimension is recorded in settings so
we never mix vectors from different models in the same collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import Settings
from .logging_setup import get_logger

log = get_logger("vector")
from .indexing import Chunk


@dataclass(slots=True, frozen=True)
class VectorHit:
    chunk_id: str
    rel_path: str
    file_path: str
    start: int
    end: int
    text: str
    score: float  # cosine similarity in [0, 1] for normalized vectors


class Embedder:
    """Wraps a sentence-transformers model with explicit caching and offline tolerance."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._offline_unavailable = False

        if settings.offline_mode:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                settings.embedding_model,
                local_files_only=settings.offline_mode,
            )
        except Exception as exc:
            log.warning(
                "embedder_init_degraded",
                model=settings.embedding_model,
                offline=settings.offline_mode,
                error=str(exc),
            )
            self._offline_unavailable = True

    def is_available(self) -> bool:
        return self._model is not None and not self._offline_unavailable

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or self._model is None:
            return [[0.0] * self.settings.embedding_dim for _ in texts]
        # normalize_embeddings=True -> cosine sim via inner product
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.tolist()


class VectorStore:
    """Chroma-backed vector index."""

    COLLECTION_NAME = "filesystem_rag_chunks"

    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings
        self.embedder = embedder
        chroma_dir = settings.data_dir / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [
            {
                "rel_path": c.rel_path,
                "file_path": c.file_path,
                "start": c.start,
                "end": c.end,
            }
            for c in chunks
        ]
        embs = self.embedder.embed(docs)
        # Chroma wants lists, not tuples; the IDs must be unique.
        # We de-duplicate by chunk_id, preferring the latest occurrence.
        unique: dict[str, tuple[str, str, dict[str, object], list[float]]] = {}
        for cid, doc, meta, emb in zip(ids, docs, metas, embs):
            unique[cid] = (cid, doc, meta, emb)
        self._collection.upsert(
            ids=[v[0] for v in unique.values()],
            documents=[v[1] for v in unique.values()],
            metadatas=[v[2] for v in unique.values()],
            embeddings=[v[3] for v in unique.values()],
        )

    def delete_by_rel_path(self, rel_path: str) -> None:
        self._collection.delete(where={"rel_path": rel_path})

    def delete_by_chunk_id(self, chunk_id: str) -> None:
        self._collection.delete(ids=[chunk_id])

    def all_chunk_ids(self) -> set[str]:
        result = self._collection.get(include=[])
        return set(result.get("ids", []))

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        if not query.strip() or not self.embedder.is_available():
            return []
        emb = self.embedder.embed([query])[0]
        result = self._collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        out: list[VectorHit] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance = 1 - cos_sim; for normalized vectors this is
            # in [0, 2]. We clamp and convert to similarity.
            try:
                d = float(dist)
            except (TypeError, ValueError):
                continue
            sim = max(0.0, min(1.0, 1.0 - d))
            out.append(
                VectorHit(
                    chunk_id=cid,
                    rel_path=str(meta.get("rel_path", "")),
                    file_path=str(meta.get("file_path", "")),
                    start=int(meta.get("start", 0)),
                    end=int(meta.get("end", 0)),
                    text=str(doc),
                    score=sim,
                )
            )
        return out

    def count(self) -> int:
        return self._collection.count()
