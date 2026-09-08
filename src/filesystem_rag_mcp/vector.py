"""Vector search over chunked documents.

We use Chroma as the embedded vector store. Chroma persists to disk under
`<data_dir>/chroma/` and handles HNSW indexing internally.

Embeddings are computed with `sentence-transformers` using a small local
model (`all-MiniLM-L6-v2` by default). The same model is used for both
indexing and querying; the embedding dimension is recorded in settings so
we never mix vectors from different models in the same collection.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .indexing import Chunk
from .logging_setup import get_logger

log = get_logger("vector")


class VectorHit(BaseModel):
    """A single vector-similarity search result.

    Frozen so downstream code can't mutate a returned hit (the previous
    `@dataclass(slots=True, frozen=True)` contract). Pydantic gives us
    free `model_dump()` for the search JSON output.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Stable content-derived chunk identifier")
    rel_path: str = Field(description="Path relative to the indexed root")
    file_path: str = Field(description="Absolute filesystem path")
    start: int = Field(description="Char offset where the chunk begins in the source file")
    end: int = Field(description="Char offset one past the chunk's last char")
    text: str = Field(description="Chunk text used for embedding")
    score: float = Field(
        description="Cosine similarity in [0, 1] for normalized vectors"
    )


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

            # HuggingFace Hub emits a "You are sending unauthenticated
            # requests to the HF Hub. Please set a HF_TOKEN..." WARNING
            # every time the model is downloaded from the public Hub.
            # This project ships without a token by design (users bring
            # their own), so the warning is informational noise for every
            # CLI subcommand and first-run test. Silence it on the
            # specific logger that emits it. Other HF warnings still pass
            # through at WARNING/CRITICAL.
            try:
                from huggingface_hub.utils import logging as _hf_logging

                _hf_logging.set_verbosity_error()  # type: ignore[no-untyped-call]
            except Exception:
                pass

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
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return cast(list[list[float]], vectors.tolist())


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
        unique: dict[str, tuple[str, str, dict[str, Any], list[float]]] = {}
        for cid, doc, meta, emb in zip(ids, docs, metas, embs):
            unique[cid] = (cid, doc, meta, emb)
        self._collection.upsert(
            ids=[v[0] for v in unique.values()],
            documents=[v[1] for v in unique.values()],
            metadatas=cast(Any, [v[2] for v in unique.values()]),
            embeddings=cast(Any, [v[3] for v in unique.values()]),
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
            query_embeddings=cast(Any, [emb]),
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
            start_val = meta.get("start", 0)
            end_val = meta.get("end", 0)
            start_int = int(start_val) if isinstance(start_val, (int, str, float)) else 0
            end_int = int(end_val) if isinstance(end_val, (int, str, float)) else 0
            out.append(
                VectorHit(
                    chunk_id=cid,
                    rel_path=str(meta.get("rel_path", "")),
                    file_path=str(meta.get("file_path", "")),
                    start=start_int,
                    end=end_int,
                    text=str(doc),
                    score=sim,
                )
            )
        return out

    def count(self) -> int:
        return self._collection.count()
