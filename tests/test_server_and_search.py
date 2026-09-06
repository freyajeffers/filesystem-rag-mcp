from pathlib import Path
import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import build_server
from filesystem_rag_mcp.indexing import discover_files, chunk_file
from filesystem_rag_mcp.vector import VectorStore, Embedder
from filesystem_rag_mcp.fulltext import FullTextStore
from filesystem_rag_mcp.search import SearchEngine
from filesystem_rag_mcp.oauth import MCPFileRAGAuthProvider


def test_server_creation_stdio(tmp_path: Path):
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
    )
    server = build_server(settings)
    assert server is not None
    assert server.name == "filesystem-rag-mcp"


def test_server_creation_http(tmp_path: Path):
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        oauth_issuer="http://127.0.0.1:8765",
        oauth_allow_dynamic_registration=True,
    )
    provider = MCPFileRAGAuthProvider(settings)
    server = build_server(settings, auth_provider=provider)
    assert server is not None


def test_fulltext_search_engine(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    f1 = docs / "article.txt"
    f1.write_text(
        "Modern neural network architectures for natural language processing. "
        "Transformers utilize self-attention mechanisms to capture dependencies across long text sequences."
    )
    f2 = docs / "code.py"
    f2.write_text(
        "def compute_attention(q, k, v):\n"
        "    scores = q @ k.T\n"
        "    return softmax(scores) @ v\n"
    )

    settings = Settings(
        root_dir=docs,
        data_dir=tmp_path / "data",
    )

    ft_store = FullTextStore(settings)
    files = discover_files(settings)
    assert len(files) == 2

    chunks = []
    for f in files:
        chunks.extend(chunk_file(f, settings))
    assert len(chunks) >= 2

    ft_store.upsert(chunks)
    assert ft_store.count() >= 2

    # Test full-text search
    ft_results = ft_store.search("attention", top_k=5)
    assert len(ft_results) > 0
