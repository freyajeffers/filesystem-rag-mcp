import asyncio
from pathlib import Path

import pytest

from filesystem_rag_mcp.cli import parse_args
from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.converter import convert_file_to_markdown
from filesystem_rag_mcp.server import _ServerState
from filesystem_rag_mcp.vector import Embedder, VectorStore


@pytest.mark.asyncio
async def test_large_file_guard(tmp_path: Path):
    large_file = tmp_path / "huge.log"
    # Write 150KB
    large_file.write_text("A" * 150_000)

    # Calling with limit of 50KB should trigger safety guard truncation
    md = convert_file_to_markdown(large_file, max_bytes=50_000)
    assert "Large File Notice" in md
    assert "Warning" in md
    assert "150,000 bytes" in md


@pytest.mark.asyncio
async def test_binary_vector_exclusion(tmp_path: Path):
    # Binary file
    bin_file = tmp_path / "test.bin"
    bin_file.write_bytes(bytes([0x00, 0xFF, 0xFE, 0x01, 0x02, 0x03] * 100))

    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello readable text world")

    # Settings with index_binary_files = True and index_binary_vectors = False
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / ".fsrag",
        index_binary_files=True,
        index_binary_vectors=False,
    )
    state = _ServerState(settings)
    await state.ensure_indexed_synchronously(require_thorough=True)

    # Verify that the binary chunk is NOT in vector store, but text IS in vector store
    vec_ids = state._vec.all_chunk_ids()
    ft_ids = state._ft.all_chunk_ids()

    # Fulltext has both
    assert len(ft_ids) >= 2
    # Vector store only has text file chunks
    assert len(vec_ids) == 1
    state.stop()


def test_offline_mode_embedder(tmp_path: Path):
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / ".fsrag",
        offline_mode=True,
    )
    embedder = Embedder(settings)
    # Embedder should not throw even if model lookup operates under offline restrictions
    assert embedder.is_available() or embedder._offline_unavailable
    vectors = embedder.embed(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == settings.embedding_dim


def test_config_snippet_cli():
    args = parse_args(["--config-snippet", "all", "--root-dir", "/tmp/test"])
    assert args.config_snippet == "all"
