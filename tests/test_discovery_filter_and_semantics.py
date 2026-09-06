from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.indexing import FileMeta, chunk_file
from filesystem_rag_mcp.semantic_chunker import semantic_chunk_text
from filesystem_rag_mcp.server import _ServerState


def test_semantic_chunker_keeps_markdown_sections_intact():
    text = "# First\n\nFirst section body.\n\n# Second\n\nSecond section body."
    chunks = semantic_chunk_text(text, chunk_size=35, overlap=0)
    assert len(chunks) == 2
    assert chunks[0][2].startswith("# First")
    assert chunks[1][2].startswith("# Second")


def test_semantic_chunker_keeps_python_definitions_intact():
    text = "def alpha():\n    return 'alpha'\n\ndef beta():\n    return 'beta'\n"
    chunks = semantic_chunk_text(text, chunk_size=40, overlap=0)
    assert len(chunks) == 2
    assert "def alpha" in chunks[0][2]
    assert "def beta" in chunks[1][2]


@pytest.mark.asyncio
async def test_path_glob_filtered_search_and_directory_listing(tmp_path: Path):
    src = tmp_path / "src"
    docs = tmp_path / "docs"
    src.mkdir()
    docs.mkdir()
    (src / "app.py").write_text("def unique_component():\n    return 'glob-search-token'\n", encoding="utf-8")
    (docs / "guide.md").write_text("# Guide\n\nglob-search-token appears in documentation.\n", encoding="utf-8")

    state = _ServerState(Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag"))
    await state.refresh_index(full_rebuild=True)

    result = await state.search(
        query="glob-search-token",
        top_k=10,
        alpha=0.0,
        path_glob="src/**/*.py",
        wait_for_indexing=False,
    )
    assert result["success"] is True
    assert result["results"]
    assert all(hit["rel_path"] == "src/app.py" for hit in result["results"])

    listing = state.list_directory(rel_path="", max_depth=2, pattern="*.py")
    assert listing["success"] is True
    assert any(entry["rel_path"] == "src/app.py" for entry in listing["entries"])
    assert not any(entry.get("rel_path") == "docs/guide.md" for entry in listing["entries"])
