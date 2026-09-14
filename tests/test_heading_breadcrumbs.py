from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.indexing import FileMeta, chunk_file
from filesystem_rag_mcp.semantic_chunker import (
    extract_heading_breadcrumbs,
)
from filesystem_rag_mcp.server import _ServerState


def test_heading_breadcrumbs_extraction():
    md = """# Project Title
Introduction paragraph text.

## Architecture
Overview of system architecture.

### Storage
SQLite WAL storage details.

## Deployment
Systemd unit setup.
"""
    spans = extract_heading_breadcrumbs(md)
    # Span for introduction
    assert spans[0][2] == "Project Title"
    # Span for architecture
    assert spans[1][2] == "Project Title > Architecture"
    # Span for storage
    assert spans[2][2] == "Project Title > Architecture > Storage"
    # Span for deployment (after storage and architecture popped)
    assert spans[3][2] == "Project Title > Deployment"


def test_chunk_file_attaches_breadcrumbs(tmp_path: Path):
    doc_path = tmp_path / "notes.md"
    doc_path.write_text(
        "# Knowledge Base\n\n## Section A\n\nDetailed content for section A with multiple sentences to ensure chunk size boundaries.\n\n## Section B\n\nContent for section B with additional explanations.",
        encoding="utf-8",
    )

    fm = FileMeta(
        abs_path=doc_path,
        rel_path="notes.md",
        size=doc_path.stat().st_size,
        mtime_ns=doc_path.stat().st_mtime_ns,
        sha256="fake_sha",
    )

    settings = Settings(
        root_dir=tmp_path, data_dir=tmp_path / ".fsrag", chunk_size=64, chunk_overlap=0
    )
    chunks = chunk_file(fm, settings)

    assert len(chunks) >= 2
    assert any("Knowledge Base" in c.breadcrumbs for c in chunks)


@pytest.mark.asyncio
async def test_server_search_returns_breadcrumbs(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "arch.md").write_text(
        "# Master Spec\n\n## Database Storage Engine\n\nUniqueTokenBreadcrumbTarget is stored here.",
        encoding="utf-8",
    )

    settings = Settings(root_dir=vault, data_dir=tmp_path / ".fsrag")
    state = _ServerState(settings)
    await state.refresh_index(full_rebuild=True)

    res = await state.search(query="UniqueTokenBreadcrumbTarget", top_k=5)
    assert res["success"] is True
    assert len(res["results"]) > 0
    hit = res["results"][0]
    assert "breadcrumbs" in hit
    assert "Master Spec" in hit["breadcrumbs"]
