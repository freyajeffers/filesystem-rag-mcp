import asyncio
from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import _ServerState


@pytest.mark.asyncio
async def test_quick_and_thorough_parallel_indexing(tmp_path: Path):
    doc = tmp_path / "research.txt"
    doc.write_text("ThoroughDeepVectorSemanticEmbeddingVerification2026", encoding="utf-8")

    settings = Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag")
    state = _ServerState(settings)

    # Start both background index tasks
    state.start_background_indexing()

    # Fast check: quick index should become ready
    await asyncio.wait_for(state._quick_index_ready.wait(), timeout=10.0)
    assert state._quick_index_ready.is_set()

    # Search with pure full-text (alpha=0.0) needs only quick index
    res_ft = await state.search(
        query="ThoroughDeepVectorSemanticEmbeddingVerification2026", top_k=5, alpha=0.0
    )
    assert res_ft["success"] is True
    assert len(res_ft["results"]) > 0

    # Complex tool requiring thorough vector index (alpha=0.5) with wait_for_indexing=True awaits thorough completion
    res_hybrid = await state.search(
        query="ThoroughDeepVectorSemanticEmbeddingVerification2026",
        top_k=5,
        alpha=0.5,
        wait_for_indexing=True,
    )
    assert res_hybrid["success"] is True
    assert len(res_hybrid["results"]) > 0
    assert state._thorough_index_ready.is_set()
