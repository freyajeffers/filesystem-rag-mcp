from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import _ServerState


@pytest.mark.asyncio
async def test_search_non_blocking_with_index_status(tmp_path: Path):
    doc = tmp_path / "fast_note.txt"
    doc.write_text("ImmediateNonBlockingCorpusLookup2026", encoding="utf-8")

    settings = Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag")
    state = _ServerState(settings)

    # 1. Query status before background indexing starts
    status_before = await state.get_indexing_status(wait=False)
    assert status_before["quick_index"]["ready"] is False
    assert status_before["thorough_index"]["ready"] is False

    # 2. Trigger background indexing
    state.start_background_indexing()

    # 3. Search immediately with wait_for_indexing=False -> DOES NOT BLOCK
    # Even if hybrid vector search (alpha=0.5) is requested, it falls back to text search
    # and tells caller background indexing is underway
    res = await state.search(
        query="ImmediateNonBlockingCorpusLookup2026",
        top_k=5,
        alpha=0.5,
        wait_for_indexing=False,
    )
    assert res["success"] is True
    assert "index_state" in res
    assert "quick_index_ready" in res["index_state"]

    # 4. Caller explicitly requests to wait for thorough indexing to complete
    status_waited = await state.get_indexing_status(wait=True, timeout_seconds=15.0)
    assert status_waited["thorough_index"]["ready"] is True

    # 5. Subsequent search with wait_for_indexing=True returns with thorough vector indexing applied
    res_thorough = await state.search(
        query="ImmediateNonBlockingCorpusLookup2026",
        top_k=5,
        alpha=0.5,
        wait_for_indexing=True,
    )
    assert res_thorough["success"] is True
    assert res_thorough["alpha"] == 0.5
    assert res_thorough["index_state"]["thorough_index_ready"] is True
