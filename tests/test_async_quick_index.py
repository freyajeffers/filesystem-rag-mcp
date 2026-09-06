import asyncio
from pathlib import Path
import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import _ServerState, build_server


@pytest.mark.asyncio
async def test_async_background_index_and_sync_drain(tmp_path: Path):
    # Setup test file
    doc = tmp_path / "hello.txt"
    doc.write_text("UniqueAsynchronousKeywordTest2026", encoding="utf-8")

    settings = Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag")
    state = _ServerState(settings)

    # Trigger background indexing
    state.start_background_indexing()

    # Complex tool requiring index (search) with wait_for_indexing=True awaits background indexing
    res = await state.search(
        query="UniqueAsynchronousKeywordTest2026",
        top_k=5,
        alpha=0.5,
        wait_for_indexing=True,
    )

    assert res["success"] is True
    assert len(res["results"]) > 0
    assert "UniqueAsynchronousKeywordTest2026" in res["results"][0]["snippet"]
