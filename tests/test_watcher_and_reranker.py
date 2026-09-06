import asyncio
from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import _ServerState


@pytest.mark.asyncio
async def test_live_directory_watcher_and_reranking(tmp_path: Path):
    doc1 = tmp_path / "article.txt"
    doc1.write_text(
        "General introduction about distributed consensus mechanisms.", encoding="utf-8"
    )

    settings = Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag")
    state = _ServerState(settings)
    state.start_background_indexing()

    # Initial search with reranking
    res1 = await state.search(
        query="distributed consensus",
        top_k=5,
        alpha=0.5,
        rerank=True,
        wait_for_indexing=True,
    )
    assert res1["success"] is True
    assert res1["results"]

    # Trigger live watcher by creating a new document
    doc2 = tmp_path / "raft.txt"
    doc2.write_text(
        "The Raft consensus algorithm is designed to be understandable and robust.",
        encoding="utf-8",
    )

    # Await watcher debounce
    await asyncio.sleep(2.0)

    # Search for new document
    res2 = await state.search(
        query="Raft consensus algorithm",
        top_k=5,
        alpha=0.5,
        wait_for_indexing=True,
    )
    assert res2["success"] is True
    assert any("raft" in hit["snippet"].lower() for hit in res2["results"])

    # Stop watcher cleanly
    state._watcher.stop()
