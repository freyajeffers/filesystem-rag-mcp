from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import _ServerState


@pytest.mark.asyncio
async def test_batch_read_and_grep_tools(tmp_path: Path):
    sub = tmp_path / "src"
    sub.mkdir()
    f1 = sub / "service.py"
    f1.write_text(
        "class PaymentProcessor:\n    def process_transaction(self, amount):\n        return True\n"
    )

    f2 = sub / "models.py"
    f2.write_text("class TransactionRecord:\n    id: str\n    amount: float\n")

    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / ".fsrag",
    )
    state = _ServerState(settings)

    # 1. Test grep_search
    grep_res = state.grep(pattern="process_transaction", context_lines=1)
    assert grep_res["success"] is True
    assert grep_res["total_matches"] == 1
    match = grep_res["matches"][0]
    assert match["rel_path"] == "src/service.py"
    assert match["line_number"] == 2
    assert len(match["context"]) == 3  # line 1, 2 (match), 3

    # 2. Test read_files_batch
    batch_res = await state.read_files_batch(
        rel_paths=["src/service.py", "src/models.py", "nonexistent.py"]
    )
    assert batch_res["success"] is True
    assert batch_res["count"] == 3
    assert "src/service.py" in batch_res["files"]
    assert "class PaymentProcessor" in batch_res["files"]["src/service.py"]["markdown"]
    assert batch_res["files"]["nonexistent.py"]["error"]["code"] == "FILE_NOT_FOUND"

    state.stop()


@pytest.mark.asyncio
async def test_refresh_file_and_chunk_context(tmp_path: Path):
    doc = tmp_path / "article.md"
    doc.write_text(
        "# Title 1\nSection 1 text here with some details.\n\n"
        "# Title 2\nSection 2 text discussing algorithms.\n\n"
        "# Title 3\nSection 3 conclusion.\n"
    )

    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / ".fsrag",
        chunk_size=64,
        chunk_overlap=0,
    )
    state = _ServerState(settings)

    # Re-index single file selectively
    ref_res = await state.refresh_file(rel_path="article.md")
    assert ref_res["success"] is True
    assert ref_res["chunks_indexed"] >= 2

    # Search for Section 2
    search_res = await state.search(query="algorithms", top_k=5, alpha=0.5, fuzzy=False)
    assert search_res["success"] is True
    assert len(search_res["results"]) > 0
    target_cid = search_res["results"][0]["chunk_id"]

    # Test chunk context (neighbor retrieval)
    ctx_res = await state.get_chunk_context(chunk_id=target_cid, before_chunks=1, after_chunks=1)
    assert ctx_res["success"] is True
    assert ctx_res["chunk_id"] == target_cid
    assert "before" in ctx_res
    assert "after" in ctx_res

    state.stop()


@pytest.mark.asyncio
async def test_fuzzy_search_expansion(tmp_path: Path):
    doc = tmp_path / "distributed.txt"
    doc.write_text("Consensus mechanisms in distributed systems rely on Byzantine fault tolerance.")

    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / ".fsrag",
    )
    state = _ServerState(settings)
    await state.ensure_indexed_synchronously(require_thorough=False)

    # Misspelling: 'Byzantiene' instead of 'Byzantine'
    res = await state.search(query="Byzantiene", top_k=5, alpha=0.5, fuzzy=True)
    assert res["success"] is True
    assert len(res["results"]) >= 1
    assert "distributed.txt" in res["results"][0]["rel_path"]

    state.stop()
