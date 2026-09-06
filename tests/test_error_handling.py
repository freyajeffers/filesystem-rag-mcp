from pathlib import Path
import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import _ServerState


@pytest.mark.asyncio
async def test_agent_readable_errors(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    valid_file = root / "valid.txt"
    valid_file.write_text("Valid content")
    sub_dir = root / "subdir"
    sub_dir.mkdir()

    settings = Settings(root_dir=root, data_dir=tmp_path / "data")
    state = _ServerState(settings)

    # 1. Path Traversal Error
    res_traversal = await state.read_file(rel_path="../outside.txt", max_bytes=None)
    assert res_traversal["success"] is False
    err = res_traversal["error"]
    assert err["code"] == "PATH_TRAVERSAL_BLOCKED"
    assert "suggested_fix" in err
    assert "allowed_root" in err["details"]

    # 2. File Not Found Error
    res_not_found = await state.read_file(rel_path="missing.txt", max_bytes=None)
    assert res_not_found["success"] is False
    err = res_not_found["error"]
    assert err["code"] == "FILE_NOT_FOUND"
    assert "suggested_fix" in err
    assert err["details"]["path"] == "missing.txt"

    # 3. Not A File Error (directory requested)
    res_dir = await state.read_file(rel_path="subdir", max_bytes=None)
    assert res_dir["success"] is False
    err = res_dir["error"]
    assert err["code"] == "NOT_A_REGULAR_FILE"
    assert "suggested_fix" in err

    # 4. Chunk Not Found Error
    res_chunk = await state.get_chunk("nonexistent_chunk_id")
    assert res_chunk["success"] is False
    err = res_chunk["error"]
    assert err["code"] == "CHUNK_NOT_FOUND"
    assert "refresh_index" in err["suggested_fix"]

    # 5. Invalid Search Parameter
    res_empty_search = await state.search(query="   ", top_k=5, alpha=0.5)
    assert res_empty_search["success"] is False
    assert res_empty_search["error"]["code"] == "INVALID_PARAMETER"

    res_invalid_k = await state.search(query="python", top_k=-1, alpha=0.5)
    assert res_invalid_k["success"] is False
    assert res_invalid_k["error"]["code"] == "INVALID_PARAMETER"

    res_invalid_alpha = await state.search(query="python", top_k=5, alpha=2.5)
    assert res_invalid_alpha["success"] is False
    assert res_invalid_alpha["error"]["code"] == "INVALID_PARAMETER"
