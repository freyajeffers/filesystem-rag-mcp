from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.git_ops import git_search
from filesystem_rag_mcp.server import _ServerState


@pytest.mark.asyncio
async def test_agent_error_schemas_on_invalid_parameters(tmp_path: Path):
    settings = Settings(root_dir=tmp_path, data_dir=tmp_path / ".data")
    state = _ServerState(settings)
    await state.ensure_indexed_synchronously(require_thorough=False)

    # 1. search empty query
    res = await state.search(query="")
    assert res["success"] is False
    assert "error" in res
    assert res["error"]["code"] == "INVALID_PARAMETER"
    assert "suggested_fix" in res["error"]

    # 2. deep_search invalid top_k
    res = await state.deep_search(query="test", top_k_per_subquery=0)
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_PARAMETER"

    # 3. pack_context invalid tokens
    res = await state.pack_context(query="test", max_tokens=10)
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_PARAMETER"

    # 4. get_corpus_graph invalid max_files
    res = state.get_corpus_graph(max_files=-5)
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_PARAMETER"

    # 5. git_search non-git repository
    non_git_dir = tmp_path / "not_git"
    non_git_dir.mkdir()
    res = git_search(non_git_dir, mode="commits")
    assert res["success"] is False
    assert res["error"]["code"] == "GIT_OPERATION_FAILED"
    assert "suggested_fix" in res["error"]

    # 6. git_search invalid mode
    (non_git_dir / ".git").mkdir()
    res = git_search(non_git_dir, mode="invalid_mode")
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_PARAMETER"

    # 7. patch_file non-existent file
    res = await state.patch_file(rel_path="missing.txt", old_string="a", new_string="b")
    assert res["success"] is False
    assert res["error"]["code"] == "FILE_NOT_FOUND"

    state.stop()
