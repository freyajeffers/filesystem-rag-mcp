from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.errors import (
    AgentError,
    ambiguous_match_error,
    directory_not_found_error,
    fetch_error,
    file_not_found_error,
    target_not_found_error,
    write_error,
)
from filesystem_rag_mcp.fetcher import fetch_sqlite_query
from filesystem_rag_mcp.git_ops import git_search
from filesystem_rag_mcp.graph import CorpusGraphBuilder
from filesystem_rag_mcp.server import _ServerState

REQUIRED_TOP_KEYS = {"success", "error"}
REQUIRED_ERR_KEYS = {"code", "message", "suggested_fix", "details"}


def _assert_envelope(payload: dict, expected_code: str) -> None:
    """All error envelopes share the canonical AgentError shape."""
    assert payload["success"] is False
    assert REQUIRED_TOP_KEYS.issubset(payload.keys()), (
        f"missing top-level keys: {REQUIRED_TOP_KEYS - payload.keys()}"
    )
    err = payload["error"]
    assert err["code"] == expected_code
    assert REQUIRED_ERR_KEYS.issubset(err.keys()), (
        f"missing error keys: {REQUIRED_ERR_KEYS - err.keys()}"
    )
    assert err["message"], "message must be non-empty"
    assert err["suggested_fix"], "suggested_fix must be non-empty"
    assert isinstance(err["details"], dict), "details must be a dict"


def test_agent_error_model_dump_matches_to_dict():
    """AgentError.model_dump() and .to_dict() produce the canonical envelope."""
    err = file_not_found_error("/x", "/root")
    assert err["success"] is False
    assert err["error"]["code"] == "FILE_NOT_FOUND"
    # Round-trip via Pydantic
    agent = AgentError.model_validate(err["error"])
    assert agent.model_dump() == err["error"]


def test_fetch_error_envelope_shape():
    payload = fetch_error("SQLite", "query 'SELECT'", "no such table")
    _assert_envelope(payload, "FETCH_FAILED")
    assert payload["error"]["details"]["format"] == "SQLite"
    assert "no such table" in payload["error"]["message"]


def test_directory_not_found_error_envelope_shape():
    payload = directory_not_found_error("missing_dir", root="/workspace")
    _assert_envelope(payload, "DIRECTORY_NOT_FOUND")
    assert payload["error"]["details"]["sub_dir"] == "missing_dir"
    assert payload["error"]["details"]["root"] == "/workspace"


def test_target_not_found_error_envelope_shape():
    payload = target_not_found_error("app.py", "def foo():")
    _assert_envelope(payload, "TARGET_NOT_FOUND")
    assert payload["error"]["details"]["rel_path"] == "app.py"


def test_ambiguous_match_error_envelope_shape():
    payload = ambiguous_match_error("app.py", occurrences=3)
    _assert_envelope(payload, "AMBIGUOUS_MATCH")
    assert payload["error"]["details"]["occurrences"] == 3


def test_write_error_envelope_shape():
    payload = write_error("app.py", "Permission denied")
    _assert_envelope(payload, "WRITE_ERROR")
    assert payload["error"]["details"]["os_error"] == "Permission denied"


def test_corpus_graph_directory_not_found(tmp_path: Path):
    """build_graph() on a missing sub_dir must produce the canonical envelope."""
    builder = CorpusGraphBuilder(tmp_path)
    payload = builder.build_graph(sub_dir="nonexistent_dir")
    _assert_envelope(payload, "DIRECTORY_NOT_FOUND")


def test_fetcher_sqlite_error_is_canonical(tmp_path: Path):
    """fetch_sqlite_query() on a corrupt SQL string uses the canonical envelope."""
    fake_db = tmp_path / "fake.db"
    fake_db.write_bytes(b"not a real sqlite database")
    payload = fetch_sqlite_query(fake_db, sql="SELECT 1", limit=10)
    _assert_envelope(payload, "FETCH_FAILED")
    assert payload["error"]["details"]["format"] == "SQLite"


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

    # 8. patch_file target not found (existing file, wrong target)
    state.settings.root_dir.mkdir(parents=True, exist_ok=True)
    existing = tmp_path / "exists.txt"
    existing.write_text("hello world\n", encoding="utf-8")
    res = await state.patch_file(
        rel_path="exists.txt", old_string="nonexistent_marker", new_string="x"
    )
    assert res["success"] is False
    assert res["error"]["code"] == "TARGET_NOT_FOUND"

    state.stop()
