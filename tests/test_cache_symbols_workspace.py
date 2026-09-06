from pathlib import Path
import pytest
from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import _ServerState
from filesystem_rag_mcp.query_cache import QueryCache
from filesystem_rag_mcp.symbols import search_symbols


def test_query_cache_lru_and_invalidation():
    cache = QueryCache(max_entries=2, ttl_seconds=10.0)
    cache.set({"res": 1}, query="q1", top_k=5)
    cache.set({"res": 2}, query="q2", top_k=5)

    # Touch q1 so q1 becomes MRU and q2 becomes LRU
    assert cache.get(query="q1", top_k=5) == {"res": 1}

    # Overflow LRU by inserting q3
    cache.set({"res": 3}, query="q3", top_k=5)
    # q2 should now be evicted
    assert cache.get(query="q2", top_k=5) is None
    assert cache.get(query="q1", top_k=5) == {"res": 1}
    assert cache.get(query="q3", top_k=5) == {"res": 3}

    # Invalidation
    cache.invalidate()
    assert cache.get(query="q1", top_k=5) is None
    assert cache.size == 0


def test_search_symbols(tmp_path: Path):
    py_file = tmp_path / "module.py"
    py_file.write_text(
        '''
class PaymentGateway:
    """Handles payments."""
    def process_transaction(self, amount: float):
        pass

async def authenticate_user(token: str):
    """Verify auth token."""
    return True
''',
        encoding="utf-8",
    )

    res = search_symbols(tmp_path, name="Payment")
    assert res["success"] is True
    assert res["total_matches"] == 1
    assert res["symbols"][0]["name"] == "PaymentGateway"
    assert res["symbols"][0]["type"] == "class"

    res_fn = search_symbols(tmp_path, symbol_type="function")
    assert res_fn["success"] is True
    names = [s["name"] for s in res_fn["symbols"]]
    assert "authenticate_user" in names
    assert "process_transaction" in names


@pytest.mark.asyncio
async def test_workspace_management(tmp_path: Path):
    ws1 = tmp_path / "ws1"
    ws1.mkdir()
    (ws1 / "test.txt").write_text("hello", encoding="utf-8")

    ws2 = tmp_path / "ws2"
    ws2.mkdir()

    settings = Settings(root_dir=ws1, data_dir=tmp_path / "data")
    state = _ServerState(settings)

    # Initial workspace list
    workspaces = state.list_workspaces()
    assert "default" in workspaces["workspaces"]

    # Add workspace
    res = state.add_workspace(name="secondary", path=str(ws2))
    assert res["success"] is True
    assert "secondary" in state.list_workspaces()["workspaces"]

    # Symbol search on invalid workspace
    invalid_res = state.search_symbols(workspace="non_existent")
    assert invalid_res["success"] is False
    assert invalid_res["error"]["code"] == "INVALID_PARAMETER"
