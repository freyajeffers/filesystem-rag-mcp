import json
import sqlite3
from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import _ServerState


@pytest.mark.asyncio
async def test_targeted_sqlite_fetch(tmp_path: Path):
    db_file = tmp_path / "app.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE accounts (id INT PRIMARY KEY, username TEXT, balance REAL)")
    conn.execute("INSERT INTO accounts VALUES (1, 'alice', 500.50), (2, 'bob', 120.00)")
    conn.commit()
    conn.close()

    state = _ServerState(Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag"))

    # Targeted query
    res = await state.fetch_targeted(
        rel_path="app.db",
        query="SELECT username, balance FROM accounts WHERE balance > 200",
    )
    assert res["success"] is True
    assert res["rows_returned"] == 1
    assert res["rows"][0] == {"username": "alice", "balance": 500.50}


@pytest.mark.asyncio
async def test_targeted_json_fetch(tmp_path: Path):
    json_file = tmp_path / "config.json"
    data = {
        "server": {"host": "0.0.0.0", "port": 8080},
        "database": {"url": "postgres://localhost/db"},
        "features": ["rag", "oauth", "search"],
    }
    json_file.write_text(json.dumps(data), encoding="utf-8")

    state = _ServerState(Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag"))

    # Nested path
    res = await state.fetch_targeted(rel_path="config.json", query="server.port")
    assert res["success"] is True
    assert res["result"] == 8080

    # Array path
    res_array = await state.fetch_targeted(rel_path="config.json", query="features[1]")
    assert res_array["success"] is True
    assert res_array["result"] == "oauth"


@pytest.mark.asyncio
async def test_targeted_csv_fetch(tmp_path: Path):
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text(
        "timestamp,metric,value\n1000,cpu,85.2\n1001,ram,64.0\n1002,cpu,91.4\n",
        encoding="utf-8",
    )

    state = _ServerState(Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag"))

    # Filter by metric=cpu and select value only
    res = await state.fetch_targeted(
        rel_path="metrics.csv",
        columns=["value"],
        filter_col="metric",
        filter_value="cpu",
    )
    assert res["success"] is True
    assert res["columns"] == ["value"]
    assert len(res["rows"]) == 2
    assert res["rows"][0]["value"] == "85.2"
    assert res["rows"][1]["value"] == "91.4"


@pytest.mark.asyncio
async def test_search_results_include_contextual_snippets(tmp_path: Path):
    text_file = tmp_path / "article.txt"
    text_file.write_text(
        "This is an introductory preamble that has lots of filler text to check bounds. "
        "The critical discovery about quantum computing was made in late 2026 by scientists. "
        "Here is more concluding boilerplate that extends past normal snippet boundaries.",
        encoding="utf-8",
    )

    state = _ServerState(Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag"))
    await state.refresh_index(full_rebuild=True)

    res = await state.search(query="quantum computing", top_k=5, alpha=0.5)
    assert "results" in res
    assert len(res["results"]) > 0
    hit = res["results"][0]
    assert "snippet" in hit
    assert "quantum computing" in hit["snippet"].lower()
