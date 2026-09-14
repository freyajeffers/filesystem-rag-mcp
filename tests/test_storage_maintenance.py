import sqlite3
from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.fulltext import FullTextStore
from filesystem_rag_mcp.maintenance import optimize_storage
from filesystem_rag_mcp.server import _ServerState


def test_optimize_storage_reclaims_and_checks_integrity(tmp_path: Path):
    data_dir = tmp_path / "fsrag_data"
    data_dir.mkdir()

    # Create dummy SQLite database with WAL and tables
    db_file = data_dir / "meta.sqlite"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA auto_vacuum = INCREMENTAL;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, content TEXT);")
    conn.executemany(
        "INSERT INTO items (content) VALUES (?);",
        [(f"sample content item {i}",) for i in range(500)],
    )
    conn.commit()

    # Delete half rows to create reclaimable pages
    conn.execute("DELETE FROM items WHERE id > 250;")
    conn.commit()
    conn.close()

    # Initialize Whoosh index
    settings = Settings(data_dir=data_dir)
    FullTextStore(settings)

    report = optimize_storage(settings)
    assert report.ok is True
    assert report.integrity == "ok"
    assert report.duration_ms >= 0.0
    assert "fulltext_optimized" in report.details
    assert report.details["fulltext_optimized"] is True


@pytest.mark.asyncio
async def test_server_optimize_database_tool(tmp_path: Path):
    settings = Settings(root_dir=tmp_path, data_dir=tmp_path / ".fsrag")
    state = _ServerState(settings)
    await state.refresh_index(full_rebuild=True)

    from filesystem_rag_mcp.server import build_server

    server = build_server(settings)
    # Verify optimize_database tool is registered
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "optimize_database" in tool_names
    assert "search_notes" in tool_names
    assert "index_file" in tool_names
    assert "reindex_directory" in tool_names
    assert "get_index_stats" in tool_names
