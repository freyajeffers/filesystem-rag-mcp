import subprocess
from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.server import _ServerState


@pytest.mark.asyncio
async def test_deep_search_and_pack_context(tmp_path: Path):
    doc1 = tmp_path / "auth.py"
    doc1.write_text(
        "class TokenManager:\n"
        "    def generate_token(self, user_id: str) -> str:\n"
        "        return 'secret-token-xyz'\n"
        "    def revoke_token(self, token: str) -> bool:\n"
        "        return True\n"
    )

    doc2 = tmp_path / "db.py"
    doc2.write_text(
        "class Database:\n"
        "    def delete_session(self, token: str):\n"
        "        # execute SQL delete for token session\n"
        "        pass\n"
    )

    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / ".fsrag",
    )
    state = _ServerState(settings)
    await state.ensure_indexed_synchronously(require_thorough=False)

    # 1. Test deep_search with multi-query decomposition
    deep_res = await state.deep_search(
        query="token revocation database session flow",
        sub_queries=["generate_token user_id", "delete_session token"],
    )
    assert deep_res["success"] is True
    assert deep_res["total_unique_chunks"] >= 2
    assert "generate_token user_id" in deep_res["hits_by_subquery"]

    # 2. Test pack_context with strict token limit
    packed = await state.pack_context(
        query="token revocation",
        max_tokens=500,
    )
    assert packed["success"] is True
    assert packed["estimated_tokens"] <= 500
    assert "# Context Bundle for Query: token revocation" in packed["markdown"]
    assert "## File:" in packed["markdown"]

    state.stop()


def test_corpus_graph_builder(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f1 = pkg / "core.py"
    f1.write_text("class CoreEngine:\n    pass\n")

    f2 = pkg / "client.py"
    f2.write_text("from pkg.core import CoreEngine\n\nclass Client:\n    pass\n")

    readme = tmp_path / "README.md"
    readme.write_text("# Welcome\nSee [Core Docs](pkg/core.py) for details.\n")

    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / ".fsrag",
    )
    state = _ServerState(settings)
    graph = state.get_corpus_graph()

    assert graph["success"] is True
    assert graph["total_files"] >= 2
    assert len(graph["hubs"]) > 0
    # CoreEngine should have in-degree > 0
    assert any("core.py" in h["file"] for h in graph["hubs"])


def test_git_search_tool(tmp_path: Path):
    # Initialize a git repository
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)

    test_file = tmp_path / "hello.txt"
    test_file.write_text("line 1\nline 2\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit for git search test"], cwd=tmp_path, check=True
    )

    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / ".fsrag",
    )
    state = _ServerState(settings)

    # 1. Commit search
    commits = state.git_search(mode="commits")
    assert commits["success"] is True
    assert commits["count"] == 1
    assert "Initial commit" in commits["commits"][0]["subject"]

    # 2. Blame search
    blame = state.git_search(mode="blame", rel_path="hello.txt")
    assert blame["success"] is True
    assert blame["total_lines"] == 2
    assert blame["blame"][0]["author"] == "Test User"
