from pathlib import Path

import pytest

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.patcher import patch_file
from filesystem_rag_mcp.server import _ServerState
from filesystem_rag_mcp.symbols import find_symbol_references


def test_patch_file_dry_run_and_execution(tmp_path: Path):
    target = tmp_path / "app.py"
    target.write_text("def hello():\n    return 'old'\n", encoding="utf-8")

    # Ambiguous or non-existent
    err = patch_file(tmp_path, "app.py", "nonexistent", "new")
    assert err["success"] is False
    assert err["error"]["code"] == "TARGET_NOT_FOUND"

    # Dry run
    dry = patch_file(tmp_path, "app.py", "'old'", "'dry_new'", dry_run=True)
    assert dry["success"] is True
    assert dry["dry_run"] is True
    assert "old" in target.read_text(encoding="utf-8")

    # Real patch
    real = patch_file(tmp_path, "app.py", "'old'", "'brand_new'")
    assert real["success"] is True
    assert real["dry_run"] is False
    assert "brand_new" in target.read_text(encoding="utf-8")


def test_find_symbol_references(tmp_path: Path):
    f1 = tmp_path / "service.py"
    f1.write_text(
        "from db import get_user\n\ndef run():\n    user = get_user(1)\n", encoding="utf-8"
    )
    f2 = tmp_path / "db.py"
    f2.write_text("def get_user(id):\n    return {}\n", encoding="utf-8")

    refs = find_symbol_references(tmp_path, symbol_name="get_user")
    assert refs["success"] is True
    assert refs["total_references"] >= 2
    kinds = [r["kind"] for r in refs["references"]]
    assert "import" in kinds
    assert "reference" in kinds


@pytest.mark.asyncio
async def test_search_compact_and_pack_line_numbers(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(
        "import sys\n\ndef compute(a, b):\n    return a + b\n\ndef main():\n    print(compute(1, 2))\n",
        encoding="utf-8",
    )
    settings = Settings(root_dir=root, data_dir=tmp_path / "data")
    state = _ServerState(settings)
    await state.ensure_indexed_synchronously(require_thorough=False)

    # Search with compact=True (using alpha=0.0 for fulltext)
    res = await state.search(
        query="compute",
        top_k=5,
        alpha=0.0,
        compact=True,
    )
    assert res["success"] is True
    assert res["compact"] is True
    assert len(res["results"]) > 0
    # Confirm text was stripped from compact response to save agent tokens
    assert "text" not in res["results"][0]
    assert "snippet" in res["results"][0]

    # Context packing with line numbers
    pack = await state.pack_context(
        query="compute",
        max_tokens=1000,
        alpha=0.0,
        include_line_numbers=True,
    )
    assert pack["success"] is True
    assert " | " in pack["markdown"]

    # Patch file through server state and verify immediate refresh
    patch_res = await state.patch_file(
        rel_path="main.py",
        old_string="return a + b",
        new_string="return a * b",
    )
    assert patch_res["success"] is True
    assert "return a * b" in (root / "main.py").read_text(encoding="utf-8")

    # Find symbol references through server state
    sym_refs = state.find_symbol_references(symbol_name="compute")
    assert sym_refs["success"] is True
    assert sym_refs["total_references"] >= 2
