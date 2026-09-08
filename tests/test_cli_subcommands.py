"""Tests for the new maintenance subcommands (doctor, search, stats, index).

These tests exercise the dispatcher (`_argv_uses_subcommand`,
`_build_subcommand_parser`) plus the end-to-end behavior of each
subcommand against an isolated temp directory. They are intentionally
not marked slow — quick-mode indexing uses no embedding model so it stays
cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

from filesystem_rag_mcp.cli import (
    _argv_uses_subcommand,
    _build_subcommand_parser,
    main,
)


def _make_corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "hello.md").write_text(
        "# Hello\n\nRAG search combines full-text and vector retrieval.\n"
        "The quick brown fox jumps over the lazy dog.\n"
    )
    (root / "notes.md").write_text(
        "# Notes\n\nSemantic chunking and full-text indexing share a chunk store.\n"
        "Neural reranking improves precision at the cost of latency.\n"
    )
    return root


def test_argv_uses_subcommand_positional():
    assert _argv_uses_subcommand(["doctor"]) == "doctor"
    assert _argv_uses_subcommand(["search", "foo"]) == "search"
    assert _argv_uses_subcommand(["stats"]) == "stats"
    assert _argv_uses_subcommand(["index"]) == "index"


def test_argv_uses_subcommand_legacy_flags():
    """Legacy flag-only invocations must NOT be mistaken for subcommands."""
    assert _argv_uses_subcommand(["--transport", "stdio"]) is None
    assert _argv_uses_subcommand(["--doctor"]) is None
    assert _argv_uses_subcommand(["--create-client", "x"]) is None
    assert _argv_uses_subcommand([]) is None
    # Any unknown positional is also not a subcommand
    assert _argv_uses_subcommand(["bogus"]) is None


def test_build_subcommand_parser_search():
    parser = _build_subcommand_parser("fsrag-test")
    ns = parser.parse_args(["search", "hello world", "--top-k", "5", "--fuzzy"])
    assert ns.command == "search"
    assert ns.query == "hello world"
    assert ns.top_k == 5
    assert ns.fuzzy is True


def test_build_subcommand_parser_index():
    parser = _build_subcommand_parser("fsrag-test")
    ns = parser.parse_args(["index", "--thorough"])
    assert ns.command == "index"
    assert ns.thorough is True


def test_build_subcommand_parser_stats_json():
    parser = _build_subcommand_parser("fsrag-test")
    ns = parser.parse_args(["stats", "--json"])
    assert ns.command == "stats"
    assert ns.as_json is True


def test_index_subcommand_quick_and_thorough(tmp_path):
    """Quick mode indexes text only; thorough adds vector chunks."""
    root = _make_corpus(tmp_path / "corpus")
    data = tmp_path / "data"

    rc, out = _capture_main(["index", "--root-dir", str(root), "--data-dir", str(data)])
    assert rc == 0
    assert "files indexed:" in out
    assert "2" in out  # 2 markdown files

    _, stats_quick_str = _capture_main(
        ["stats", "--json", "--root-dir", str(root), "--data-dir", str(data)]
    )
    stats_quick = json.loads(stats_quick_str)
    assert stats_quick["fulltext_chunks"] == 2
    assert stats_quick["vector_chunks"] == 0

    # thorough mode (re-index with thorough)
    rc, _ = _capture_main(["index", "--thorough", "--root-dir", str(root), "--data-dir", str(data)])
    assert rc == 0
    _, stats_full_str = _capture_main(
        ["stats", "--json", "--root-dir", str(root), "--data-dir", str(data)]
    )
    stats_full = json.loads(stats_full_str)
    assert stats_full["fulltext_chunks"] == 2
    assert stats_full["vector_chunks"] == 2


def _capture_main(argv: list[str]) -> tuple[int, str]:
    """Run main(argv) capturing stdout; tolerate sys.exit raises.

    `main()` is allowed to call sys.exit() on completion (a legitimate
    CLI convention), so we catch SystemExit and read its code. This lets
    us assert on the rendered output without subverting that contract.
    """
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    rc = 0
    with redirect_stdout(buf):
        try:
            main(argv)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
    return rc, buf.getvalue()


def test_search_subcommand_returns_hits(tmp_path):
    root = _make_corpus(tmp_path / "corpus")
    data = tmp_path / "data"
    rc, _ = _capture_main(["index", "--root-dir", str(root), "--data-dir", str(data)])
    assert rc == 0

    rc, buf_str = _capture_main(
        [
            "search",
            "vector embeddings",
            "--root-dir",
            str(root),
            "--data-dir",
            str(data),
            "--json",
        ]
    )
    assert rc == 0
    hits = json.loads(buf_str)
    assert len(hits) >= 1
    top = hits[0]
    assert "rel_path" in top and "score" in top and "sources" in top


def test_stats_subcommand_returns_counts(tmp_path):
    root = _make_corpus(tmp_path / "corpus")
    data = tmp_path / "data"

    # Before indexing: both counts are zero
    _, out = _capture_main(["stats", "--json", "--root-dir", str(root), "--data-dir", str(data)])
    stats = json.loads(out)
    assert stats["fulltext_chunks"] == 0
    assert stats["vector_chunks"] == 0

    # After quick indexing: text chunks present, vector chunks still zero
    rc, _ = _capture_main(["index", "--root-dir", str(root), "--data-dir", str(data)])
    assert rc == 0
    _, stats_str = _capture_main(
        ["stats", "--json", "--root-dir", str(root), "--data-dir", str(data)]
    )
    stats = json.loads(stats_str)
    assert stats["fulltext_chunks"] >= 1
    assert stats["vector_chunks"] == 0


def test_legacy_flag_path_still_works(capsys):
    """The legacy ``--doctor`` flag must still work alongside subcommands."""
    # Make sure legacy path doesn't crash on argument parsing
    # We do NOT actually want to start a server here, but we want to confirm
    # the dispatcher doesn't accidentally route legacy flags to subcommands.
    assert _argv_uses_subcommand(["--doctor"]) is None


def test_subcommand_dispatch_isolated(monkeypatch):
    """If argv[0] is a known subcommand, _argv_uses_subcommand returns it."""
    assert _argv_uses_subcommand(["doctor"]) == "doctor"
    assert _argv_uses_subcommand(["search", "x", "--top-k", "3"]) == "search"
