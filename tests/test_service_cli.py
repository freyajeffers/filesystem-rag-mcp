import argparse
from pathlib import Path
from unittest.mock import patch

from filesystem_rag_mcp.cli import _build_subcommand_parser, _run_service


def test_cli_subcommand_parser_service():
    parser = _build_subcommand_parser("filesystem-rag-mcp")
    args = parser.parse_args(["service", "install", "--profile", "notes", "--json"])
    assert args.command == "service"
    assert args.service_action == "install"
    assert args.profile == "notes"
    assert args.as_json is True


def test_cli_subcommand_parser_benchmark():
    parser = _build_subcommand_parser("filesystem-rag-mcp")
    args = parser.parse_args(["benchmark", "--json"])
    assert args.command == "benchmark"
    assert args.as_json is True


def test_cli_subcommand_parser_optimize():
    parser = _build_subcommand_parser("filesystem-rag-mcp")
    args = parser.parse_args(["optimize", "--json"])
    assert args.command == "optimize"
    assert args.as_json is True


def test_service_install_action(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    args = argparse.Namespace(
        service_action="install",
        profile="notes",
        root_dir=tmp_path / "notes",
        data_dir=tmp_path / ".fsrag",
        as_json=True,
    )

    with patch("subprocess.run") as mock_run:
        exit_code = _run_service(args)
        assert exit_code == 0
        assert mock_run.called

    service_file = tmp_path / ".config" / "systemd" / "user" / "mcp-rag-watcher.service"
    timer_file = tmp_path / ".config" / "systemd" / "user" / "mcp-rag-maintenance.timer"
    assert service_file.exists()
    assert timer_file.exists()
    content = service_file.read_text(encoding="utf-8")
    assert "FSRAG_PROFILE=notes" in content
