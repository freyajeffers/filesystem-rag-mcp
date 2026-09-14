from pathlib import Path

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.indexing import discover_files
from filesystem_rag_mcp.security import is_editor_artifact


def test_is_editor_artifact_patterns():
    # Vim/Neovim swap files
    assert is_editor_artifact(".file.md.swp") is True
    assert is_editor_artifact(".index.py.swo") is True
    assert is_editor_artifact(".draft.txt.swx") is True

    # Persistent undo
    assert is_editor_artifact(".notes.md.un~") is True

    # Backup files
    assert is_editor_artifact("notes.md~") is True
    assert is_editor_artifact("config.yaml~") is True

    # Temporary files
    assert is_editor_artifact("temp.tmp") is True
    assert is_editor_artifact(".scratch.tmp") is True

    # Atomic write artifacts
    assert is_editor_artifact(".goutputstream-ABCD123") is True

    # Hidden directories
    assert is_editor_artifact(Path(".git/objects/abc")) is True
    assert is_editor_artifact(Path(".obsidian/workspace.json")) is True
    assert is_editor_artifact(Path(".trash/deleted_note.md")) is True

    # Normal valid files
    assert is_editor_artifact("normal_note.md") is False
    assert is_editor_artifact("src/main.py") is False
    assert is_editor_artifact("docs/README.md") is False


def test_discover_files_excludes_artifacts(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()

    (root / "valid.md").write_text("# Valid\nReal content", encoding="utf-8")
    (root / ".valid.md.swp").write_text("swap data", encoding="utf-8")
    (root / ".valid.md.un~").write_text("undo data", encoding="utf-8")
    (root / "valid.md~").write_text("backup data", encoding="utf-8")
    (root / "draft.tmp").write_text("temp data", encoding="utf-8")
    (root / ".goutputstream-1234").write_text("atomic data", encoding="utf-8")

    obsidian_dir = root / ".obsidian"
    obsidian_dir.mkdir()
    (obsidian_dir / "app.json").write_text("{}", encoding="utf-8")

    settings = Settings(root_dir=root, data_dir=tmp_path / ".fsrag")
    files = discover_files(settings)
    rel_paths = [f.rel_path for f in files]

    assert rel_paths == ["valid.md"]
