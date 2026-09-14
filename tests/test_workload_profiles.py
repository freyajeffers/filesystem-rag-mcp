from pathlib import Path

from filesystem_rag_mcp.config import (
    PERSONAL_NOTES_PROFILE,
    Settings,
    get_profile_config,
)
from filesystem_rag_mcp.indexing import discover_files


def test_profile_resolution_and_defaults():
    cfg_notes = get_profile_config("notes")
    assert cfg_notes.name == "notes"
    assert cfg_notes.allowed_extensions == (".md", ".markdown")
    assert cfg_notes.enable_symbols is False
    assert cfg_notes.enable_git is False

    cfg_codebase = get_profile_config("codebase")
    assert cfg_codebase.name == "codebase"
    assert cfg_codebase.allowed_extensions is None
    assert cfg_codebase.enable_symbols is True
    assert cfg_codebase.enable_git is True


def test_settings_from_env_profile_default_root():
    # When profile is notes and no root_dir specified in env, defaults to notes root
    env = {"FSRAG_PROFILE": "notes"}
    s = Settings.from_env(env)
    assert s.profile == "notes"
    assert s.root_dir == PERSONAL_NOTES_PROFILE.default_root

    # Explicit root_dir overrides default notes root
    env_override = {"FSRAG_PROFILE": "notes", "FSRAG_ROOT_DIR": "/custom/notes"}
    s_override = Settings.from_env(env_override)
    assert s_override.profile == "notes"
    assert s_override.root_dir == Path("/custom/notes").resolve()


def test_profile_extension_filtering_in_discovery(tmp_path: Path):
    vault = tmp_path / "notes"
    vault.mkdir()
    (vault / "note.md").write_text("# Note 1\nContent", encoding="utf-8")
    (vault / "paper.markdown").write_text("# Paper\nContent", encoding="utf-8")
    (vault / "script.py").write_text("print('hello')", encoding="utf-8")
    (vault / "data.csv").write_text("a,b,c\n1,2,3", encoding="utf-8")

    # Notes profile: only .md and .markdown discovered
    s_notes = Settings(profile="notes", root_dir=vault, data_dir=tmp_path / ".fsrag")
    files_notes = discover_files(s_notes)
    rel_paths_notes = {f.rel_path for f in files_notes}
    assert "note.md" in rel_paths_notes
    assert "paper.markdown" in rel_paths_notes
    assert "script.py" not in rel_paths_notes
    assert "data.csv" not in rel_paths_notes

    # Codebase profile: discovers all text and convertible files
    s_code = Settings(profile="codebase", root_dir=vault, data_dir=tmp_path / ".fsrag")
    files_code = discover_files(s_code)
    rel_paths_code = {f.rel_path for f in files_code}
    assert "note.md" in rel_paths_code
    assert "paper.markdown" in rel_paths_code
    assert "script.py" in rel_paths_code
    assert "data.csv" in rel_paths_code
