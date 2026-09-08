"""Targeted file patching with sandbox enforcement and automatic incremental reindexing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import (
    ambiguous_match_error,
    file_not_found_error,
    file_read_error,
    invalid_parameter_error,
    path_traversal_error,
    target_not_found_error,
    write_error,
)
from .security import PathSecurityError, safe_resolve


def patch_file(
    root: Path,
    rel_path: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Safely find and replace unique string in a sandboxed workspace file."""
    try:
        abs_path = safe_resolve(root, rel_path)
    except PathSecurityError as exc:
        return path_traversal_error(rel_path, str(root), str(exc))

    if not abs_path.exists():
        return file_not_found_error(rel_path, str(root))
    if not abs_path.is_file():
        return invalid_parameter_error(
            "rel_path",
            rel_path,
            "Path is not a regular file",
            "Target a regular text file to patch.",
        )

    if not old_string:
        return invalid_parameter_error(
            "old_string",
            old_string,
            "old_string cannot be empty",
            "Provide the exact text to replace.",
        )
    if old_string == new_string:
        return invalid_parameter_error(
            "new_string",
            new_string,
            "new_string must differ from old_string",
            "Provide distinct replacement text.",
        )

    try:
        content = abs_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return file_read_error(rel_path, str(exc))

    occurrences = content.count(old_string)
    if occurrences == 0:
        return target_not_found_error(rel_path, old_string)

    if not replace_all and occurrences > 1:
        return ambiguous_match_error(rel_path, occurrences)

    count = occurrences if replace_all else 1
    new_content = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "rel_path": rel_path,
            "occurrences": occurrences,
            "replacements": count,
        }

    try:
        abs_path.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        return write_error(rel_path, str(exc))

    return {
        "success": True,
        "dry_run": False,
        "rel_path": rel_path,
        "occurrences": occurrences,
        "replacements": count,
    }
