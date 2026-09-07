"""Targeted file patching with sandbox enforcement and automatic incremental reindexing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import file_not_found_error, invalid_parameter_error, path_traversal_error
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
        return {
            "success": False,
            "code": "FILE_READ_ERROR",
            "message": f"Failed to read file: {exc}",
            "details": {"rel_path": rel_path},
        }

    occurrences = content.count(old_string)
    if occurrences == 0:
        return {
            "success": False,
            "code": "TARGET_NOT_FOUND",
            "message": f"Target string not found in '{rel_path}'",
            "suggested_fix": "Verify exact whitespace, indentation, and surrounding lines.",
            "details": {"rel_path": rel_path, "occurrences": 0},
        }

    if not replace_all and occurrences > 1:
        return {
            "success": False,
            "code": "AMBIGUOUS_MATCH",
            "message": f"Target string occurs {occurrences} times in '{rel_path}'. Unique match required.",
            "suggested_fix": "Add surrounding context lines to make the match unique, or pass replace_all=True.",
            "details": {"rel_path": rel_path, "occurrences": occurrences},
        }

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
        return {
            "success": False,
            "code": "WRITE_ERROR",
            "message": f"Failed to write changes to '{rel_path}': {exc}",
            "details": {"rel_path": rel_path},
        }

    return {
        "success": True,
        "dry_run": False,
        "rel_path": rel_path,
        "occurrences": occurrences,
        "replacements": count,
    }
