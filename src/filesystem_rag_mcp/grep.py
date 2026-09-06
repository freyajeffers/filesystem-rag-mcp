"""Fast, sandboxed regex and substring search over workspace files."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePath
from typing import Any

from .errors import invalid_parameter_error
from .security import is_indexable_file, safe_resolve


def grep_search(
    root: Path,
    pattern: str,
    *,
    path_glob: str | None = None,
    max_matches: int = 100,
    case_sensitive: bool = False,
    context_lines: int = 2,
    sub_dir: str = "",
) -> dict[str, Any]:
    """Search for a regex or exact string pattern across files in root."""
    if not pattern:
        return invalid_parameter_error(
            "pattern",
            pattern,
            "Pattern cannot be empty",
            "Provide a regex or substring to search across workspace files.",
        )

    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags=flags)
    except re.error as exc:
        return invalid_parameter_error(
            "pattern",
            pattern,
            f"Invalid regular expression: {exc}",
            "Ensure the pattern is valid regex syntax or escape special characters.",
        )

    search_dir = safe_resolve(root, sub_dir)
    if not search_dir.exists() or not search_dir.is_dir():
        return {
            "success": True,
            "pattern": pattern,
            "total_matches": 0,
            "matches": [],
            "files_searched": 0,
        }

    glob_pat = path_glob.strip().lstrip("/") if path_glob else None

    def _matches_glob(p: str) -> bool:
        if not glob_pat:
            return True
        cleaned = p.lstrip("/")
        if PurePath(cleaned).match(glob_pat) or fnmatch.fnmatch(cleaned, glob_pat):
            return True
        if "**" in glob_pat:
            reg = re.escape(glob_pat).replace(r"\*\*/", "(.+/)?").replace(r"\*", "[^/]*")
            if re.fullmatch(reg, cleaned):
                return True
        return False

    matches: list[dict[str, Any]] = []
    files_searched = 0

    for path in search_dir.rglob("*"):
        if not path.is_file():
            continue
        # Skip internal and hidden directories
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            continue

        if not _matches_glob(rel):
            continue
        if not is_indexable_file(path, allow_binary=False):
            continue

        files_searched += 1

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lines = content.splitlines()
        for idx, line in enumerate(lines):
            if compiled.search(line):
                start_c = max(0, idx - context_lines)
                end_c = min(len(lines), idx + context_lines + 1)
                context_block = [
                    {"line_number": i + 1, "text": lines[i], "is_match": i == idx}
                    for i in range(start_c, end_c)
                ]
                matches.append(
                    {
                        "rel_path": rel,
                        "line_number": idx + 1,
                        "line_text": line,
                        "context": context_block,
                    }
                )
                if len(matches) >= max_matches:
                    break

        if len(matches) >= max_matches:
            break

    return {
        "success": True,
        "pattern": pattern,
        "case_sensitive": case_sensitive,
        "total_matches": len(matches),
        "files_searched": files_searched,
        "truncated": len(matches) >= max_matches,
        "matches": matches,
    }
