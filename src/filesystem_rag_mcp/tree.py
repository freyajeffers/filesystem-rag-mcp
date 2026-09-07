"""Directory structure and tree exploration for AI agents.

Provides sandboxed directory listing with depth controls, glob pattern filtering,
and rich metadata summaries (size, detected type, mime, is_directory).
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from .detector import detect_file_type
from .errors import path_traversal_error
from .security import PathSecurityError, safe_resolve


def list_directory(
    root: Path,
    rel_path: str = "",
    *,
    max_depth: int = 2,
    include_files: bool = True,
    include_dirs: bool = True,
    pattern: str | None = None,
    limit: int = 150,
) -> dict[str, Any]:
    """List directory contents up to max_depth."""
    try:
        target = safe_resolve(root, rel_path) if rel_path else root
    except PathSecurityError as exc:
        return path_traversal_error(rel_path, str(root), str(exc))

    if not target.exists():
        return {
            "success": False,
            "error": f"Path '{rel_path}' does not exist inside root workspace.",
        }

    if not target.is_dir():
        return {
            "success": False,
            "error": f"Path '{rel_path}' is a file, not a directory. Use read_file instead.",
        }

    entries: list[dict[str, Any]] = []

    def _scan(current: Path, depth: int) -> None:
        if depth > max_depth or len(entries) >= limit:
            return

        try:
            for item in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                # Ignore hidden directories like .git or .fsrag
                if item.name.startswith("."):
                    continue

                rel_item = str(item.relative_to(root))
                is_dir = item.is_dir()

                # Check pattern filter
                if pattern and not (
                    fnmatch.fnmatch(item.name, pattern) or fnmatch.fnmatch(rel_item, pattern)
                ):
                    if is_dir:
                        _scan(item, depth + 1)
                    continue

                if is_dir and include_dirs:
                    entries.append(
                        {
                            "name": item.name,
                            "rel_path": rel_item,
                            "type": "directory",
                            "depth": depth,
                        }
                    )
                elif not is_dir and include_files:
                    t_info = detect_file_type(item)
                    try:
                        sz = item.stat().st_size
                    except OSError:
                        sz = 0
                    entries.append(
                        {
                            "name": item.name,
                            "rel_path": rel_item,
                            "type": "file",
                            "size_bytes": sz,
                            "detected_label": t_info.label,
                            "mime_type": t_info.mime_type,
                            "is_convertible": t_info.is_convertible,
                            "depth": depth,
                        }
                    )

                if len(entries) >= limit:
                    break

                if is_dir:
                    _scan(item, depth + 1)
        except PermissionError:
            pass

    _scan(target, 1)

    return {
        "success": True,
        "root": str(root),
        "target_path": rel_path or ".",
        "total_entries": len(entries),
        "truncated": len(entries) >= limit,
        "entries": entries,
    }
