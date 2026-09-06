"""Safe git history, log, and blame inspection over local git repositories."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from .errors import invalid_parameter_error
from .security import safe_resolve


def git_search(
    root: Path,
    mode: str = "commits",
    *,
    query: str | None = None,
    rel_path: str | None = None,
    limit: int = 20,
    line_start: int | None = None,
    line_end: int | None = None,
) -> dict[str, Any]:
    """Inspect local git repository history without network calls."""
    git_dir = root / ".git"
    if not git_dir.exists():
        return {
            "success": False,
            "error": "NOT_A_GIT_REPOSITORY",
            "message": f"Path '{root}' does not contain a .git directory.",
        }

    valid_modes = {"commits", "diff", "blame", "recent_changes"}
    if mode not in valid_modes:
        return invalid_parameter_error(
            "mode",
            mode,
            f"Mode must be one of: {', '.join(sorted(valid_modes))}",
            "Select mode='commits', 'recent_changes', 'diff', or 'blame'.",
        )

    resolved_path: Path | None = None
    if rel_path:
        resolved_path = safe_resolve(root, rel_path)

    cmd = ["git", "-C", str(root)]

    if mode in ("commits", "recent_changes"):
        cmd.extend(["log", f"-n{max(1, min(limit, 100))}", "--pretty=format:%H|%an|%ad|%s", "--date=short"])
        if query:
            cmd.extend(["--grep", query])
        if resolved_path:
            cmd.extend(["--", str(resolved_path.relative_to(root))])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            entries = []
            for l in lines:
                parts = l.split("|", 3)
                if len(parts) == 4:
                    entries.append(
                        {
                            "commit": parts[0],
                            "author": parts[1],
                            "date": parts[2],
                            "subject": parts[3],
                        }
                    )
            return {
                "success": True,
                "mode": mode,
                "count": len(entries),
                "commits": entries,
            }
        except subprocess.SubprocessError as exc:
            return {"success": False, "error": str(exc)}

    elif mode == "diff":
        # Show commit diff or working tree diff
        commit_target = query if query and len(query) in (7, 8, 40) and not query.startswith("-") else "HEAD~1..HEAD"
        cmd.extend(["diff", commit_target])
        if resolved_path:
            cmd.extend(["--", str(resolved_path.relative_to(root))])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            diff_text = res.stdout[:50000]
            return {
                "success": True,
                "mode": mode,
                "target": commit_target,
                "diff": diff_text,
                "truncated": len(res.stdout) > 50000,
            }
        except subprocess.SubprocessError as exc:
            return {"success": False, "error": str(exc)}

    elif mode == "blame":
        if not resolved_path or not resolved_path.exists():
            return invalid_parameter_error(
                "rel_path",
                rel_path or "",
                "A valid existing rel_path is required for blame mode.",
                "Provide a relative path to a tracked file in the repository.",
            )
        cmd.extend(["blame", "--line-porcelain"])
        if line_start is not None and line_end is not None:
            cmd.extend([f"-L{line_start},{line_end}"])
        cmd.append(str(resolved_path.relative_to(root)))

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            # Parse porcelain blame
            blame_lines = []
            current_commit: dict[str, Any] = {}
            for line in res.stdout.splitlines():
                if not line.startswith("\t"):
                    parts = line.split(" ", 1)
                    if len(parts[0]) == 40:
                        current_commit["commit"] = parts[0]
                    elif parts[0] == "author":
                        current_commit["author"] = parts[1]
                    elif parts[0] == "summary":
                        current_commit["summary"] = parts[1]
                else:
                    blame_lines.append(
                        {
                            "line": line[1:],
                            "commit": current_commit.get("commit", "")[:8],
                            "author": current_commit.get("author", ""),
                            "summary": current_commit.get("summary", ""),
                        }
                    )
            return {
                "success": True,
                "mode": mode,
                "rel_path": str(resolved_path.relative_to(root)),
                "total_lines": len(blame_lines),
                "blame": blame_lines[:limit],
            }
        except subprocess.SubprocessError as exc:
            return {"success": False, "error": str(exc)}

    return {"success": False, "error": f"Unsupported mode: {mode}"}
