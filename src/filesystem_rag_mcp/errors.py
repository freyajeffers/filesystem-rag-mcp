"""Structured, machine-readable, agent-friendly error representations.

When AI agents interact with MCP tools, bare error strings like "file not found"
force agents to guess what happened. This module standardizes errors with:
- error_code: Unique, machine-parsable snake_case string (e.g., 'path_traversal_blocked')
- message: Concise human/agent explanation
- context: Structured dictionary with relevant paths, allowed roots, file types, etc.
- suggested_fix: Explicit actionable remedy the agent can take immediately
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class AgentError(BaseModel):
    """Structured error envelope returned by every MCP tool on failure.

    The model is frozen so error envelopes are immutable once constructed
    (matching the previous `@dataclass(slots=True, frozen=True)` semantics).
    `model_dump()` replaces the old `to_dict()` helper.
    """

    # Pydantic's `model_config` is class-level; declare it via ClassVar so
    # strict static type checkers (mypy, pyright) don't flag the override.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    code: str = Field(
        description="Machine-parseable snake_case error code (e.g., 'PATH_TRAVERSAL_BLOCKED')"
    )
    message: str = Field(description="Concise human/agent explanation")
    suggested_fix: str = Field(description="Actionable remedy the agent can take immediately")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured context: paths, allowed roots, detected types, etc.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Backward-compatible alias for the previous dataclass API.

        New code should prefer the Pydantic-native `model_dump()`. This shim
        is kept so existing call sites in server.py and friends continue to
        work without churn and the public error contract is preserved.
        """
        return {
            "success": False,
            "error": self.model_dump(),
        }


def path_traversal_error(path: str, root: str, reason: str) -> dict[str, Any]:
    return AgentError(
        code="PATH_TRAVERSAL_BLOCKED",
        message=f"Access to path '{path}' was blocked for security reasons.",
        suggested_fix="Ensure the requested path is inside the allowed root directory and does not contain '..' components or escaping symlinks.",
        details={"path": path, "allowed_root": root, "reason": reason},
    ).to_dict()


def file_not_found_error(path: str, root: str) -> dict[str, Any]:
    return AgentError(
        code="FILE_NOT_FOUND",
        message=f"The requested file '{path}' does not exist.",
        suggested_fix="Verify the relative path. Use search or list resources to find valid file paths within the workspace root.",
        details={"path": path, "allowed_root": root},
    ).to_dict()


def not_a_file_error(path: str) -> dict[str, Any]:
    return AgentError(
        code="NOT_A_REGULAR_FILE",
        message=f"Path '{path}' is a directory or special file, not a regular file.",
        suggested_fix="Specify a path to a regular file rather than a directory or device.",
        details={"path": path},
    ).to_dict()


def file_read_error(path: str, reason: str) -> dict[str, Any]:
    return AgentError(
        code="FILE_READ_FAILED",
        message=f"Failed to read file '{path}'.",
        suggested_fix="Check file permissions, encoding, or whether the file is locked by another process.",
        details={"path": path, "os_error": reason},
    ).to_dict()


def file_not_indexable_error(
    path: str, reason: str, suggested_fix: str | None = None
) -> dict[str, Any]:
    """Returned when `read_file`/`list_directory`/etc. encounter a file the
    indexer was configured to skip (binary, oversized, extension-blacklisted)."""
    return AgentError(
        code="FILE_NOT_INDEXABLE",
        message=f"File '{path}' was skipped: {reason}",
        suggested_fix=suggested_fix
        or (
            "Adjust settings.index_binary_files / index_ignore_globs to allow this "
            "file, or use download_file_raw to inspect it as bytes."
        ),
        details={"path": path, "reason": reason},
    ).to_dict()


def conversion_error(path: str, detected_type: str, reason: str) -> dict[str, Any]:
    return AgentError(
        code="CONVERSION_FAILED",
        message=f"Failed to convert file '{path}' (detected as '{detected_type}') to Markdown.",
        suggested_fix="If the file is corrupted, check its source. Alternatively, use 'download_file_raw' to inspect the raw bytes.",
        details={"path": path, "detected_type": detected_type, "reason": reason},
    ).to_dict()


def chunk_not_found_error(chunk_id: str) -> dict[str, Any]:
    return AgentError(
        code="CHUNK_NOT_FOUND",
        message=f"Chunk with ID '{chunk_id}' was not found in the index.",
        suggested_fix="Perform a search query first to obtain fresh chunk IDs, or invoke 'refresh_index' if documents recently changed.",
        details={"chunk_id": chunk_id},
    ).to_dict()


def search_error(query: str, mode: str, reason: str) -> dict[str, Any]:
    return AgentError(
        code="SEARCH_EXECUTION_FAILED",
        message=f"Failed to execute {mode} search for query '{query}'.",
        suggested_fix="Ensure query is not malformed. If the index is uninitialized or corrupted, invoke 'refresh_index'.",
        details={"query": query, "mode": mode, "reason": reason},
    ).to_dict()


def git_error(command: str, reason: str, suggested_fix: str | None = None) -> dict[str, Any]:
    return AgentError(
        code="GIT_OPERATION_FAILED",
        message=f"Git command failed: {reason}",
        suggested_fix=suggested_fix or "Verify repository status and parameters.",
        details={"command": command, "reason": reason},
    ).to_dict()


def invalid_parameter_error(
    param: str, value: Any, constraint: str, suggested_fix: str
) -> dict[str, Any]:
    return AgentError(
        code="INVALID_PARAMETER",
        message=f"Invalid value {value!r} for parameter '{param}': {constraint}.",
        suggested_fix=suggested_fix,
        details={"parameter": param, "value": value, "constraint": constraint},
    ).to_dict()


def fetch_error(format: str, operation: str, reason: str) -> dict[str, Any]:
    """Error envelope for targeted data fetching (SQLite/JSON/CSV/line).

    `format` is the file format being queried (e.g. 'SQLite', 'JSON',
    'CSV', 'line range'); `operation` is what was attempted (the SQL
    query, JSONPath, regex, etc.); `reason` is the underlying exception
    text. Wrapping all fetcher errors in one shape lets the agent
    distinguish "your query was malformed" from "the file is unreadable"
    without parsing free-form strings.
    """
    return AgentError(
        code="FETCH_FAILED",
        message=f"{format} {operation} failed: {reason}",
        suggested_fix=(
            "Verify the query syntax for this format, confirm the file is not "
            "corrupted or locked, and retry. For SQLite, the query must be a "
            "read-only SELECT. For JSON, check the JSONPath syntax. For CSV, "
            "ensure columns are comma-separated and the row limit is positive."
        ),
        details={"format": format, "operation": operation, "reason": reason},
    ).to_dict()


def directory_not_found_error(sub_dir: str, root: str | None = None) -> dict[str, Any]:
    """Returned when a graph/symbol/grep query asks for a sub-directory
    that doesn't exist or escapes the indexed root."""
    details: dict[str, Any] = {"sub_dir": sub_dir}
    if root is not None:
        details["root"] = root
    return AgentError(
        code="DIRECTORY_NOT_FOUND",
        message=f"Directory '{sub_dir}' was not found in the indexed workspace.",
        suggested_fix=(
            "Pass an empty sub_dir for the workspace root, or a relative path "
            "that exists within it. Use list_directory to discover valid paths."
        ),
        details=details,
    ).to_dict()


def target_not_found_error(rel_path: str, target: str, hint: str = "") -> dict[str, Any]:
    """Returned by patch_file when the requested `old_string` does not
    appear in the target file."""
    details: dict[str, Any] = {"rel_path": rel_path, "target_length": len(target)}
    if hint:
        details["hint"] = hint
    return AgentError(
        code="TARGET_NOT_FOUND",
        message=f"Target string was not found in '{rel_path}'.",
        suggested_fix=(
            "Verify exact whitespace, indentation, and surrounding lines. "
            "Use read_file to fetch the current contents and copy the target "
            "string verbatim (or pass replace_all=True if you want every "
            "occurrence replaced)."
        ),
        details=details,
    ).to_dict()


def ambiguous_match_error(rel_path: str, occurrences: int) -> dict[str, Any]:
    """Returned by patch_file when a non-replace_all patch matches
    more than one location."""
    return AgentError(
        code="AMBIGUOUS_MATCH",
        message=(
            f"Target string occurs {occurrences} times in '{rel_path}'. "
            "patch_file requires a unique match unless replace_all=True."
        ),
        suggested_fix=(
            "Add surrounding context lines (a few before and after) to make "
            "the match unique, or pass replace_all=True if every occurrence "
            "should be replaced."
        ),
        details={"rel_path": rel_path, "occurrences": occurrences},
    ).to_dict()


def write_error(rel_path: str, reason: str) -> dict[str, Any]:
    """Returned by patch_file when the post-edit write fails."""
    return AgentError(
        code="WRITE_ERROR",
        message=f"Failed to write changes to '{rel_path}': {reason}",
        suggested_fix=(
            "Check filesystem permissions, available disk space, and that the "
            "file is not locked by another process. The original file is "
            "untouched."
        ),
        details={"rel_path": rel_path, "os_error": reason},
    ).to_dict()
