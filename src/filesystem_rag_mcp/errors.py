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
