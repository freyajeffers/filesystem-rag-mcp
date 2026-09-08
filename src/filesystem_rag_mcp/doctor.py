"""System diagnostics and health check for filesystem-rag-mcp."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings


class DiagnosticCheck(BaseModel):
    """A single health-check result emitted by `run_diagnostics()`.

    Returned by ~15 individual check functions and serialized into the
    `doctor --json` output. Promoted to BaseModel so the JSON Schema is
    auto-derivable and field metadata travels with the type (rather
    than living only in a docstring).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    name: str = Field(description="Short label, e.g. 'Python Version'")
    category: str = Field(
        description="Bucket the check belongs to: Environment, Filesystem, Dependencies, etc."
    )
    passed: bool = Field(description="True if the check succeeded")
    details: str = Field(description="Human-readable explanation of the result")
    remediation: str | None = Field(
        default=None,
        description="Actionable advice for the user when the check fails",
    )


def check_python_version() -> DiagnosticCheck:
    v = sys.version_info
    passed = v >= (3, 11)
    details = f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"
    remediation = (
        "Python >= 3.11 is required. Please install Python 3.11 or newer." if not passed else None
    )
    return DiagnosticCheck(
        name="Python Version",
        category="Environment",
        passed=passed,
        details=details,
        remediation=remediation,
    )


def check_root_directory(root_dir: Path) -> DiagnosticCheck:
    resolved = root_dir.resolve()
    if not resolved.exists():
        return DiagnosticCheck(
            name="Root Directory",
            category="Filesystem",
            passed=False,
            details=f"Path '{resolved}' does not exist",
            remediation=f"Create the directory or specify a valid path: mkdir -p '{resolved}'",
        )
    if not resolved.is_dir():
        return DiagnosticCheck(
            name="Root Directory",
            category="Filesystem",
            passed=False,
            details=f"Path '{resolved}' is not a directory",
            remediation="Provide a valid directory path to index.",
        )
    if not os.access(resolved, os.R_OK):
        return DiagnosticCheck(
            name="Root Directory",
            category="Filesystem",
            passed=False,
            details=f"Directory '{resolved}' is not readable",
            remediation=f"Grant read permissions: chmod +r '{resolved}'",
        )
    try:
        sample_count = sum(1 for _ in resolved.iterdir())
        details = f"'{resolved}' (readable, contains ~{sample_count} top-level entries)"
        passed = True
    except Exception as exc:
        details = f"'{resolved}' readable with error during scan: {exc}"
        passed = False

    return DiagnosticCheck(
        name="Root Directory",
        category="Filesystem",
        passed=passed,
        details=details,
    )


def check_data_directory(data_dir: Path) -> DiagnosticCheck:
    resolved = data_dir.resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        test_file = resolved / ".health_check_probe"
        test_file.write_text("probe", encoding="utf-8")
        test_file.unlink()
        details = f"'{resolved}' (writable storage path)"
        passed = True
        remediation = None
    except Exception as exc:
        details = f"Cannot write to '{resolved}': {exc}"
        passed = False
        remediation = f"Ensure write permissions for data directory: chmod +w '{resolved}'"

    return DiagnosticCheck(
        name="Data Directory",
        category="Filesystem",
        passed=passed,
        details=details,
        remediation=remediation,
    )


def check_core_dependencies() -> list[DiagnosticCheck]:
    core_modules = [
        ("mcp", "Model Context Protocol SDK"),
        ("pydantic", "Schema validation"),
        ("structlog", "Structured logging"),
        ("whoosh", "BM25 Full-text search"),
        ("chromadb", "Embedded Vector database"),
        ("sentence_transformers", "Dense neural embeddings"),
        ("starlette", "Streamable HTTP ASGI framework"),
        ("uvicorn", "ASGI web server"),
        ("jwt", "OAuth 2.1 JWT tokens (PyJWT)"),
        ("authlib", "OAuth 2.1 authorization engine"),
    ]
    checks = []
    for mod_name, desc in core_modules:
        spec = importlib.util.find_spec(mod_name)
        passed = spec is not None
        details = f"{desc} ({mod_name})" if passed else f"Missing required module '{mod_name}'"
        remediation = f"Run: uv pip install -e . or pip install {mod_name}" if not passed else None
        checks.append(
            DiagnosticCheck(
                name=f"Core Dependency: {mod_name}",
                category="Dependencies",
                passed=passed,
                details=details,
                remediation=remediation,
            )
        )
    return checks


def check_format_converters() -> list[DiagnosticCheck]:
    converters = [
        ("pymupdf", "PDF / EPUB conversion"),
        ("docx", "Word documents (.docx)"),
        ("pptx", "PowerPoint presentations (.pptx)"),
        ("openpyxl", "Excel spreadsheets (.xlsx)"),
        ("nbformat", "Jupyter Notebooks (.ipynb)"),
        ("striprtf", "Rich Text Format (.rtf)"),
        ("flashrank", "Neural cross-encoder reranking"),
        ("markitdown", "Universal Microsoft MarkItDown engine"),
        ("magic", "System MIME/Magic detection (libmagic)"),
    ]
    checks = []
    for mod_name, desc in converters:
        spec = importlib.util.find_spec(mod_name)
        passed = spec is not None
        details = (
            f"{desc} available" if passed else f"Optional converter '{mod_name}' not installed"
        )
        remediation = f"Install with: uv pip install {mod_name}" if not passed else None
        checks.append(
            DiagnosticCheck(
                name=f"Format Converter: {mod_name}",
                category="Optional Converters",
                passed=passed,
                details=details,
                remediation=remediation,
            )
        )
    return checks


def check_system_tools() -> list[DiagnosticCheck]:
    tools = [
        ("git", "Git version control CLI", False),
    ]
    checks = []
    for tool_name, desc, required in tools:
        path = shutil.which(tool_name)
        passed = path is not None
        details = f"{desc} found at {path}" if passed else f"CLI '{tool_name}' not found in PATH"
        remediation = (
            f"Install {tool_name} from your system package manager"
            if not passed and required
            else None
        )
        checks.append(
            DiagnosticCheck(
                name=f"System Tool: {tool_name}",
                category="System",
                passed=passed,
                details=details,
                remediation=remediation,
            )
        )
    return checks


def run_diagnostics(settings: Settings | None = None) -> tuple[bool, list[DiagnosticCheck]]:
    """Execute all diagnostic checks and return (all_passed, checks)."""
    if settings is None:
        settings = Settings()

    checks: list[DiagnosticCheck] = []
    checks.append(check_python_version())
    checks.append(check_root_directory(settings.root_dir))
    checks.append(check_data_directory(settings.data_dir))
    checks.extend(check_core_dependencies())
    checks.extend(check_format_converters())
    checks.extend(check_system_tools())

    all_passed = all(
        c.passed for c in checks if c.category in ("Environment", "Filesystem", "Dependencies")
    )
    return all_passed, checks


def print_doctor_report(settings: Settings | None = None, as_json: bool = False) -> int:
    """Print formatted diagnostic report and return exit code (0 = healthy, 1 = issue)."""
    all_passed, checks = run_diagnostics(settings)

    if as_json:
        import json

        data = {
            "healthy": all_passed,
            "status": "HEALTHY" if all_passed else "WARNING",
            # DiagnosticCheck is now a Pydantic BaseModel; use its native
            # serializer instead of dataclasses.asdict.
            "checks": [c.model_dump() for c in checks],
        }
        print(json.dumps(data, indent=2))
        return 0 if all_passed else 1

    print("=" * 70)
    print("  filesystem-rag-mcp System Diagnostics (Doctor)")
    print("=" * 70)

    categories: dict[str, list[DiagnosticCheck]] = {}
    for c in checks:
        categories.setdefault(c.category, []).append(c)

    for cat_name, cat_checks in categories.items():
        print(f"\n[{cat_name}]")
        for check in cat_checks:
            mark = "  ✓" if check.passed else "  ✗"
            print(f"{mark} {check.name}: {check.details}")
            if not check.passed and check.remediation:
                print(f"      → Suggestion: {check.remediation}")

    print("\n" + "-" * 70)
    if all_passed:
        print("  Status: HEALTHY — all essential services and dependencies are ready!")
        print("=" * 70)
        return 0
    else:
        print("  Status: WARNING — one or more required components need attention.")
        print("=" * 70)
        return 1


# Alias for backward-compatibility
run_doctor = run_diagnostics
