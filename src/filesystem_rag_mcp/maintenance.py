"""Automated SQLite and storage maintenance primitives.

Provides compaction, segment optimization, and integrity checks across the
full-text index, vector store, and SQLite metadata stores.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .fulltext import FullTextStore
from .logging_setup import get_logger

log = get_logger("maintenance")


class MaintenanceReport(BaseModel):
    """Structured report returned after database and storage maintenance."""

    model_config = ConfigDict(frozen=True)

    ok: bool = Field(description="True if maintenance passed with zero errors")
    pre_bytes: int = Field(description="Total storage footprint in bytes before maintenance")
    post_bytes: int = Field(description="Total storage footprint in bytes after maintenance")
    reclaimed_bytes: int = Field(description="Total space reclaimed in bytes")
    duration_ms: float = Field(description="Elapsed execution time in milliseconds")
    integrity: str = Field(description="Status of storage integrity verification")
    details: dict[str, Any] = Field(default_factory=dict, description="Detailed subsystem metrics")


def _get_dir_size(path: Path) -> int:
    """Calculate recursive total byte size of a directory or file."""
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return total


def optimize_storage(settings: Settings) -> MaintenanceReport:
    """Execute complete storage optimization: FTS5 compaction, SQLite vacuuming, integrity checks."""
    start_time = time.perf_counter()
    pre_bytes = _get_dir_size(settings.data_dir)
    details: dict[str, Any] = {}
    errors: list[str] = []

    # 1. Whoosh FTS5 segment compaction
    try:
        ft = FullTextStore(settings)
        ft.optimize()
        details["fulltext_optimized"] = True
    except Exception as exc:
        details["fulltext_optimized"] = False
        errors.append(f"FullText optimization failed: {exc}")

    # 2. SQLite maintenance on any .sqlite / .db files under data_dir
    sqlite_files = list(settings.data_dir.rglob("*.sqlite")) + list(settings.data_dir.rglob("*.db"))
    details["sqlite_files_checked"] = len(sqlite_files)
    for db_file in sqlite_files:
        try:
            conn = sqlite3.connect(str(db_file), timeout=5.0)
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA incremental_vacuum;")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

            # Integrity check
            integrity_rows = conn.execute("PRAGMA integrity_check;").fetchall()
            conn.close()
            details[f"{db_file.name}_integrity"] = (
                integrity_rows[0][0] if integrity_rows else "unknown"
            )
        except Exception as exc:
            errors.append(f"SQLite maintenance on {db_file.name} failed: {exc}")

    post_bytes = _get_dir_size(settings.data_dir)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    reclaimed = max(0, pre_bytes - post_bytes)

    integrity_status = "ok" if not errors else f"warning: {'; '.join(errors)}"

    log.info(
        "storage_maintenance_complete",
        pre_bytes=pre_bytes,
        post_bytes=post_bytes,
        reclaimed_bytes=reclaimed,
        duration_ms=duration_ms,
        integrity=integrity_status,
    )

    return MaintenanceReport(
        ok=len(errors) == 0,
        pre_bytes=pre_bytes,
        post_bytes=post_bytes,
        reclaimed_bytes=reclaimed,
        duration_ms=round(duration_ms, 2),
        integrity=integrity_status,
        details=details,
    )
