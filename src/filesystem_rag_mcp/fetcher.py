"""Targeted data fetcher for structured and semi-structured files.

Allows agents to execute precise targeted queries, lookups, and filters against:
1. SQLite: Raw SQL queries (read-only enforced) with parameters, column names, row limits.
2. JSON / JSONL: JSONPath-like key queries, array indexing, and field filters.
3. CSV / TSV: Header-based column filtering, row range slicing, and WHERE-style row value matches.
4. Line/Range queries: Specific line numbers or char ranges from any text/markdown file.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from filesystem_rag_mcp.errors import fetch_error, invalid_parameter_error
from filesystem_rag_mcp.logging_setup import get_logger

log = get_logger("fetcher")


def fetch_sqlite_query(
    path: Path,
    sql: str,
    params: list[Any] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Execute a read-only query on a local SQLite database."""
    # Ensure read-only
    normalized = sql.strip().lower()
    dangerous_keywords = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "vacuum",
        "attach",
        "detach",
    )
    if any(normalized.startswith(kw) or f" {kw} " in normalized for kw in dangerous_keywords):
        return invalid_parameter_error(
            "sql",
            sql[:80] + ("..." if len(sql) > 80 else ""),
            "Only read-only SELECT queries are permitted for targeted SQLite fetching",
            "Rewrite the query as a single SELECT statement; INSERT/UPDATE/DELETE/ATTACH/PRAGMA/vacuum are blocked.",
        )

    try:
        conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql, params or [])
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(limit)
        conn.close()

        dict_rows = [dict(zip(col_names, r)) for r in rows]
        return {
            "success": True,
            "columns": col_names,
            "rows_returned": len(rows),
            "rows": dict_rows,
        }
    except Exception as exc:
        return fetch_error("SQLite", f"query {sql!r}", str(exc))


def _resolve_json_path(data: Any, path_expr: str) -> Any:
    """Evaluate simple dot-and-bracket path expressions, e.g. 'users[0].name' or 'data.items'."""
    if not path_expr or path_expr == "$":
        return data

    tokens = path_expr.lstrip("$.").replace("[", ".").replace("]", "").split(".")
    curr = data
    for token in tokens:
        if not token:
            continue
        if isinstance(curr, dict):
            if token in curr:
                curr = curr[token]
            else:
                return None
        elif isinstance(curr, list):
            try:
                idx = int(token)
                curr = curr[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return curr


def fetch_json_data(
    path: Path,
    path_expr: str | None = None,
    max_items: int = 100,
) -> dict[str, Any]:
    """Query a JSON or JSONL file with optional path expressions."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return {"success": True, "result": None}

        # Handle JSONL
        if "\n" in content and not content.startswith("["):
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            items = []
            for line in lines[:max_items]:
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
            parsed_data = items
        else:
            parsed_data = json.loads(content)

        extracted = _resolve_json_path(parsed_data, path_expr) if path_expr else parsed_data

        # Limit large list responses
        if isinstance(extracted, list) and len(extracted) > max_items:
            extracted = extracted[:max_items]
            truncated = True
        else:
            truncated = False

        return {
            "success": True,
            "path_expr": path_expr,
            "result": extracted,
            "truncated": truncated,
        }
    except Exception as exc:
        return fetch_error("JSON", f"path {path_expr!r}", str(exc))


def fetch_csv_data(
    path: Path,
    columns: list[str] | None = None,
    row_offset: int = 0,
    row_limit: int = 50,
    filter_col: str | None = None,
    filter_value: str | None = None,
) -> dict[str, Any]:
    """Targeted slicing and column/value filtering of CSV and TSV files."""
    try:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            all_rows = list(reader)

        if not all_rows:
            return {"success": True, "columns": [], "rows": [], "total_rows": 0}

        headers = all_rows[0]
        data_rows = all_rows[1:]

        # Select column indices
        col_indices = []
        out_headers = []
        if columns:
            for c in columns:
                if c in headers:
                    col_indices.append(headers.index(c))
                    out_headers.append(c)
        else:
            col_indices = list(range(len(headers)))
            out_headers = headers

        # Filter by value if specified
        if filter_col and filter_value is not None and filter_col in headers:
            filt_idx = headers.index(filter_col)
            filter_str = str(filter_value).lower()
            data_rows = [
                r for r in data_rows if filt_idx < len(r) and filter_str in r[filt_idx].lower()
            ]

        total_matching = len(data_rows)
        sliced_rows = data_rows[row_offset : row_offset + row_limit]

        # Extract selected columns
        results = []
        for r in sliced_rows:
            row_dict = {}
            for col_name, idx in zip(out_headers, col_indices):
                row_dict[col_name] = r[idx] if idx < len(r) else ""
            results.append(row_dict)

        return {
            "success": True,
            "columns": out_headers,
            "row_offset": row_offset,
            "row_limit": row_limit,
            "total_matching": total_matching,
            "rows": results,
        }
    except Exception as exc:
        return fetch_error("CSV", f"filter {filter_col!r}={filter_value!r}", str(exc))


def fetch_lines(
    path: Path,
    start_line: int,
    end_line: int,
) -> dict[str, Any]:
    """Fetch specific line numbers from any plain text or source code file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)
        start_idx = max(1, start_line) - 1
        end_idx = min(total_lines, end_line)

        selected = lines[start_idx:end_idx]
        formatted = "\n".join(f"{start_idx + i + 1:5d} | {line}" for i, line in enumerate(selected))

        return {
            "success": True,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "total_lines": total_lines,
            "lines": formatted,
        }
    except Exception as exc:
        return fetch_error("line range", f"lines {start_line}-{end_line}", str(exc))
