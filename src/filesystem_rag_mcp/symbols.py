"""Symbol and AST declaration extraction (functions, classes, methods, interfaces)."""

from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path, PurePath
from typing import Any

from .errors import file_not_found_error, invalid_parameter_error
from .security import safe_resolve


def search_symbols(
    root: Path,
    name: str = "",
    *,
    symbol_type: str = "all",
    path_glob: str | None = None,
    max_matches: int = 100,
    sub_dir: str = "",
) -> dict[str, Any]:
    """Search for class and function declarations across Python, JS/TS, and generic code."""
    valid_types = {"all", "function", "class", "method"}
    if symbol_type not in valid_types:
        return invalid_parameter_error(
            "symbol_type",
            symbol_type,
            f"symbol_type must be one of: {', '.join(sorted(valid_types))}",
            "Select symbol_type='all', 'function', 'class', or 'method'.",
        )

    search_dir = safe_resolve(root, sub_dir)
    if not search_dir.exists() or not search_dir.is_dir():
        return {
            "success": True,
            "name_filter": name,
            "total_matches": 0,
            "symbols": [],
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

    name_lower = name.strip().lower()
    symbols: list[dict[str, Any]] = []

    # JS/TS regex patterns for function/class
    JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z0-9_$]+)", re.MULTILINE)
    JS_FUNC_RE = re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)|^\s*(?:export\s+)?const\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
        re.MULTILINE,
    )

    for path in search_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            continue

        if not _matches_glob(rel):
            continue

        ext = path.suffix.lower()

        if ext in (".py", ".pyi"):
            try:
                code = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(code, filename=str(path))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if it's a method inside a class
                    # Heuristic: inspect line or context, or treat top-level vs nested
                    sym_name = node.name
                    if name_lower and name_lower not in sym_name.lower():
                        continue
                    if symbol_type not in ("all", "function", "method"):
                        continue

                    doc = ast.get_docstring(node) or ""
                    # Reconstruct basic arg names
                    args = [a.arg for a in node.args.args]
                    symbols.append(
                        {
                            "name": sym_name,
                            "type": "function",
                            "rel_path": rel,
                            "line_start": node.lineno,
                            "line_end": getattr(node, "end_lineno", node.lineno),
                            "parameters": args,
                            "docstring": doc.split("\n")[0] if doc else "",
                        }
                    )
                    if len(symbols) >= max_matches:
                        break

                elif isinstance(node, ast.ClassDef):
                    if symbol_type not in ("all", "class"):
                        continue
                    sym_name = node.name
                    if name_lower and name_lower not in sym_name.lower():
                        continue

                    doc = ast.get_docstring(node) or ""
                    bases = [getattr(b, "id", "") for b in node.bases if hasattr(b, "id")]
                    symbols.append(
                        {
                            "name": sym_name,
                            "type": "class",
                            "rel_path": rel,
                            "line_start": node.lineno,
                            "line_end": getattr(node, "end_lineno", node.lineno),
                            "bases": bases,
                            "docstring": doc.split("\n")[0] if doc else "",
                        }
                    )
                    if len(symbols) >= max_matches:
                        break

        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if symbol_type in ("all", "class"):
                    m_cls = JS_CLASS_RE.search(line)
                    if m_cls:
                        cname = m_cls.group(1)
                        if not name_lower or name_lower in cname.lower():
                            symbols.append(
                                {
                                    "name": cname,
                                    "type": "class",
                                    "rel_path": rel,
                                    "line_start": idx + 1,
                                    "line_end": idx + 1,
                                }
                            )

                if symbol_type in ("all", "function", "method"):
                    m_fn = JS_FUNC_RE.search(line)
                    if m_fn:
                        fname = m_fn.group(1) or m_fn.group(2)
                        if fname and (not name_lower or name_lower in fname.lower()):
                            symbols.append(
                                {
                                    "name": fname,
                                    "type": "function",
                                    "rel_path": rel,
                                    "line_start": idx + 1,
                                    "line_end": idx + 1,
                                }
                            )
                if len(symbols) >= max_matches:
                    break

        if len(symbols) >= max_matches:
            break

    return {
        "success": True,
        "name_filter": name,
        "symbol_type": symbol_type,
        "total_matches": len(symbols),
        "symbols": symbols,
    }


def find_symbol_references(
    root: Path,
    *,
    symbol_name: str,
    path_glob: str | None = None,
    max_matches: int = 100,
    sub_dir: str = "",
) -> dict[str, Any]:
    """Find call-sites and references of a symbol across workspace code files."""
    if not symbol_name or not symbol_name.strip():
        return invalid_parameter_error(
            "symbol_name",
            symbol_name,
            "symbol_name cannot be empty",
            "Provide a symbol name to locate.",
        )

    target_dir = root / sub_dir if sub_dir else root
    if not target_dir.exists():
        return file_not_found_error(sub_dir or ".", str(root))

    # Match symbol bounded by word boundaries
    pattern = re.compile(rf"\b{re.escape(symbol_name.strip())}\b")
    references: list[dict[str, Any]] = []

    # Common code file extensions to search
    code_extensions = {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
    }

    for path in target_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in code_extensions:
            continue
        try:
            rel_path = str(path.relative_to(root))
        except ValueError:
            continue

        if any(
            part.startswith(".") or part in ("node_modules", "dist", "build", "target")
            for part in path.parts
        ):
            continue

        if path_glob and not (
            PurePath(rel_path).match(path_glob) or fnmatch.fnmatch(rel_path, path_glob)
        ):
            continue

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        for idx, line in enumerate(lines, start=1):
            if pattern.search(line):
                # Classify usage line (heuristic: import vs invocation/mention)
                stripped = line.strip()
                kind = (
                    "import"
                    if stripped.startswith(("import ", "from ", "require("))
                    else "reference"
                )
                references.append(
                    {
                        "rel_path": rel_path,
                        "line": idx,
                        "kind": kind,
                        "snippet": stripped[:200],
                    }
                )
                if len(references) >= max_matches:
                    break
        if len(references) >= max_matches:
            break

    return {
        "success": True,
        "symbol_name": symbol_name,
        "total_references": len(references),
        "references": references,
    }
