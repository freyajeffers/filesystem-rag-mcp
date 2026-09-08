"""Corpus dependency and reference topology graph builder."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .errors import directory_not_found_error
from .security import is_indexable_file, safe_resolve


class CorpusGraphBuilder:
    """Extracts internal imports and Markdown links to construct a topology graph."""

    # Python import patterns
    PY_IMPORT_RE = re.compile(
        r"^(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.,\s]+))", re.MULTILINE
    )
    # JS/TS import patterns
    JS_IMPORT_RE = re.compile(
        r"""(?:import\s+.*?from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\))"""
    )
    # Markdown link pattern [title](relative_path.md)
    MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def build_graph(self, sub_dir: str = "", max_files: int = 500) -> dict[str, Any]:
        search_dir = safe_resolve(self.root, sub_dir)
        if not search_dir.exists():
            return directory_not_found_error(sub_dir, root=str(self.root))

        all_files: list[Path] = []
        for p in search_dir.rglob("*"):
            if not p.is_file():
                continue
            if any(part.startswith(".") for part in p.parts):
                continue
            if is_indexable_file(p, allow_binary=False):
                all_files.append(p)
            if len(all_files) >= max_files:
                break

        rel_paths_set = {str(p.relative_to(self.root)) for p in all_files}
        rel_to_path = {str(p.relative_to(self.root)): p for p in all_files}

        edges: list[dict[str, str]] = []
        in_degree: dict[str, int] = defaultdict(int)
        out_degree: dict[str, int] = defaultdict(int)
        dependencies: dict[str, list[str]] = defaultdict(list)

        for rel, path in rel_to_path.items():
            ext = path.suffix.lower()
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            targets: set[str] = set()

            if ext in (".py", ".pyi"):
                # Parse python imports
                for match in self.PY_IMPORT_RE.finditer(content):
                    from_mod, imp_mod = match.groups()
                    mod = from_mod or imp_mod
                    if not mod:
                        continue
                    # Check if resolves to any local relative module
                    parts = mod.strip().split()[0].replace(".", "/")
                    for candidate in (f"{parts}.py", f"{parts}/__init__.py"):
                        for existing in rel_paths_set:
                            if existing.endswith(candidate):
                                targets.add(existing)

            elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs"):
                # Parse JS/TS imports
                for match in self.JS_IMPORT_RE.finditer(content):
                    import_path = match.group(1) or match.group(2)
                    if import_path and (
                        import_path.startswith("./") or import_path.startswith("../")
                    ):
                        resolved_target = (path.parent / import_path).resolve()
                        try:
                            t_rel = str(resolved_target.relative_to(self.root))
                            for c_ext in (
                                "",
                                ".ts",
                                ".tsx",
                                ".js",
                                ".jsx",
                                "/index.ts",
                                "/index.js",
                            ):
                                if f"{t_rel}{c_ext}" in rel_paths_set:
                                    targets.add(f"{t_rel}{c_ext}")
                                    break
                        except ValueError:
                            pass

            elif ext in (".md", ".markdown"):
                # Parse markdown links
                for match in self.MD_LINK_RE.finditer(content):
                    link = match.group(2).split("#")[0].split("?")[0].strip()
                    if link and not link.startswith(("http://", "https://", "mailto:")):
                        resolved_target = (path.parent / link).resolve()
                        try:
                            t_rel = str(resolved_target.relative_to(self.root))
                            if t_rel in rel_paths_set:
                                targets.add(t_rel)
                        except ValueError:
                            pass

            for t in targets:
                if t != rel:
                    edges.append({"source": rel, "target": t})
                    out_degree[rel] += 1
                    in_degree[t] += 1
                    dependencies[rel].append(t)

        # Architectural hubs: highest in-degree (most imported/linked files)
        hubs = sorted(
            [
                {"file": f, "imported_by_count": in_degree[f]}
                for f in rel_paths_set
                if in_degree[f] > 0
            ],
            key=lambda x: in_degree[str(x["file"])],
            reverse=True,
        )

        orphans = [f for f in rel_paths_set if in_degree[f] == 0 and out_degree[f] == 0]

        return {
            "success": True,
            "total_files": len(rel_paths_set),
            "total_dependencies": len(edges),
            "hubs": hubs[:15],
            "orphan_count": len(orphans),
            "dependencies": dict(dependencies),
        }
