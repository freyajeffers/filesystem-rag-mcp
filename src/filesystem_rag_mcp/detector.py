"""Content-based file type detection using deep inspection (Magika AI + magic bytes).

Inspects file content directly rather than relying solely on file extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filesystem_rag_mcp.logging_setup import get_logger

log = get_logger("detector")

_magika_instance: Any = None


def _get_magika():
    global _magika_instance
    if _magika_instance is None:
        try:
            import magika

            _magika_instance = magika.Magika()
        except Exception as err:
            log.warning("magika_init_failed", error=str(err))
            _magika_instance = False
    return _magika_instance if _magika_instance is not False else None


@dataclass(slots=True, frozen=True)
class FileTypeInfo:
    label: str
    mime_type: str
    group: str
    is_text: bool
    is_convertible: bool


CONVERTIBLE_LABELS: frozenset[str] = frozenset({
    "pdf",
    "docx",
    "doc",
    "pptx",
    "ppt",
    "xlsx",
    "xls",
    "epub",
    "ipynb",
    "html",
    "htm",
    "xml",
    "svg",
    "rtf",
    "csv",
    "tsv",
    "json",
    "jsonl",
    "yaml",
    "toml",
    "markdown",
    "txt",
    "rst",
    "latex",
})

CONVERTIBLE_GROUPS: frozenset[str] = frozenset({
    "document",
    "code",
    "text",
})


def detect_file_type(path: Path) -> FileTypeInfo:
    """Detect the true file type by inspecting raw content/bytes."""
    # Fast path: check if file is empty
    try:
        if path.stat().st_size == 0:
            return FileTypeInfo(
                label="empty",
                mime_type="inode/x-empty",
                group="text",
                is_text=True,
                is_convertible=True,
            )
    except OSError:
        pass

    # Read leading header bytes for magic bytes sniffing
    header = b""
    try:
        with path.open("rb") as f:
            header = f.read(4096)
    except OSError:
        pass

    # Magic byte signatures
    if header.startswith(b"%PDF-"):
        return FileTypeInfo(
            label="pdf",
            mime_type="application/pdf",
            group="document",
            is_text=False,
            is_convertible=True,
        )
    if header.startswith(b"{\\rtf"):
        return FileTypeInfo(
            label="rtf",
            mime_type="application/rtf",
            group="document",
            is_text=False,
            is_convertible=True,
        )
    if header.startswith(b"\x1f\x8b"):
        # gzip
        pass

    # Check zip-based formats (DOCX, PPTX, XLSX, EPUB)
    if header.startswith(b"PK\x03\x04"):
        # Inspect zip entries to identify exact format
        try:
            import zipfile

            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path, "r") as zf:
                    namelist = set(zf.namelist())
                    if "word/document.xml" in namelist:
                        return FileTypeInfo(
                            label="docx",
                            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            group="document",
                            is_text=False,
                            is_convertible=True,
                        )
                    if "ppt/presentation.xml" in namelist:
                        return FileTypeInfo(
                            label="pptx",
                            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            group="document",
                            is_text=False,
                            is_convertible=True,
                        )
                    if "xl/workbook.xml" in namelist:
                        return FileTypeInfo(
                            label="xlsx",
                            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            group="document",
                            is_text=False,
                            is_convertible=True,
                        )
                    if "mimetype" in namelist:
                        try:
                            mimetype_data = zf.read("mimetype").decode("utf-8", errors="ignore").strip()
                            if "application/epub+zip" in mimetype_data:
                                return FileTypeInfo(
                                    label="epub",
                                    mime_type="application/epub+zip",
                                    group="document",
                                    is_text=False,
                                    is_convertible=True,
                                )
                        except Exception:
                            pass
        except Exception:
            pass

    # Check for IPYNB JSON format
    if header.strip().startswith(b"{"):
        try:
            import json

            with path.open("r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                if isinstance(data, dict) and "cells" in data and "nbformat" in data:
                    return FileTypeInfo(
                        label="ipynb",
                        mime_type="application/x-ipynb+json",
                        group="document",
                        is_text=True,
                        is_convertible=True,
                    )
                if isinstance(data, (dict, list)):
                    return FileTypeInfo(
                        label="json",
                        mime_type="application/json",
                        group="code",
                        is_text=True,
                        is_convertible=True,
                    )
        except Exception:
            pass

    # Use Magika (deep content-type classification model)
    mg = _get_magika()
    if mg:
        try:
            res = mg.identify_path(path)
            label = res.output.label
            mime = res.output.mime_type
            group = res.output.group
            is_convertible = (
                label in CONVERTIBLE_LABELS
                or group in CONVERTIBLE_GROUPS
                or mime.startswith("text/")
            )
            is_text = group in ("text", "code") or mime.startswith("text/")
            return FileTypeInfo(
                label=label,
                mime_type=mime,
                group=group,
                is_text=is_text,
                is_convertible=is_convertible,
            )
        except Exception as e:
            log.warning("magika_identify_failed", path=str(path), error=str(e))

    # Text heuristic fallback: inspect byte distribution
    is_text = False
    try:
        decoded = header.decode("utf-8")
        control = sum(1 for c in decoded if ord(c) < 32 and c not in "\n\r\t\f\v")
        is_text = (control / max(len(decoded), 1)) < 0.05
    except UnicodeDecodeError:
        is_text = False

    return FileTypeInfo(
        label="text" if is_text else "binary",
        mime_type="text/plain" if is_text else "application/octet-stream",
        group="text" if is_text else "binary",
        is_text=is_text,
        is_convertible=is_text,
    )
