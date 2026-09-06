"""Converter module for transforming rich document formats into Markdown.

Supports:
- Office: DOCX, PPTX, XLSX, XLS
- Documents: PDF, RTF, EPUB
- Web & Markup: HTML, HTM, XML, SVG
- Notebooks: IPYNB
- Data/Config: CSV, TSV, JSON, YAML, TOML, INI
- Code & Text: All plain text, Markdown, RST, TeX, source code files
- MarkItDown integration with robust format-specific fallbacks.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from filesystem_rag_mcp.logging_setup import get_logger

log = get_logger("converter")

# Lazy MarkItDown instance
_markitdown_instance: Any = None


def _get_markitdown():
    global _markitdown_instance
    if _markitdown_instance is None:
        try:
            from markitdown import MarkItDown

            _markitdown_instance = MarkItDown()
        except Exception as err:
            log.warning("markitdown_import_failed", error=str(err))
            _markitdown_instance = False
    return _markitdown_instance if _markitdown_instance is not False else None


def _convert_ipynb(path: Path) -> str:
    import nbformat

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        nb = nbformat.read(f, as_version=4)

    lines: list[str] = [f"# Notebook: {path.name}\n"]
    for i, cell in enumerate(nb.cells):
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", "").strip()
        if not source:
            continue

        if cell_type == "markdown":
            lines.append(f"\n{source}\n")
        elif cell_type == "code":
            lines.append(f"\n```python\n# [In {i + 1}]\n{source}\n```\n")
            # Include text outputs if present
            outputs = cell.get("outputs", [])
            for out in outputs:
                out_type = out.get("output_type", "")
                if out_type == "stream":
                    text = out.get("text", "").strip()
                    if text:
                        lines.append(f"```output\n{text}\n```\n")
                elif out_type in ("execute_result", "display_data"):
                    data = out.get("data", {})
                    if "text/plain" in data:
                        lines.append(f"```output\n{data['text/plain'].strip()}\n```\n")
    return "\n".join(lines)


def _convert_csv(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return f"# {path.name}\n\n*(Empty CSV file)*"

    lines = [f"# {path.name}\n"]
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:101]:  # limit first 100 rows in table
        # Pad row if columns don't match
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(cell.replace("\n", " ").strip() for cell in padded[: len(header)]) + " |")

    if len(rows) > 101:
        lines.append(f"\n*(Showing 100 of {len(rows)-1} rows)*")

    return "\n".join(lines)


def _convert_rtf(path: Path) -> str:
    from striprtf.striprtf import rtf_to_text

    raw = path.read_text(encoding="utf-8", errors="replace")
    text = rtf_to_text(raw)
    return f"# {path.name}\n\n{text}"


def _convert_pdf_fallback(path: Path) -> str:
    import pymupdf

    doc = pymupdf.open(str(path))
    pages_text: list[str] = [f"# {path.name}\n"]
    for i in range(len(doc)):
        page = doc[i]
        text = str(page.get_text())
        pages_text.append(f"\n## Page {i + 1}\n\n{text.strip()}\n")
    doc.close()
    return "\n".join(pages_text)


def _convert_docx_fallback(path: Path) -> str:
    import docx

    doc = docx.Document(str(path))
    lines = [f"# {path.name}\n"]
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            lines.append(f"{txt}\n")
    for table in doc.tables:
        lines.append("\n")
        for row in table.rows:
            lines.append("| " + " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells) + " |")
    return "\n".join(lines)


def _convert_pptx_fallback(path: Path) -> str:
    import pptx

    prs = pptx.Presentation(str(path))
    lines = [f"# {path.name}\n"]
    for i, slide in enumerate(prs.slides):
        lines.append(f"\n## Slide {i + 1}\n")
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    txt = paragraph.text.strip()
                    if txt:
                        lines.append(f"- {txt}")
    return "\n".join(lines)


def _convert_xlsx_fallback(path: Path) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    lines = [f"# {path.name}\n"]
    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        lines.append(f"\n## Sheet: {sheet_name}\n")
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            lines.append("*(Empty sheet)*\n")
            continue
        header = [str(c) if c is not None else "" for c in rows[0]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for r in rows[1:51]:
            row_cells = [str(c) if c is not None else "" for c in r]
            padded = row_cells + [""] * (len(header) - len(row_cells))
            lines.append("| " + " | ".join(c.replace("\n", " ").strip() for c in padded[: len(header)]) + " |")
        if len(rows) > 51:
            lines.append(f"\n*(Showing 50 of {len(rows)-1} rows)*\n")
    wb.close()
    return "\n".join(lines)


def _convert_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    html_content = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for s in soup(["script", "style"]):
        s.decompose()

    text = soup.get_text(separator="\n\n")
    # Clean up excess blank lines
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    title = soup.title.string if soup.title and soup.title.string else path.name
    return f"# {title}\n\n{cleaned}"


def convert_file_to_markdown(path: Path) -> str:
    """Convert any supported file type to clean Markdown text.

    Uses MarkItDown first where appropriate, with specialized fallbacks for
    IPYNB, CSV, RTF, PDF, DOCX, PPTX, XLSX, HTML, JSON, YAML, etc.
    """
    ext = path.suffix.lower()

    # Specialized handlers
    if ext == ".ipynb":
        return _convert_ipynb(path)
    if ext in (".csv", ".tsv"):
        return _convert_csv(path)
    if ext == ".rtf":
        return _convert_rtf(path)
    if ext in (".html", ".htm"):
        return _convert_html(path)

    # Try MarkItDown for rich office/document formats
    md_converter = _get_markitdown()
    if md_converter and ext in (
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".xls",
        ".html",
        ".htm",
        ".xml",
        ".epub",
        ".zip",
    ):
        try:
            result = md_converter.convert(str(path))
            if result and result.text_content:
                return result.text_content
        except Exception as e:
            log.warning("markitdown_conversion_failed", path=str(path), error=str(e))

    # Fallback to direct handlers
    if ext == ".pdf":
        return _convert_pdf_fallback(path)
    if ext == ".docx":
        return _convert_docx_fallback(path)
    if ext == ".pptx":
        return _convert_pptx_fallback(path)
    if ext in (".xlsx", ".xls"):
        return _convert_xlsx_fallback(path)

    # Structured data formats -> wrap in markdown fences
    if ext == ".json":
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            formatted = json.dumps(json.loads(raw), indent=2)
            return f"# {path.name}\n\n```json\n{formatted}\n```"
        except Exception:
            return f"# {path.name}\n\n```json\n{raw}\n```"

    if ext in (".yaml", ".yml"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        return f"# {path.name}\n\n```yaml\n{raw}\n```"

    if ext == ".toml":
        raw = path.read_text(encoding="utf-8", errors="replace")
        return f"# {path.name}\n\n```toml\n{raw}\n```"

    if ext == ".xml":
        raw = path.read_text(encoding="utf-8", errors="replace")
        return f"# {path.name}\n\n```xml\n{raw}\n```"

    # Default fallback: Treat as text
    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        lang = ext.lstrip(".") if ext else "text"
        if ext in (".md", ".markdown", ".txt"):
            return raw_text
        return f"# {path.name}\n\n```{lang}\n{raw_text}\n```"
    except Exception as err:
        log.error("conversion_failed", path=str(path), error=str(err))
        raise RuntimeError(f"Could not convert {path.name} to Markdown: {err}") from err
