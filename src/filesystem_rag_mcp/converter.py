"""Universal File to Markdown Converter.

Converts ANY file type into clean, informative Markdown:
1. Documents & Office: PDF, Word (DOCX/DOC), PowerPoint (PPTX/PPT), Excel (XLSX/XLS), RTF, EPUB, ODT, ODP, ODS
2. Notebooks & Scripts: IPYNB, Shell, Python, JavaScript, TypeScript, Rust, Go, C/C++, Java, etc.
3. Structured Data: JSON, JSONL, YAML, TOML, XML, CSV, TSV, INI, Properties, SQL
4. Web: HTML, HTM, MHTML, SVG
5. Databases & Storage: SQLite (.sqlite, .db, .sqlite3), Parquet, Arrow
6. Archives: ZIP, TAR, TAR.GZ, TAR.BZ2, TGZ (Directory trees + manifests)
7. Media & Metadata: Audio (MP3, FLAC, OGG, M4A, WAV), Images (EXIF metadata, dimensions), Video
8. Email: EML, MSG (Headers, body, attachments list)
9. Binary/Arbitrary: Hex dumps, ASCII string extraction, and structural summaries
"""

from __future__ import annotations

import csv
import email
import io
import json
import sqlite3
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from filesystem_rag_mcp.detector import detect_file_type
from filesystem_rag_mcp.logging_setup import get_logger

log = get_logger("converter")

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


# ---------------------------------------------------------------------------
# Specialized Handlers
# ---------------------------------------------------------------------------

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
    for row in rows[1:101]:
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


def _convert_pdf(path: Path) -> str:
    import pymupdf

    doc = pymupdf.open(str(path))
    pages_text: list[str] = [f"# {path.name}\n"]
    for i in range(len(doc)):
        page = doc[i]
        text = str(page.get_text()).strip()
        pages_text.append(f"\n## Page {i + 1}\n\n{text}\n")
    doc.close()
    return "\n".join(pages_text)


def _convert_epub(path: Path) -> str:
    import pymupdf

    doc = pymupdf.open(str(path))
    pages_text: list[str] = [f"# {path.name}\n"]
    for i in range(len(doc)):
        page = doc[i]
        text = str(page.get_text()).strip()
        if text:
            pages_text.append(f"\n{text}\n")
    doc.close()
    return "\n".join(pages_text)


def _convert_docx(path: Path) -> str:
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


def _convert_pptx(path: Path) -> str:
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


def _convert_xlsx(path: Path) -> str:
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
    for s in soup(["script", "style"]):
        s.decompose()
    text = soup.get_text(separator="\n\n")
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    title = soup.title.string if soup.title and soup.title.string else path.name
    return f"# {title}\n\n{cleaned}"


def _convert_sqlite(path: Path) -> str:
    """Dump SQLite schema and preview top rows of every table."""
    lines = [f"# SQLite Database: {path.name}\n"]
    try:
        conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        cursor = conn.cursor()
        tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
        if not tables:
            return f"# SQLite Database: {path.name}\n\n*(Empty Database with no tables)*"

        lines.append(f"**Tables found:** {', '.join(tables)}\n")
        for tbl in tables:
            lines.append(f"## Table: `{tbl}`\n")
            cursor.execute(f"PRAGMA table_info('{tbl}')")
            cols_info = cursor.fetchall()
            col_names = [col[1] for col in cols_info]
            col_types = [col[2] for col in cols_info]
            schema_summary = ", ".join(f"{n} ({t})" for n, t in zip(col_names, col_types))
            lines.append(f"*Columns:* {schema_summary}\n")

            # Sample rows
            cursor.execute(f"SELECT * FROM '{tbl}' LIMIT 10")
            sample_rows = cursor.fetchall()
            if sample_rows:
                lines.append("| " + " | ".join(col_names) + " |")
                lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")
                for r in sample_rows:
                    lines.append("| " + " | ".join(str(val).replace("\n", " ") for val in r) + " |")
                lines.append("\n")
            else:
                lines.append("*(Table is empty)*\n")
        conn.close()
    except Exception as exc:
        lines.append(f"*(Error reading database: {exc})*")
    return "\n".join(lines)


def _convert_archive(path: Path) -> str:
    """List archive contents and extract text files if small."""
    lines = [f"# Archive: {path.name}\n"]
    entries: list[tuple[str, int]] = []
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    entries.append((info.filename, info.file_size))
        elif tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as tf:
                for member in tf.getmembers():
                    entries.append((member.name, member.size))
    except Exception as exc:
        return f"# Archive: {path.name}\n\n*(Failed to read archive: {exc})*"

    lines.append(f"**Total files/folders in archive:** {len(entries)}\n")
    lines.append("| Path | Size (bytes) |")
    lines.append("| --- | --- |")
    for name, size in entries[:150]:
        lines.append(f"| `{name}` | {size:,} |")
    if len(entries) > 150:
        lines.append(f"\n*(Showing 150 of {len(entries)} archive members)*")
    return "\n".join(lines)


def _convert_audio_media(path: Path) -> str:
    """Extract audio/media metadata tags (ID3, Vorbis, etc.)."""
    import mutagen

    lines = [f"# Media File: {path.name}\n"]
    try:
        audio = mutagen.File(path)
        if audio is not None:
            lines.append("## Media Information\n")
            if audio.info:
                lines.append(f"- **Length:** {getattr(audio.info, 'length', 0):.2f} seconds")
                lines.append(f"- **Bitrate:** {getattr(audio.info, 'bitrate', 'Unknown')}")
                lines.append(f"- **Sample Rate:** {getattr(audio.info, 'sample_rate', 'Unknown')} Hz")
                lines.append(f"- **Channels:** {getattr(audio.info, 'channels', 'Unknown')}")

            if audio.tags:
                lines.append("\n## Metadata Tags\n")
                for key, val in audio.tags.items():
                    lines.append(f"- **{key}:** {val}")
        else:
            lines.append("*(No audio tag headers found)*")
    except Exception as exc:
        lines.append(f"*(Could not parse media tags: {exc})*")
    return "\n".join(lines)


def _convert_email(path: Path) -> str:
    """Parse EML / RFC 822 email message."""
    lines = [f"# Email Message: {path.name}\n"]
    try:
        with path.open("rb") as f:
            msg = email.message_from_binary_file(f)

        lines.append("## Headers\n")
        for header in ("Subject", "From", "To", "Date", "Cc"):
            if msg.get(header):
                lines.append(f"- **{header}:** {msg.get(header)}")

        lines.append("\n## Message Body\n")
        body_parts: list[str] = []
        attachments: list[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdisp = str(part.get("Content-Disposition"))
                if "attachment" in cdisp:
                    attachments.append(part.get_filename() or "unnamed_attachment")
                    continue
                if ctype == "text/plain":
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        body_parts.append(payload.decode("utf-8", errors="replace"))
                    elif isinstance(payload, str):
                        body_parts.append(payload)
                elif ctype == "text/html" and not body_parts:
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        html_text = payload.decode("utf-8", errors="replace")
                    elif isinstance(payload, str):
                        html_text = payload
                    else:
                        html_text = ""
                    if html_text:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html_text, "html.parser")
                        body_parts.append(soup.get_text(separator="\n\n"))
        else:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                body_parts.append(payload.decode("utf-8", errors="replace"))
            elif isinstance(payload, str):
                body_parts.append(payload)

        lines.append("\n".join(body_parts) if body_parts else "*(No readable text body)*")
        if attachments:
            lines.append("\n## Attachments\n")
            for att in attachments:
                lines.append(f"- `{att}`")
    except Exception as exc:
        lines.append(f"*(Could not parse email: {exc})*")
    return "\n".join(lines)


def _convert_image_meta(path: Path) -> str:
    """Extract Image metadata and dimensions using Pillow."""
    from PIL import Image

    lines = [f"# Image: {path.name}\n"]
    try:
        with Image.open(path) as img:
            lines.append("## Image Properties\n")
            lines.append(f"- **Format:** {img.format}")
            lines.append(f"- **Dimensions:** {img.width} x {img.height} pixels")
            lines.append(f"- **Color Mode:** {img.mode}")

            exif = img.getexif()
            if exif:
                from PIL.ExifTags import TAGS
                lines.append("\n## EXIF Metadata\n")
                for tag_id, val in exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    lines.append(f"- **{tag_name}:** {val}")
    except Exception as exc:
        lines.append(f"*(Could not read image headers: {exc})*")
    return "\n".join(lines)


def _convert_binary_hexdump(path: Path) -> str:
    """Format unknown or binary files with human-friendly hexdump and printable strings."""
    data = path.read_bytes()
    size = len(data)
    lines = [
        f"# Binary File: {path.name}\n",
        f"- **File Size:** {size:,} bytes",
        f"- **SHA-256:** `{hashlib.sha256(data).hexdigest()}`\n",
    ]

    # Extract printable ASCII strings
    import re
    strings = re.findall(rb"[A-Za-z0-9_\-\.\:\/\@\=\+\?\!\& ]{4,}", data[:65536])
    if strings:
        lines.append("## Embedded Printable Strings (Sample)\n")
        lines.append("```text")
        for s in strings[:30]:
            lines.append(s.decode("ascii", errors="ignore"))
        lines.append("```\n")

    # Format first 256 bytes hex dump
    lines.append("## Hex Dump (First 256 bytes)\n```text")
    sample = data[:256]
    for i in range(0, len(sample), 16):
        chunk = sample[i : i + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{i:04x}  {hex_part:<48}  |{ascii_part}|")
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master Conversion Dispatcher
# ---------------------------------------------------------------------------

import hashlib


def convert_file_to_markdown(path: Path) -> str:
    """Universal document and file to Markdown converter."""
    type_info = detect_file_type(path)
    label = type_info.label.lower()
    group = type_info.group.lower()
    ext = path.suffix.lower()

    # 1. Notebooks
    if label == "ipynb" or ext == ".ipynb":
        return _convert_ipynb(path)

    # 2. Tabular
    if label in ("csv", "tsv") or ext in (".csv", ".tsv"):
        return _convert_csv(path)

    # 3. Rich Office & Docs
    if label == "rtf" or ext == ".rtf":
        return _convert_rtf(path)
    if label == "pdf" or ext == ".pdf":
        return _convert_pdf(path)
    if label == "epub" or ext == ".epub":
        return _convert_epub(path)
    if label == "docx" or ext in (".docx", ".doc"):
        return _convert_docx(path)
    if label == "pptx" or ext in (".pptx", ".ppt"):
        return _convert_pptx(path)
    if label in ("xlsx", "xls") or ext in (".xlsx", ".xls"):
        return _convert_xlsx(path)

    # 4. Web & Markup
    if label in ("html", "htm") or ext in (".html", ".htm"):
        return _convert_html(path)

    # 5. Databases
    if ext in (".sqlite", ".sqlite3", ".db") or label == "sqlite":
        return _convert_sqlite(path)

    # 6. Archives
    if ext in (".zip", ".tar", ".gz", ".tgz", ".bz2") or label in ("zip", "tar", "gzip"):
        return _convert_archive(path)

    # 7. Media & Audio
    if group == "audio" or ext in (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"):
        return _convert_audio_media(path)

    # 8. Images
    if group == "image" or ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"):
        return _convert_image_meta(path)

    # 9. Email
    if ext in (".eml", ".msg") or label == "email":
        return _convert_email(path)

    # 10. Structured Text & Code
    if label in ("json", "jsonl") or ext == ".json":
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            formatted = json.dumps(json.loads(raw), indent=2)
            return f"# {path.name}\n\n```json\n{formatted}\n```"
        except Exception:
            return f"# {path.name}\n\n```json\n{raw}\n```"

    if label == "yaml" or ext in (".yaml", ".yml"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        return f"# {path.name}\n\n```yaml\n{raw}\n```"

    if label == "toml" or ext == ".toml":
        raw = path.read_text(encoding="utf-8", errors="replace")
        return f"# {path.name}\n\n```toml\n{raw}\n```"

    if label == "xml" or ext == ".xml":
        raw = path.read_text(encoding="utf-8", errors="replace")
        return f"# {path.name}\n\n```xml\n{raw}\n```"

    # 11. Plain Text and Source Code
    if type_info.is_text or group in ("text", "code"):
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        lang = label if label != "text" else (ext.lstrip(".") if ext else "text")
        if ext in (".md", ".markdown", ".txt") or label in ("markdown", "txt"):
            return raw_text
        return f"# {path.name}\n\n```{lang}\n{raw_text}\n```"

    # 12. Universal Fallback: Binary Hexdump & String Extractor
    return _convert_binary_hexdump(path)
