from pathlib import Path
import pytest
import pymupdf
import json

from filesystem_rag_mcp.detector import detect_file_type
from filesystem_rag_mcp.converter import convert_file_to_markdown
from filesystem_rag_mcp.security import is_indexable_file


def test_content_detection_pdf_no_extension(tmp_path: Path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "PDF content without extension")
    pdf_path = tmp_path / "raw_pdf_binary"
    doc.save(str(pdf_path))
    doc.close()

    info = detect_file_type(pdf_path)
    assert info.label == "pdf"
    assert info.is_convertible is True
    assert is_indexable_file(pdf_path) is True

    md = convert_file_to_markdown(pdf_path)
    assert "PDF content without extension" in md


def test_content_detection_json_with_misleading_extension(tmp_path: Path):
    p = tmp_path / "data.log"
    p.write_text(json.dumps({"status": "healthy", "service": "auth"}))

    info = detect_file_type(p)
    assert info.label in ("json", "jsonl")
    assert info.is_convertible is True
    assert is_indexable_file(p) is True

    md = convert_file_to_markdown(p)
    assert "```json" in md
    assert "healthy" in md


def test_content_detection_csv_with_arbitrary_extension(tmp_path: Path):
    p = tmp_path / "table.xyz"
    p.write_text("city,pop,country\nParis,2161000,France\nBerlin,3645000,Germany\n")

    info = detect_file_type(p)
    assert info.label == "csv"
    assert info.is_convertible is True
    assert is_indexable_file(p) is True

    md = convert_file_to_markdown(p)
    assert "| city | pop | country |" in md
    assert "| Paris | 2161000 | France |" in md
