import sqlite3
import zipfile
from pathlib import Path

from filesystem_rag_mcp.converter import convert_file_to_markdown


def test_convert_sqlite_database(tmp_path: Path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE products (sku TEXT PRIMARY KEY, price REAL, stock INTEGER)")
    conn.execute("INSERT INTO products VALUES ('SKU1', 9.99, 100), ('SKU2', 49.50, 15)")
    conn.commit()
    conn.close()

    md = convert_file_to_markdown(db_path)
    assert "# SQLite Database: test.db" in md
    assert "## Table: `products`" in md
    assert "sku (TEXT), price (REAL), stock (INTEGER)" in md
    assert "| SKU1 | 9.99 | 100 |" in md


def test_convert_zip_archive(tmp_path: Path):
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("docs/readme.txt", "Read me first")
        zf.writestr("src/main.py", "print('hello')")

    md = convert_file_to_markdown(zip_path)
    assert "# Archive: bundle.zip" in md
    assert "**Total files/folders in archive:** 2" in md
    assert "`docs/readme.txt`" in md
    assert "`src/main.py`" in md


def test_convert_unknown_binary_hexdump(tmp_path: Path):
    bin_path = tmp_path / "firmware.bin"
    content = b"\x00\x01\x02\x03FirmwareV1SecretKey2026\xff\xfe\x00\x10"
    bin_path.write_bytes(content)

    md = convert_file_to_markdown(bin_path)
    assert "# Binary File: firmware.bin" in md
    assert "FirmwareV1SecretKey2026" in md
    assert "Hex Dump (First 256 bytes)" in md
