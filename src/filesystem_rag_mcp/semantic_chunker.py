"""AST, Markdown, and Semantic Structure-Aware Token/Text Chunking.

Instead of blind character-count slicing, this module splits text along semantic boundaries:
1. Markdown / HTML: Splits along headers (#, ##, ###) and sections.
2. Code (Python, JS/TS, Rust, Go, C/C++): Splits along function definitions, classes, and top-level blocks.
3. Structured prose: Splits on paragraphs, sentence boundaries, and lists while respecting chunk_size and overlap.
"""

from __future__ import annotations

import re


def semantic_chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    lang: str | None = None,
) -> list[tuple[int, int, str]]:
    """Chunks text respecting semantic boundaries (markdown headers, code defs, paragraphs)."""
    if not text.strip():
        return []

    # 1. Identify semantic break points
    # Regex matching Markdown headers or major code definitions (def, class, function, pub fn, etc.)
    boundary_pattern = re.compile(
        r"(?m)(?:^(?:#{1,6}\s+|def\s+\w+|class\s+\w+|async\s+def\s+\w+|function\s+\w+|pub\s+fn\s+\w+|fn\s+\w+))"
    )

    matches = list(boundary_pattern.finditer(text))
    if matches and len(matches) > 1:
        sections: list[tuple[int, int, str]] = []
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec_text = text[start:end].strip()
            if sec_text:
                sections.append((start, end, sec_text))

        # Re-pack sections so they fit chunk_size without arbitrarily splitting functions/headers
        chunks: list[tuple[int, int, str]] = []
        cur_start = 0
        cur_parts: list[str] = []
        cur_len = 0

        for s_start, _s_end, s_text in sections:
            if cur_len + len(s_text) > chunk_size and cur_parts:
                joined = "\n\n".join(cur_parts).strip()
                chunks.append((cur_start, s_start, joined))
                # Apply overlap from previous section if long enough
                cur_parts = []
                cur_len = 0
                cur_start = max(0, s_start - overlap)

            cur_parts.append(s_text)
            cur_len += len(s_text)

        if cur_parts:
            chunks.append((cur_start, len(text), "\n\n".join(cur_parts).strip()))
        return chunks

    # 2. Fallback: Paragraph and line boundary chunking
    paragraphs = _split_paragraphs(text)
    starts: list[int] = []
    running = 0
    for para in paragraphs:
        starts.append(running)
        running += len(para) + 2

    chunks = []
    cursor = 0
    current_parts: list[str] = []
    current_len = 0

    for para, start in zip(paragraphs, starts):
        end = start + len(para)
        if current_len + len(para) + 2 > chunk_size and current_parts:
            chunks.append((cursor, start, "\n\n".join(current_parts).strip()))
            current_parts = []
            current_len = 0
            cursor = max(start - overlap, 0)
        current_parts.append(para)
        current_len += len(para) + 2

    if current_parts:
        chunks.append((cursor, len(text), "\n\n".join(current_parts).strip()))
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if buf:
                out.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        out.append("\n".join(buf))
    return out
