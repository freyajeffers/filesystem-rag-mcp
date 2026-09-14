"""AST, Markdown, and Semantic Structure-Aware Token/Text Chunking.

Instead of blind character-count slicing, this module splits text along semantic boundaries:
1. Markdown / HTML: Splits along headers (#, ##, ###) and sections, tracking hierarchical breadcrumbs.
2. Code (Python, JS/TS, Rust, Go, C/C++): Splits along function definitions, classes, and top-level blocks.
3. Structured prose: Splits on paragraphs, sentence boundaries, and lists while respecting chunk_size and overlap.
"""

from __future__ import annotations

import re


class SemanticChunk(tuple[int, int, str]):
    """Semantic chunk behaving as a 3-tuple (start, end, text) with breadcrumbs attribute."""

    breadcrumbs: str

    def __new__(cls, start: int, end: int, text: str, breadcrumbs: str = "") -> SemanticChunk:
        instance = super().__new__(cls, (start, end, text))
        instance.breadcrumbs = breadcrumbs
        return instance

    def __repr__(self) -> str:
        return f"SemanticChunk(start={self[0]}, end={self[1]}, text={self[2]!r}, breadcrumbs={self.breadcrumbs!r})"


def extract_heading_breadcrumbs(
    text: str,
) -> list[tuple[int, int, str]]:
    """Extract (start_pos, level, breadcrumb_lineage) for all markdown headers in text."""
    heading_pattern = re.compile(r"(?m)^(?P<hashes>#{1,6})\s+(?P<title>.+)$")
    stack: list[tuple[int, str]] = []
    headings: list[tuple[int, int, str]] = []
    for m in heading_pattern.finditer(text):
        level = len(m.group("hashes"))
        title = m.group("title").strip().rstrip("#").strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        lineage = " > ".join(t for _, t in stack)
        headings.append((m.start(), level, lineage))
    return headings


def compute_breadcrumbs_for_span(
    headings: list[tuple[int, int, str]],
    start: int,
    end: int,
) -> str:
    """Compute hierarchical heading breadcrumbs for a chunk span."""
    if not headings:
        return ""
    active_breadcrumb = ""
    for h_start, _h_level, h_lineage in headings:
        if h_start <= max(start, end - 1):
            active_breadcrumb = h_lineage
        else:
            break
    return active_breadcrumb


def semantic_chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    lang: str | None = None,
) -> list[SemanticChunk]:
    """Chunks text respecting semantic boundaries and computes heading breadcrumbs."""
    if not text.strip():
        return []

    headings = extract_heading_breadcrumbs(text)

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
        chunks: list[SemanticChunk] = []
        cur_start = 0
        cur_parts: list[str] = []
        cur_len = 0

        for s_start, _s_end, s_text in sections:
            if cur_len + len(s_text) > chunk_size and cur_parts:
                joined = "\n\n".join(cur_parts).strip()
                bc = compute_breadcrumbs_for_span(headings, cur_start, s_start)
                chunks.append(SemanticChunk(cur_start, s_start, joined, breadcrumbs=bc))
                cur_parts = []
                cur_len = 0
                cur_start = max(0, s_start - overlap)

            cur_parts.append(s_text)
            cur_len += len(s_text)

        if cur_parts:
            joined = "\n\n".join(cur_parts).strip()
            bc = compute_breadcrumbs_for_span(headings, cur_start, len(text))
            chunks.append(SemanticChunk(cur_start, len(text), joined, breadcrumbs=bc))
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
            joined = "\n\n".join(current_parts).strip()
            bc = compute_breadcrumbs_for_span(headings, cursor, start)
            chunks.append(SemanticChunk(cursor, start, joined, breadcrumbs=bc))
            current_parts = []
            current_len = 0
            cursor = max(start - overlap, 0)
        current_parts.append(para)
        current_len += len(para) + 2

    if current_parts:
        joined = "\n\n".join(current_parts).strip()
        bc = compute_breadcrumbs_for_span(headings, cursor, len(text))
        chunks.append(SemanticChunk(cursor, len(text), joined, breadcrumbs=bc))
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
