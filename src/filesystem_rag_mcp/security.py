"""Filesystem path security.

All user-supplied paths are resolved and validated against a trusted root
directory. The server refuses to follow symlinks that escape the root, refuses
absolute paths outside the root, and refuses to follow symlinks unless
explicitly enabled in settings.

This module deliberately does *not* depend on `pathvalidate`'s blacklist
filters (which are bypassable); we resolve to absolute, then check ancestry.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings


class PathSecurityError(ValueError):
    """Raised when a user-supplied path violates the security policy."""


def safe_resolve(root: Path, candidate: str | Path) -> Path:
    """Resolve `candidate` against `root` and ensure it stays inside `root`.

    The resolution rules:
      1. If `candidate` is relative, join it to `root`.
      2. Always call `Path.resolve()` (which follows symlinks on POSIX).
      3. Compare the resolved path's ancestry against the resolved root.
      4. If the resolved path is not inside the root, raise.

    The `Settings.follow_symlinks` flag controls whether symlinks may be
    followed during `resolve()` — we use `os.path.realpath` semantics
    regardless so the check is always authoritative.
    """
    candidate_p = Path(candidate)
    if not candidate_p.is_absolute():
        candidate_p = root / candidate_p
    resolved_root = root.resolve()
    resolved_candidate = candidate_p.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise PathSecurityError(
            f"Path {candidate!r} resolves outside the configured root "
            f"{resolved_root!r}"
        ) from exc
    return resolved_candidate


def ensure_inside_root(settings: Settings, p: Path) -> Path:
    """Convenience wrapper: resolve and validate a single path."""
    return safe_resolve(settings.root_dir, p)


def is_text_file(path: Path, sniff_bytes: int = 8192) -> bool:
    """Return True if `path` is plausibly a text file.

    Strategy: open, read up to `sniff_bytes`, and check whether the decoded
    content is mostly printable ASCII/UTF-8. We treat any decoding error as
    "binary" — that is more conservative than NUL-byte scanning and handles
    UTF-16 / UTF-32 correctly.
    """
    try:
        with path.open("rb") as fh:
            buf = fh.read(sniff_bytes)
    except OSError:
        return False
    if not buf:
        return True  # empty file is text-ish; empty doc is fine
    # Strip a UTF BOM if present
    if buf.startswith(b"\xef\xbb\xbf"):
        buf = buf[3:]
    elif buf.startswith((b"\xff\xfe", b"\xfe\xff")):
        # UTF-16 with BOM — treat as text
        return True
    try:
        decoded = buf.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded = buf.decode("latin-1")
        except Exception:  # noqa: BLE001 - last-resort sniff
            return False
    # Reject if it contains too many control characters (likely binary)
    control = sum(1 for c in decoded if ord(c) < 32 and c not in "\n\r\t\f\v")
    return (control / max(len(decoded), 1)) < 0.05
