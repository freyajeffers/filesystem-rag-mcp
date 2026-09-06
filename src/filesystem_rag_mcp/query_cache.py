"""Persistent LRU and TTL-aware search query cache."""

from __future__ import annotations

import collections
import hashlib
import time
from typing import Any


class QueryCache:
    """In-memory LRU query cache with TTL and global version invalidation."""

    def __init__(self, max_entries: int = 1000, ttl_seconds: float = 300.0) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: collections.OrderedDict[str, tuple[float, int, dict[str, Any]]] = (
            collections.OrderedDict()
        )
        self._version = 0

    def _make_key(self, **kwargs: Any) -> str:
        h = hashlib.sha256()
        for k in sorted(kwargs.keys()):
            val = kwargs[k]
            h.update(f"{k}:{val}|".encode("utf-8", errors="replace"))
        return h.hexdigest()

    def get(self, **kwargs: Any) -> dict[str, Any] | None:
        key = self._make_key(**kwargs)
        if key not in self._cache:
            return None

        entry_time, version, result = self._cache[key]
        if version != self._version:
            # Stale due to index refresh
            del self._cache[key]
            return None

        if time.time() - entry_time > self.ttl_seconds:
            # Stale due to TTL expiry
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return result

    def set(self, result: dict[str, Any], **kwargs: Any) -> None:
        key = self._make_key(**kwargs)
        now = time.time()
        self._cache[key] = (now, self._version, result)
        self._cache.move_to_end(key)

        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def invalidate(self) -> None:
        """Bump cache version to instantly invalidate all cached search results."""
        self._version += 1
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
