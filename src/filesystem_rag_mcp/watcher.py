"""Live directory watcher using `watchfiles` to incrementally re-index on file changes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from watchfiles import awatch

from .logging_setup import get_logger

log = get_logger("watcher")


class DirectoryWatcher:
    def __init__(
        self,
        root_dir: Path,
        on_change_callback: Callable[[], Coroutine[Any, Any, None]],
        debounce_seconds: float = 1.0,
    ) -> None:
        self.root_dir = root_dir
        self.on_change_callback = on_change_callback
        self.debounce_seconds = debounce_seconds
        self._task: asyncio.Task[Any] | None = None
        self._running = False

    def start(self) -> None:
        """Start directory watching in background."""
        if self._running:
            return
        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._watch_loop())
            log.info("directory_watcher_started", root=str(self.root_dir))
        except RuntimeError:
            log.warning("directory_watcher_no_running_loop")

    def stop(self) -> None:
        """Stop directory watching."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            log.info("directory_watcher_stopped")

    async def _watch_loop(self) -> None:
        """Watches for changes with debouncing."""
        try:
            async for changes in awatch(self.root_dir, debounce=int(self.debounce_seconds * 1000)):
                if not self._running:
                    break
                # Filter out hidden or internal folders like .fsrag, .git
                meaningful_changes = [
                    c for c in changes if not any(part.startswith(".") for part in Path(c[1]).parts)
                ]
                if meaningful_changes:
                    log.info("file_changes_detected", count=len(meaningful_changes))
                    try:
                        await self.on_change_callback()
                    except Exception as exc:
                        log.error("watcher_callback_failed", error=str(exc))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("watcher_loop_error", error=str(exc))
