"""Live directory watcher using `watchfiles` to incrementally re-index on file changes."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from watchfiles import awatch

from .config import Settings
from .detector import detect_file_type
from .fulltext import FullTextStore
from .indexing import reindex
from .logging_setup import get_logger
from .security import is_editor_artifact
from .vector import Embedder, VectorStore

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
                # Filter out editor artifacts (.swp, .un~, *~, .tmp) and hidden folders (.fsrag, .git, .obsidian)
                meaningful_changes = [
                    c
                    for c in changes
                    if not is_editor_artifact(c[1])
                    and not any(part.startswith(".") for part in Path(c[1]).parts)
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


async def run_watcher_daemon(settings: Settings | None = None) -> None:
    """Run autonomous background watcher daemon with structured logging and signal trapping."""
    if settings is None:
        settings = Settings.from_env()

    settings.root_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "watcher_daemon_starting",
        profile=settings.profile,
        root_dir=str(settings.root_dir),
        data_dir=str(settings.data_dir),
    )

    ft = FullTextStore(settings)
    embedder = Embedder(settings)
    vec = VectorStore(settings, embedder)

    async def _on_change() -> None:
        log.info("watcher_triggering_reindex")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: reindex(
                settings,
                ft,
                vec,
                full_rebuild=False,
                vector_index=embedder.is_available(),
                detect_file_type=detect_file_type,
            ),
        )
        log.info("watcher_reindex_complete")

    watcher = DirectoryWatcher(
        root_dir=settings.root_dir,
        on_change_callback=_on_change,
        debounce_seconds=1.0,
    )
    watcher.start()

    stop_event = asyncio.Event()

    def _handle_signal(sig: int) -> None:
        log.info("watcher_daemon_signal_received", signal=sig)
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):

            def _sig_cb(s: int = sig) -> None:
                _handle_signal(s)

            loop.add_signal_handler(sig, _sig_cb)

    try:
        # Initial synchronization pass
        await _on_change()
        await stop_event.wait()
    finally:
        watcher.stop()
        log.info("watcher_daemon_shutdown_cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(run_watcher_daemon())
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
