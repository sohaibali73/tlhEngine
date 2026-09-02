"""Background execution for long tasks (data pulls, model fits, optimizations, Claude calls)."""
from __future__ import annotations

import traceback
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class _Signals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    text = Signal(str)


class Task(QRunnable):
    """Run `fn(*args, progress=callable, **kwargs)` on the pool. `fn` may accept a `progress` kwarg."""

    def __init__(self, fn: Callable, *args, wants_progress: bool = True, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.wants_progress = wants_progress
        self.signals = _Signals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            if self.wants_progress:
                self.kwargs["progress"] = self._progress
            result = self.fn(*self.args, **self.kwargs)
            self._emit(self.signals.finished, result)
        except Exception as e:  # surfaced to the UI
            self._emit(self.signals.failed, f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    def _progress(self, msg: str) -> None:
        self._emit(self.signals.progress, msg)

    @staticmethod
    def _emit(signal, payload) -> None:
        try:
            signal.emit(payload)
        except RuntimeError:
            pass  # receiver (window) already destroyed; nothing to report to


def run_task(fn: Callable, *args, on_done: Callable | None = None, on_error: Callable | None = None,
             on_progress: Callable | None = None, wants_progress: bool = True, **kwargs) -> Task:
    t = Task(fn, *args, wants_progress=wants_progress, **kwargs)
    if on_done:
        t.signals.finished.connect(on_done)
    if on_error:
        t.signals.failed.connect(on_error)
    if on_progress:
        t.signals.progress.connect(on_progress)
    QThreadPool.globalInstance().start(t)
    return t
