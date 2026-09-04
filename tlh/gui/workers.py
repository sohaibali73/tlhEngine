"""Background execution for long tasks (data pulls, model fits, optimizations, co-pilot calls).

Completion, failure and progress callbacks are delivered on the GUI thread through a long-lived dispatcher QObject.
Callbacks may be any Python callable (bound methods, functions, lambdas): they are stored on the task and invoked by the
dispatcher, never connected to Qt signals directly. (Connecting a lambda or a bound `signal.emit` to a worker's signal
is silently dropped by PySide once the worker object is collected, which used to leave the co-pilot's working bar on.)
"""
from __future__ import annotations

import logging
import traceback
from collections.abc import Callable

from PySide6.QtCore import QCoreApplication, QObject, QRunnable, QThreadPool, Signal, Slot

log = logging.getLogger(__name__)


class _Dispatcher(QObject):
    """Lives in the GUI thread; `call` is emitted from worker threads and queued to `_run` here."""
    call = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._run)

    @Slot(object, object)
    def _run(self, fn, payload) -> None:
        try:
            fn(payload)
        except Exception:  # a broken callback must not kill the event loop
            log.exception("task callback failed")


_DISPATCHER: _Dispatcher | None = None


def _dispatcher() -> _Dispatcher:
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = _Dispatcher()          # created on first use from the GUI thread
        app = QCoreApplication.instance()
        if app is not None:
            _DISPATCHER.setParent(app)       # destroyed with the application, never after it (clean interpreter exit)
    return _DISPATCHER


class Task(QRunnable):
    """Run `fn(*args, progress=callable, **kwargs)` on the pool. `fn` may accept a `progress` kwarg."""

    def __init__(self, fn: Callable, *args, wants_progress: bool = True, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.wants_progress = wants_progress
        self.on_done: Callable | None = None
        self.on_error: Callable | None = None
        self.on_progress: Callable | None = None
        self.setAutoDelete(True)               # the pool owns the runnable; callbacks are delivered by the dispatcher

    @Slot()
    def run(self) -> None:
        try:
            if self.wants_progress:
                self.kwargs["progress"] = self._progress
            result = self.fn(*self.args, **self.kwargs)
            self._deliver(self.on_done, result)
        except Exception as e:  # surfaced to the UI
            self._deliver(self.on_error, f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    def _progress(self, msg: str) -> None:
        self._deliver(self.on_progress, msg)

    @staticmethod
    def _deliver(fn: Callable | None, payload) -> None:
        if fn is None:
            return
        try:
            _dispatcher().call.emit(fn, payload)
        except RuntimeError:
            pass  # application already shutting down


def run_task(fn: Callable, *args, on_done: Callable | None = None, on_error: Callable | None = None,
             on_progress: Callable | None = None, wants_progress: bool = True, **kwargs) -> Task:
    t = Task(fn, *args, wants_progress=wants_progress, **kwargs)
    t.on_done, t.on_error, t.on_progress = on_done, on_error, on_progress
    _dispatcher()                              # make sure the dispatcher exists in the GUI thread before the worker starts
    QThreadPool.globalInstance().start(t)
    return t
