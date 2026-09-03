"""Lazy imports for heavy optional dependencies (cvxpy, scipy, sklearn, ...).

Importing cvxpy alone costs ~1.7 s and pulls scipy.stats; the GUI should not pay that before the window is on
screen. `cp = lazy_module("cvxpy")` behaves like the module but performs the import on first attribute access,
so `cp.Variable(n)` inside a solver works unchanged while `import tlh.optim.harvest` stays cheap.
"""
from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Any

_lock = threading.Lock()


class LazyModule(ModuleType):
    """Module proxy: the real import happens on first attribute access (thread-safe)."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__dict__["_lazy_name"] = name
        self.__dict__["_lazy_target"] = None

    def _load(self) -> ModuleType:
        target = self.__dict__["_lazy_target"]
        if target is None:
            with _lock:
                target = self.__dict__["_lazy_target"]
                if target is None:
                    target = importlib.import_module(self.__dict__["_lazy_name"])
                    self.__dict__["_lazy_target"] = target
        return target

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__") and item.endswith("__") and item not in ("__version__", "__file__", "__path__"):
            raise AttributeError(item)
        return getattr(self._load(), item)

    def __dir__(self):
        return dir(self._load())

    @property
    def loaded(self) -> bool:
        return self.__dict__["_lazy_target"] is not None


class LazyObject:
    """Attribute proxy for one object inside a module, e.g. `norm = lazy_object("scipy.stats", "norm")`."""

    def __init__(self, module: str, attr: str):
        self._module, self._attr, self._target = module, attr, None

    def _load(self) -> Any:
        if self._target is None:
            self._target = getattr(importlib.import_module(self._module), self._attr)
        return self._target

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._load(), item)

    def __call__(self, *a, **k):
        return self._load()(*a, **k)


def lazy_module(name: str) -> LazyModule:
    return LazyModule(name)


def lazy_object(module: str, attr: str) -> LazyObject:
    return LazyObject(module, attr)


def warm_up(names: tuple[str, ...] = ("cvxpy", "scipy.stats", "sklearn.covariance"), background: bool = True) -> None:
    """Import the heavy solvers ahead of first use. Called from the GUI after the window is shown."""

    def _run() -> None:
        for n in names:
            try:
                importlib.import_module(n)
            except Exception:  # optional dependency missing: the solver call will report it
                pass

    if background:
        threading.Thread(target=_run, name="tlh-warmup", daemon=True).start()
    else:
        _run()
