"""Lazily constructed tab screens.

Twelve screens, each with several QWebEngineViews, used to be built before the window appeared (seconds of work and
hundreds of MB). `ScreenRegistry` adds a placeholder tab per screen and builds the real widget on first access, either
because the user opened the tab or because code asked for it (`win.screens["Harvest"]`). Refreshes requested while a
screen is unbuilt are remembered as a dirty flag and applied when the screen is first built.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

log = logging.getLogger(__name__)


class ScreenRegistry:
    def __init__(self, tabs: QTabWidget, factories: dict[str, Callable[[], QWidget]], titles: dict[str, str] | None = None,
                 on_built: Callable[[str, QWidget], None] | None = None):
        self.tabs = tabs
        self._factories = factories
        self._titles = titles or {}
        self._on_built = on_built
        self._built: dict[str, QWidget] = {}
        self._holders: dict[str, QWidget] = {}
        self._dirty: set[str] = set()
        self._order = list(factories)
        for name in self._order:
            holder = QWidget()
            lay = QVBoxLayout(holder)
            lay.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel("Loading…")
            lbl.setProperty("muted", True)
            lay.addWidget(lbl)
            self._holders[name] = holder
            tabs.addTab(holder, self._titles.get(name, name))
        tabs.currentChanged.connect(self._tab_changed)

    # ------------------------------------------------------------------ mapping protocol (built-on-demand)
    def __getitem__(self, name: str) -> QWidget:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def names(self) -> list[str]:
        return list(self._order)

    def get(self, name: str) -> QWidget:
        if name not in self._built:
            if name not in self._factories:
                raise KeyError(name)
            w = self._factories[name]()
            self._built[name] = w
            holder = self._holders[name]
            lay = holder.layout()
            while lay.count():
                item = lay.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            lay.addWidget(w)
            if self._on_built:
                self._on_built(name, w)
            if name in self._dirty:
                self._dirty.discard(name)
                self._safe_refresh(name, w)
        return self._built[name]

    def is_built(self, name: str) -> bool:
        return name in self._built

    def items(self) -> list[tuple[str, QWidget]]:
        """Built screens only, in tab order."""
        return [(n, self._built[n]) for n in self._order if n in self._built]

    def values(self) -> list[QWidget]:
        return [w for _, w in self.items()]

    def holder(self, name: str) -> QWidget:
        return self._holders[name]

    def index_of(self, name: str) -> int:
        return self._order.index(name)

    # ------------------------------------------------------------------ refresh bookkeeping
    def refresh_all(self, skip: set[str] | None = None, on_error: Callable[[str, Exception], None] | None = None) -> None:
        """Refresh built screens now; mark unbuilt ones dirty so they refresh when first shown."""
        skip = skip or set()
        for name in self._order:
            if name in skip:
                continue
            if name in self._built:
                try:
                    self._built[name].refresh()
                except Exception as e:  # never let one screen kill the refresh
                    log.exception("refresh failed for %s", name)
                    if on_error:
                        on_error(name, e)
            else:
                self._dirty.add(name)

    def _safe_refresh(self, name: str, w: QWidget) -> None:
        try:
            w.refresh()
        except Exception:
            log.exception("initial refresh failed for %s", name)

    def _tab_changed(self, i: int) -> None:
        if 0 <= i < len(self._order):
            self.get(self._order[i])

    def show(self, name: str) -> None:
        self.tabs.setCurrentIndex(self.index_of(name))
        self.get(name)

    def ensure_visible_built(self) -> None:
        self._tab_changed(self.tabs.currentIndex())
