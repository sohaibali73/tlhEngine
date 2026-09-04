"""Shared widgets: KPI cards, filterable DataFrame table, banners, section headers."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .models import DataFrameModel, FrameProxy


class KpiCard(QFrame):
    def __init__(self, label: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setProperty("card", True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        self.lbl = QLabel(label.upper())
        self.lbl.setProperty("kpiLabel", True)
        self.val = QLabel(value)
        self.val.setProperty("kpi", True)
        self.sub = QLabel("")
        self.sub.setProperty("muted", True)
        lay.addWidget(self.lbl)
        lay.addWidget(self.val)
        lay.addWidget(self.sub)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, value: str, sub: str = "", color: str | None = None) -> None:
        self.val.setText(value)
        self.sub.setText(sub)
        self.val.setStyleSheet(f"color: {color};" if color else "")


def money(v: float | None, decimals: int = 0) -> str:
    if v is None or v != v:
        return "—"
    s = f"{abs(v):,.{decimals}f}"
    return f"-${s}" if v < 0 else f"${s}"


def pct(v: float | None, decimals: int = 2) -> str:
    if v is None or v != v:
        return "—"
    return f"{v * 100:.{decimals}f}%"


def sign_color(v: float | None) -> str | None:
    if v is None or v != v or v == 0:
        return None
    return theme.GREEN if v > 0 else theme.RED


class Banner(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.hide()

    def show_msg(self, text: str, kind: str = "warn") -> None:
        self.setProperty("banner", kind)
        self.setText(text)
        self.style().unpolish(self)
        self.style().polish(self)
        self.show()

    def clear_msg(self) -> None:
        self.hide()


def header(text: str, sub: str | None = None) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 4)
    lay.setSpacing(0)
    h = QLabel(text)
    h.setProperty("h1", True)
    lay.addWidget(h)
    if sub:
        s = QLabel(sub)
        s.setProperty("muted", True)
        s.setWordWrap(True)
        lay.addWidget(s)
    return w


class FrameTable(QWidget):
    """QTableView + sort/filter proxy + filter box + CSV export. Emits row_selected(dict) and row_activated(dict)."""
    row_selected = Signal(dict)
    row_activated = Signal(dict)

    def __init__(self, columns: list[str] | None = None, pct_cols: set[str] | None = None, filter_box: bool = True,
                 selection: str = "single", parent=None):
        super().__init__(parent)
        self.columns = columns
        self.model = DataFrameModel(pct_cols=pct_cols)
        self.proxy = FrameProxy()
        self.proxy.setSourceModel(self.model)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.setAlternatingRowColors(True)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(22)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection if selection == "multi" else QAbstractItemView.SingleSelection)
        self.view.setWordWrap(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self.proxy.setFilterFixedString)
        self.count = QLabel("")
        self.count.setProperty("muted", True)
        self.export_btn = QPushButton("CSV")
        self.export_btn.setFixedWidth(46)
        self.export_btn.clicked.connect(self.export_csv)
        if filter_box:
            top.addWidget(self.filter, 1)
            top.addWidget(self.count)
            top.addWidget(self.export_btn)
            lay.addLayout(top)
        lay.addWidget(self.view, 1)
        self.view.selectionModel().selectionChanged.connect(self._sel)
        self.view.doubleClicked.connect(lambda idx: self.row_activated.emit(self._full_row(idx)))

    def set_frame(self, df: pd.DataFrame | None) -> None:
        if df is None:
            df = pd.DataFrame()
        self._full = df.reset_index(drop=True)          # hidden columns stay available to row_selected/selected_rows
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
            df = df[cols] if cols else df
        self.model.set_frame(df)
        self.count.setText(f"{len(df):,} rows")
        self._size_columns()

    def _size_columns(self, sample: int = 60, max_width: int = 320) -> None:
        """Column widths from the header text and the first `sample` rows (resizeColumnsToContents walks every row)."""
        fm = self.view.fontMetrics()
        n = min(self.model.rowCount(), sample)
        pad = 18
        for j in range(self.model.columnCount()):
            w = fm.horizontalAdvance(str(self.model.headerData(j, Qt.Horizontal))) + pad + 12
            for i in range(n):
                w = max(w, fm.horizontalAdvance(self.model.display_at(i, j)) + pad)
            self.view.setColumnWidth(j, min(w, max_width))

    def _full_row(self, proxy_index) -> dict:
        src_row = self.proxy.mapToSource(proxy_index).row()
        full = getattr(self, "_full", None)
        if full is not None and 0 <= src_row < len(full):
            return full.iloc[src_row].to_dict()
        return self.proxy.source_row_dict(proxy_index)

    def _sel(self, *_):
        idxs = self.view.selectionModel().selectedRows()
        if idxs:
            self.row_selected.emit(self._full_row(idxs[0]))

    def selected_rows(self) -> list[dict]:
        return [self._full_row(i) for i in self.view.selectionModel().selectedRows()]

    def export_csv(self) -> None:
        if self.model.frame.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "table.csv", "CSV (*.csv)")
        if path:
            self.model.frame.to_csv(path, index=False)


def button(text: str, on_click: Callable | None = None, primary: bool = False, danger: bool = False, success: bool = False,
           tooltip: str = "") -> QPushButton:
    b = QPushButton(text)
    if primary:
        b.setProperty("primary", True)
    if danger:
        b.setProperty("danger", True)
    if success:
        b.setProperty("success", True)
    if tooltip:
        b.setToolTip(tooltip)
    if on_click:
        b.clicked.connect(on_click)
    return b


def hbox(*widgets, stretch_last: bool = False, margins: int = 0) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(margins, margins, margins, margins)
    for x in widgets:
        if x is None:
            lay.addStretch(1)
        else:
            lay.addWidget(x)
    if stretch_last:
        lay.addStretch(1)
    return w


def vbox(*widgets, margins: int = 0) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(margins, margins, margins, margins)
    for x in widgets:
        if x is None:
            lay.addStretch(1)
        else:
            lay.addWidget(x)
    return w


class TextPanel(QFrame):
    """Read-only wrapped text panel for explanations."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setProperty("card", True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        if title:
            t = QLabel(title)
            t.setProperty("kpiLabel", True)
            lay.addWidget(t)
        self.body = QLabel("")
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(self.body, 1)

    def set_text(self, text: str) -> None:
        self.body.setText(text)
