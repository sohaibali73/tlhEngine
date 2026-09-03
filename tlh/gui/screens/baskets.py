"""Model portfolios: build, inspect and use baskets as benchmarks / harvest targets."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...optim.basket import BasketSpec
from ...services.basket_service import BasketService
from .. import charts
from ..widgets import FrameTable, KpiCard, TextPanel, button, hbox, header, pct
from ..workers import run_task

BASKET_COLS = ["name", "n_names", "source", "benchmark_name", "created_at", "description"]


class BasketsScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.svc = BasketService(self.ctx)
        self._sel: str | None = None
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([360, 1100])

    def _left(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(header("Model portfolios", "Min-TE baskets with name caps, sector bands and style tilts. Use one as the benchmark to harvest toward it, or ask the co-pilot to build one."))
        g = QGroupBox("Build a basket (optimizer)")
        f = QFormLayout(g)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Quality-tilt 40")
        self.bench = QComboBox()
        self.bench.setEditable(True)
        self.n_max = QSpinBox()
        self.n_max.setRange(5, 500)
        self.n_max.setValue(40)
        self.max_w = QDoubleSpinBox()
        self.max_w.setRange(0.5, 100)
        self.max_w.setValue(8)
        self.max_w.setSuffix(" %")
        self.band = QDoubleSpinBox()
        self.band.setRange(0, 50)
        self.band.setValue(2)
        self.band.setSuffix(" %")
        self.tilts = QLineEdit()
        self.tilts.setPlaceholderText("quality=0.3, lowvol=0.2, size=-0.1")
        self.exclude = QLineEdit()
        self.exclude.setPlaceholderText("symbols to exclude, comma-separated")
        self.excl_held = QCheckBox("Exclude currently held names (replacement basket)")
        self.excl_wash = QCheckBox("Exclude names that would wash a recent loss sale")
        self.excl_wash.setChecked(True)
        self.desc = QLineEdit()
        f.addRow("Name", self.name)
        f.addRow("Benchmark", self.bench)
        f.addRow("Max names", self.n_max)
        f.addRow("Max weight", self.max_w)
        f.addRow("Sector band", self.band)
        f.addRow("Style tilts (z)", self.tilts)
        f.addRow("Exclude", self.exclude)
        f.addRow(self.excl_held)
        f.addRow(self.excl_wash)
        f.addRow("Description", self.desc)
        self.build_btn = button("Build basket", self.build, primary=True)
        f.addRow(self.build_btn)
        lay.addWidget(g)
        gs = QGroupBox("Sample model portfolios (one click)")
        gsl = QVBoxLayout(gs)
        from ...optim.basket_library import LIBRARY
        lbl = QLabel(f"{len(LIBRARY)} ready-made recipes: index trackers, integrated multi-factor, defensive equity, quality-momentum, risk parity, "
                     "HRP, style tilts, min-CVaR, Black-Litterman and 130/30 / 145/45 long-short tax engines. Built against the live snapshot "
                     "and active model; each becomes a normal saved basket.")
        lbl.setWordWrap(True)
        lbl.setProperty("muted", True)
        gsl.addWidget(lbl)
        self.lib_aud = QComboBox()
        self.lib_aud.addItem("all recipes", None)
        for a in ("core", "growth", "defensive", "income", "long_short"):
            self.lib_aud.addItem(a, a)
        self.lib_btn = button("Build sample library", self.build_library, success=True)
        gsl.addWidget(hbox(self.lib_aud, self.lib_btn))
        lay.addWidget(gs)
        g2 = QGroupBox("Saved baskets")
        gl = QVBoxLayout(g2)
        self.table = FrameTable(BASKET_COLS, filter_box=False)
        self.table.row_selected.connect(self._selected)
        gl.addWidget(self.table)
        gl.addWidget(hbox(button("Set as benchmark", self._set_bench, success=True), button("Delete", self._delete, danger=True), None))
        lay.addWidget(g2, 1)
        scroll.setWidget(w)
        return scroll

    def _right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 0, 0)
        k = QHBoxLayout()
        self.k_name = KpiCard("Basket")
        self.k_te = KpiCard("Tracking error")
        self.k_n = KpiCard("Names")
        self.k_maxw = KpiCard("Max weight")
        self.k_sec = KpiCard("Max sector active")
        for c in (self.k_name, self.k_te, self.k_n, self.k_maxw, self.k_sec):
            k.addWidget(c)
        lay.addLayout(k)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.members = FrameTable(["symbol", "weight", "benchmark_weight", "active", "name", "gics_sector"], pct_cols={"weight", "benchmark_weight", "active"})
        self.tabs.addTab(self.members, "Members")
        self.exp_chart = charts.PlotlyView()
        self.radar = charts.PlotlyView()
        self.sec_chart = charts.PlotlyView()
        row = QSplitter(Qt.Horizontal)
        row.addWidget(self.exp_chart)
        row.addWidget(self.radar)
        col = QSplitter(Qt.Vertical)
        col.addWidget(row)
        col.addWidget(self.sec_chart)
        self.tabs.addTab(col, "Exposures vs benchmark")
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.tabs.addTab(self.info, "Parameters & metrics")
        self.note = TextPanel("How to use")
        self.note.set_text("Select a basket to see members and exposures. 'Set as benchmark' makes it the target for TE everywhere; "
                           "then run Harvest in full-rebalance mode to migrate toward it while harvesting losses. The co-pilot can build "
                           "baskets too: try \"build a 35-name low-vol quality basket that excludes my current holdings\".")
        lay.addWidget(self.note)
        return w

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        df = self.svc.list()
        self.table.set_frame(df)
        cur = self.bench.currentText()
        self.bench.blockSignals(True)
        self.bench.clear()
        self.bench.addItems(["S&P 500", "SPY", "VTI", "Russell 1000"] + [f"basket:{n}" for n in (df["name"].tolist() if not df.empty else [])])
        self.bench.setCurrentText(cur or self.app.risk_service.benchmark_name())
        self.bench.blockSignals(False)
        if self._sel and (df.empty or self._sel not in set(df["name"])):
            self._sel = None
        if self._sel:
            self._show(self._sel)

    def _selected(self, row: dict) -> None:
        self._sel = row["name"]
        self._show(self._sel)

    def _show(self, name: str) -> None:
        run_task(self._load, name, on_done=self._loaded, on_error=self.app.error, wants_progress=False)

    def _load(self, name: str) -> dict:
        b = self.svc.get(name)
        res = self.svc.result(name)
        snap = self.app.data_service.latest_snapshot()
        sec = snap.securities().set_index("symbol") if snap is not None else pd.DataFrame()
        act = self.app.risk_service.active()
        bench = self.app.risk_service.benchmark_weights(snap, act[1], b.get("benchmark_name")) if (act and snap is not None) else pd.Series(dtype=float)
        m = res.weights.rename("weight").to_frame()
        m["benchmark_weight"] = bench.reindex(m.index).fillna(0.0)
        m["active"] = m["weight"] - m["benchmark_weight"]
        if not sec.empty:
            m["name"] = sec["name"].reindex(m.index)
            m["gics_sector"] = sec["gics_sector"].reindex(m.index)
        return {"basket": b, "res": res, "members": m.reset_index().rename(columns={"index": "symbol"})}

    def _loaded(self, d: dict) -> None:
        b, res = d["basket"], d["res"]
        self.k_name.set(b["name"], f"{b['source']} · vs {b.get('benchmark_name') or self.app.risk_service.benchmark_name()}")
        self.k_te.set(pct(res.tracking_error), "annualised vs benchmark")
        self.k_n.set(str(res.n_names), "")
        self.k_maxw.set(pct(res.weights.max()), res.weights.idxmax())
        self.k_sec.set(pct(res.sectors["active"].abs().max()) if len(res.sectors) else "—", "absolute sector deviation")
        self.members.set_frame(d["members"])
        tab = res.exposures.rename(columns={"basket": "portfolio"})
        tab["factor_vol"] = 0.0
        tab["kind"] = ["market" if f == "market" else "macro" if str(f).startswith("macro:") else "style" for f in tab.index]
        self.exp_chart.set_figure(charts.exposure_bars(tab, "Style exposures: basket vs benchmark"))
        self.radar.set_figure(charts.radar(tab, "Style profile"))
        if len(res.sectors):
            self.sec_chart.set_figure(charts.sector_bars(res.sectors.rename(columns={"basket": "portfolio"}), "Sector weights: basket vs benchmark"))
        self.info.setPlainText(f"Description: {b.get('description') or ''}\n\nParameters:\n{pd.Series(b['params']).to_string()}\n\nMetrics:\n{pd.Series(b['metrics']).to_string()}")

    # ------------------------------------------------------------------ actions
    def build(self) -> None:
        name = self.name.text().strip()
        if not name:
            QMessageBox.information(self, "Name", "Give the basket a name.")
            return
        if self.app.risk_service.active() is None:
            QMessageBox.information(self, "No risk model", "Fit a risk model first.")
            return
        tilts = {}
        for part in self.tilts.text().split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    tilts[k.strip()] = float(v)
                except ValueError:
                    pass
        spec = BasketSpec(n_max=self.n_max.value(), max_weight=self.max_w.value() / 100, sector_band=self.band.value() / 100, tilts=tilts,
                          exclude=[s.strip().upper() for s in self.exclude.text().split(",") if s.strip()])
        self.build_btn.setEnabled(False)
        self.app.status("Optimising basket…")
        run_task(self.svc.optimize, name, spec, self.bench.currentText().strip() or None, self.desc.text() or None, "optimizer",
                 self.excl_held.isChecked(), self.excl_wash.isChecked(), on_done=self._built, on_error=self._fail, wants_progress=False)

    def _built(self, out: dict) -> None:
        self.build_btn.setEnabled(True)
        self.app.status(f"Basket '{out['name']}' built: {out['n_names']} names, TE {out['tracking_error']:.2%}.")
        self._sel = out["name"]
        self.refresh()
        self.data_changed.emit()

    def _fail(self, msg: str) -> None:
        self.build_btn.setEnabled(True)
        self.lib_btn.setEnabled(True)
        self.app.error(msg)

    def build_library(self) -> None:
        if self.app.risk_service.active() is None:
            QMessageBox.information(self, "No risk model", "Fit a risk model first.")
            return
        self.lib_btn.setEnabled(False)
        self.app.status("Building the sample model-portfolio library…")
        run_task(self.svc.build_library, None, self.lib_aud.currentData(), on_done=self._lib_done, on_error=self._fail, on_progress=self.app.status)

    def _lib_done(self, df: pd.DataFrame) -> None:
        self.lib_btn.setEnabled(True)
        ok = int(df["status"].astype(str).str.startswith(("optimal", "closed", "mixed")).sum()) if not df.empty else 0
        self.app.status(f"Sample library built: {ok} of {len(df)} baskets succeeded.")
        self.info.setPlainText("Sample library results:\n\n" + df.drop(columns=["pitch"], errors="ignore").to_string(index=False))
        self.tabs.setCurrentWidget(self.info)
        self.refresh()
        self.data_changed.emit()

    def _set_bench(self) -> None:
        if not self._sel:
            return
        self.ctx.set("benchmark_name", f"basket:{self._sel}")
        self.app.status(f"Benchmark set to basket:{self._sel}.")
        self.data_changed.emit()

    def _delete(self) -> None:
        if self._sel and QMessageBox.question(self, "Delete basket", f"Delete '{self._sel}'?") == QMessageBox.Yes:
            self.svc.delete(self._sel)
            self._sel = None
            self.refresh()
            self.data_changed.emit()
