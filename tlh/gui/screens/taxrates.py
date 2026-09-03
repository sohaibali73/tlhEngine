"""Tax rates: every state's capital-gains treatment on a map and in a table, with a combined federal + state calculator."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...explain import explain_state
from ...tax import state_rates as sr
from .. import charts
from ..widgets import FrameTable, KpiCard, TextPanel, button, header, pct
from ..workers import run_task

COLS = ["abbrev", "name", "treatment", "ordinary_top_rate", "st_rate", "lt_rate", "combined_st_top", "combined_lt_top", "description", "local_note"]


class TaxRatesScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(header("Tax rates in every state",
                              f"How each state taxes capital gains (approximate {sr.data_year()} planning figures) and what a harvested loss is worth "
                              "once federal, NIIT and state rates are combined. Click a state on the map or in the table."))
        top = QHBoxLayout()
        g = QGroupBox("Calculator")
        f = QFormLayout(g)
        self.state = QComboBox()
        for s in sorted(sr.all_states().values(), key=lambda x: x.name):
            self.state.addItem(f"{s.name} ({s.abbrev})", s.abbrev)
        self.filing = QComboBox()
        self.filing.addItems(["single", "mfj", "mfs", "hoh"])
        self.income = QDoubleSpinBox()
        self.income.setRange(0, 1e9)
        self.income.setDecimals(0)
        self.income.setSingleStep(10_000)
        self.income.setPrefix("$ ")
        self.income.setValue(300_000)
        self.gain = QDoubleSpinBox()
        self.gain.setRange(0, 1e9)
        self.gain.setDecimals(0)
        self.gain.setSingleStep(10_000)
        self.gain.setPrefix("$ ")
        self.gain.setValue(50_000)
        f.addRow("State", self.state)
        f.addRow("Filing status", self.filing)
        f.addRow("Other taxable income", self.income)
        f.addRow("Gain / loss amount", self.gain)
        f.addRow(hbox_row(button("Compute", self._compute, primary=True), button("Use for this household", self._apply)))
        g.setMaximumWidth(380)
        top.addWidget(g)
        k = QVBoxLayout()
        r1 = QHBoxLayout()
        self.k_st = KpiCard("Short-term, all-in")
        self.k_lt = KpiCard("Long-term, all-in")
        self.k_state = KpiCard("State part")
        self.k_value = KpiCard("Value of the loss entered")
        for c in (self.k_st, self.k_lt, self.k_state, self.k_value):
            r1.addWidget(c)
        k.addLayout(r1)
        self.explain = TextPanel("In plain English")
        k.addWidget(self.explain, 1)
        top.addLayout(k, 1)
        root.addLayout(top)

        split = QSplitter(Qt.Vertical)
        self.map = charts.PlotlyView()
        self.map.point_clicked.connect(self._map_click)
        maprow = QWidget()
        ml = QHBoxLayout(maprow)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.map, 1)
        side = QVBoxLayout()
        self.metric = QComboBox()
        self.metric.addItem("Top combined rate on long-term gains", "combined_lt_top")
        self.metric.addItem("Top combined rate on short-term gains", "combined_st_top")
        self.metric.addItem("State rate on long-term gains", "lt_top_rate")
        self.metric.addItem("State rate on short-term gains", "st_top_rate")
        self.metric.addItem("Top ordinary income rate", "ordinary_top_rate")
        self.metric.currentIndexChanged.connect(lambda _i: self._draw_map())
        side.addWidget(QLabel("Map colour"))
        side.addWidget(self.metric)
        note = QLabel("Federal top rates used for the combined view: 37% ordinary, 20% long-term, plus 3.8% NIIT. "
                      "Local taxes (NYC, Maryland counties, Ohio cities…) are noted, not modelled.")
        note.setWordWrap(True)
        note.setProperty("muted", True)
        side.addWidget(note)
        side.addStretch(1)
        sw = QWidget()
        sw.setLayout(side)
        sw.setMaximumWidth(280)
        ml.addWidget(sw)
        split.addWidget(maprow)
        self.table = FrameTable(COLS, pct_cols={"ordinary_top_rate", "st_rate", "lt_rate", "combined_st_top", "combined_lt_top"})
        self.table.row_selected.connect(self._row)
        split.addWidget(self.table)
        split.setSizes([440, 420])
        root.addWidget(split, 1)
        self._df = None

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        cur = self.ctx.get("tax_state", "")
        if cur:
            i = self.state.findData(cur)
            if i >= 0:
                self.state.setCurrentIndex(i)
        self.income.setValue(float(self.ctx.get("other_income", self.income.value())))
        run_task(sr.table, self.income.value(), on_done=self._loaded, on_error=self.app.error, wants_progress=False)

    def _loaded(self, df) -> None:
        self._df = df
        self.table.set_frame(df)
        self._draw_map()
        self._compute()

    def _draw_map(self) -> None:
        if self._df is None:
            return
        col = self.metric.currentData()
        self.map.set_figure(charts.state_tax_map(self._df, col, self.metric.currentText()))

    # ------------------------------------------------------------------ interactions
    def _map_click(self, payload: dict) -> None:
        pts = payload.get("points") or []
        if pts:
            ab = pts[0].get("customdata") or pts[0].get("x")
            i = self.state.findData(ab)
            if i >= 0:
                self.state.setCurrentIndex(i)
                self._compute()

    def _row(self, row: dict) -> None:
        i = self.state.findData(row.get("abbrev"))
        if i >= 0:
            self.state.setCurrentIndex(i)
            self._compute()

    def _compute(self) -> None:
        st = self.state.currentData()
        if not st:
            return
        c = sr.combined_marginal(st, self.filing.currentText(), self.income.value(), self.gain.value())
        self.k_st.set(pct(c["total_st"], 1), f"fed {pct(c['fed_st'], 0)} + NIIT {pct(c['niit'], 1)} + state {pct(c['state_st'], 2)}")
        self.k_lt.set(pct(c["total_lt"], 1), f"fed {pct(c['fed_lt'], 0)} + NIIT {pct(c['niit'], 1)} + state {pct(c['state_lt'], 2)}")
        self.k_state.set(f"{pct(c['state_st'], 2)} / {pct(c['state_lt'], 2)}", "short / long")
        g = self.gain.value()
        self.k_value.set(f"${g * c['total_st']:,.0f} / ${g * c['total_lt']:,.0f}", "tax saved by a loss of this size, short / long")
        self.explain.set_text(explain_state(c))

    def _apply(self) -> None:
        from ...services.home_service import HomeService
        out = HomeService(self.ctx).apply_tax_setup(self.state.currentData(), self.filing.currentText(), self.income.value())
        self.app.status(f"Household tax settings set to {self.state.currentText()}: ST {out['st_rate']:.1%}, LT {out['lt_rate']:.1%}.")
        self.data_changed.emit()


def hbox_row(*ws) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in ws:
        lay.addWidget(x)
    return w
