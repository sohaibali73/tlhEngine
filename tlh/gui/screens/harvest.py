"""Harvest recommendations: constraint hierarchy, run, review trades with wash explanations, frontier, history."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...optim.harvest import PRIORITIES, HarvestConfig
from .. import charts
from ..widgets import FrameTable, KpiCard, TextPanel, button, hbox, header, money, pct, sign_color, vbox
from ..workers import run_task

TRADE_COLS = ["side", "account_name", "symbol", "quantity", "est_price", "est_value", "lot_id", "realized_gain", "term",
              "tax_benefit", "replacement_for", "wash_status", "acted_on"]
BLOCK_COLS = ["symbol", "account_id", "lot_id", "quantity", "loss", "term", "wash_status", "wash_explanation"]
REPL_COLS = ["sold_symbol", "candidate", "correlation", "source", "wash_status", "wash_explanation"]
RUN_COLS = ["id", "created_at", "run_type", "as_of_date", "harvested_loss", "tax_alpha", "te_before", "te_after", "n_trades", "mode", "priority"]
PRIO_LABEL = {"tax": "Tax alpha", "tracking_error": "Tracking error", "factor_neutrality": "Factor neutrality"}


class HarvestScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.hs = app.harvest_service
        self.result = None
        self.run_id = None
        self.frontier_df = None
        self._build()
        self._load_cfg(self.hs.load_config())

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._config_panel())
        split.addWidget(self._results_panel())
        split.setSizes([330, 1100])

    def _config_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(header("Harvest", "Wash-sale safety is a hard pre-filter. Everything below trades off tax alpha, tracking error and factor drift."))

        g = QGroupBox("Constraint hierarchy (drag to reorder)")
        gl = QVBoxLayout(g)
        self.prio = QListWidget()
        self.prio.setDragDropMode(QAbstractItemView.InternalMove)
        self.prio.setFixedHeight(80)
        for p in PRIORITIES:
            self.prio.addItem(PRIO_LABEL[p])
        gl.addWidget(self.prio)
        wrow = QFormLayout()
        self.w1, self.w2, self.w3 = (self._spin(0, 10, 0.05, 2) for _ in range(3))
        wrow.addRow("Weight 1st / 2nd / 3rd", hbox(self.w1, self.w2, self.w3))
        gl.addLayout(wrow)
        lay.addWidget(g)

        g2 = QGroupBox("Budgets & limits")
        f = QFormLayout(g2)
        self.mode = QComboBox()
        self.mode.addItems(["opportunistic", "full_rebalance"])
        self.te = self._spin(0.05, 20, 0.05, 2, suffix=" %")
        self.te_hard = QCheckBox("hard cap")
        self.fdrift = self._spin(0.01, 2, 0.01, 2)
        self.sector = self._spin(0.1, 20, 0.1, 1, suffix=" %")
        self.turn = self._spin(1, 100, 1, 0, suffix=" %")
        self.maxw = self._spin(1, 100, 0.5, 1, suffix=" %")
        self.min_trade = self._spin(0, 1e6, 100, 0, prefix="$ ")
        self.min_loss = self._spin(0, 1e6, 50, 0, prefix="$ ")
        self.cost = self._spin(0, 100, 0.5, 1, suffix=" bps")
        self.horizon = self._spin(0, 60, 1, 0, suffix=" yrs")
        self.target = self._spin(0, 1e8, 1000, 0, prefix="$ ")
        self.target.setSpecialValueText("no cap")
        self.bench = QComboBox()
        self.bench.setEditable(True)
        self.bench.addItems(["S&P 500", "SPY", "IVV", "VOO", "VTI", "Russell 1000", "S&P Composite 1500"])
        f.addRow("Mode", self.mode)
        f.addRow("TE budget", hbox(self.te, self.te_hard))
        f.addRow("Style drift unit (z)", self.fdrift)
        f.addRow("Sector drift max", self.sector)
        f.addRow("Turnover max", self.turn)
        f.addRow("Max position weight", self.maxw)
        f.addRow("Min trade value", self.min_trade)
        f.addRow("Min loss per lot", self.min_loss)
        f.addRow("Transaction cost", self.cost)
        f.addRow("Tax deferral horizon", self.horizon)
        f.addRow("Target loss (soft cap)", self.target)
        f.addRow("Benchmark", self.bench)
        lay.addWidget(g2)

        self.run_btn = button("Run harvest", self.run, primary=True)
        self.front_btn = button("Frontier sweep", self.run_frontier, tooltip="Sweep hard TE budgets; plots tax alpha vs TE")
        self.prio_btn = button("Compare priorities", self.run_priorities, tooltip="Run all 6 orderings of the hierarchy")
        lay.addWidget(self.run_btn)
        lay.addWidget(hbox(self.front_btn, self.prio_btn))
        lay.addWidget(button("Save as default config", self._save_cfg))
        self.cfg_note = QLabel("")
        self.cfg_note.setProperty("muted", True)
        self.cfg_note.setWordWrap(True)
        lay.addWidget(self.cfg_note)
        lay.addStretch(1)
        scroll.setWidget(w)
        return scroll

    def _results_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 0, 0)
        k = QHBoxLayout()
        self.k_loss = KpiCard("Harvested loss")
        self.k_benefit = KpiCard("Tax benefit (now)")
        self.k_alpha = KpiCard("Tax alpha (NPV)")
        self.k_te = KpiCard("Tracking error")
        self.k_turn = KpiCard("Turnover")
        self.k_trades = KpiCard("Trades")
        for c in (self.k_loss, self.k_benefit, self.k_alpha, self.k_te, self.k_turn, self.k_trades):
            k.addWidget(c)
        lay.addLayout(k)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        # trades
        self.trades = FrameTable(TRADE_COLS, selection="multi")
        self.trades.row_selected.connect(self._trade_selected)
        self.explain = TextPanel("Why this trade is wash-sale safe")
        self.explain.set_text("Select a trade.")
        acts = hbox(button("Mark selected as acted on", self._mark_acted),
                    button("Book selected as executed (paper)", self._book, success=True,
                           tooltip="Records the selected recommended trades as transactions at the estimated price. No orders are sent."),
                    None, button("Export workbook…", lambda: self.app.goto("Export")))
        tsplit = QSplitter(Qt.Vertical)
        tsplit.addWidget(self.trades)
        tsplit.addWidget(self.explain)
        tsplit.setSizes([500, 130])
        self.tabs.addTab(vbox(tsplit, acts), "Trades")
        # blocked
        self.blocked = FrameTable(BLOCK_COLS)
        self.blocked.row_selected.connect(lambda r: self.explain_blocked.set_text(r.get("wash_explanation", "")))
        self.explain_blocked = TextPanel("Why this lot is blocked")
        bsplit = QSplitter(Qt.Vertical)
        bsplit.addWidget(self.blocked)
        bsplit.addWidget(self.explain_blocked)
        bsplit.setSizes([500, 130])
        self.tabs.addTab(bsplit, "Blocked lots")
        # replacements
        self.repl = FrameTable(REPL_COLS)
        self.tabs.addTab(self.repl, "Replacements")
        # charts
        self.exp_chart = charts.PlotlyView()
        self.te_chart_b = charts.PlotlyView()
        self.te_chart_a = charts.PlotlyView()
        self.sec_chart = charts.PlotlyView()
        te_row = QSplitter(Qt.Horizontal)
        te_row.addWidget(self.te_chart_b)
        te_row.addWidget(self.te_chart_a)
        risk_split = QSplitter(Qt.Vertical)
        risk_split.addWidget(self.exp_chart)
        risk_split.addWidget(te_row)
        risk_split.addWidget(self.sec_chart)
        self.tabs.addTab(risk_split, "Before / after risk")
        # frontier
        self.front_chart = charts.PlotlyView()
        self.front_table = FrameTable(["te_budget", "te_after", "harvested_loss", "tax_alpha", "tax_alpha_bps", "turnover", "n_trades", "status"])
        fsplit = QSplitter(Qt.Vertical)
        fsplit.addWidget(self.front_chart)
        fsplit.addWidget(self.front_table)
        self.tabs.addTab(fsplit, "Frontier")
        # priorities
        self.prio_chart = charts.PlotlyView()
        self.prio_table = FrameTable(["priority", "harvested_loss", "tax_alpha", "te_after", "max_style_drift", "max_sector_drift", "turnover", "n_trades", "status"])
        psplit = QSplitter(Qt.Vertical)
        psplit.addWidget(self.prio_chart)
        psplit.addWidget(self.prio_table)
        self.tabs.addTab(psplit, "Priority comparison")
        # history
        self.runs = FrameTable(RUN_COLS)
        self.runs.row_activated.connect(self._load_run)
        self.tabs.addTab(vbox(QLabel("Double-click a run to load it (saved snapshot: trades, blocked lots, risk before/after)."), self.runs), "Run history")
        return w

    def _spin(self, lo, hi, step, dec, prefix="", suffix="") -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(dec)
        s.setPrefix(prefix)
        s.setSuffix(suffix)
        return s

    # ------------------------------------------------------------------ config <-> widgets
    def _load_cfg(self, c: HarvestConfig) -> None:
        self.prio.clear()
        for p in c.priority:
            self.prio.addItem(PRIO_LABEL[p])
        self.w1.setValue(c.priority_weights[0])
        self.w2.setValue(c.priority_weights[1])
        self.w3.setValue(c.priority_weights[2])
        self.mode.setCurrentText(c.mode)
        self.te.setValue(c.te_budget * 100)
        self.te_hard.setChecked(c.te_hard)
        self.fdrift.setValue(c.factor_drift_budget)
        self.sector.setValue(c.sector_drift_max * 100)
        self.turn.setValue(c.turnover_max * 100)
        self.maxw.setValue(c.max_position_weight * 100)
        self.min_trade.setValue(c.min_trade_value)
        self.min_loss.setValue(c.min_loss_value)
        self.cost.setValue(c.cost_bps)
        self.horizon.setValue(c.tax_horizon_years if c.tax_horizon_years != float("inf") else 60)
        self.target.setValue(c.target_loss or 0)
        self.bench.setCurrentText(self.app.risk_service.benchmark_name())

    def config(self) -> HarvestConfig:
        inv = {v: k for k, v in PRIO_LABEL.items()}
        order = tuple(inv[self.prio.item(i).text()] for i in range(self.prio.count()))
        return HarvestConfig(
            mode=self.mode.currentText(), priority=order,
            priority_weights=(self.w1.value(), self.w2.value(), self.w3.value()),
            te_budget=self.te.value() / 100, te_hard=self.te_hard.isChecked(), factor_drift_budget=self.fdrift.value(),
            sector_drift_max=self.sector.value() / 100, turnover_max=self.turn.value() / 100,
            max_position_weight=self.maxw.value() / 100, min_trade_value=self.min_trade.value(),
            min_loss_value=self.min_loss.value(), cost_bps=self.cost.value(),
            tax_horizon_years=self.horizon.value() if self.horizon.value() < 60 else float("inf"),
            target_loss=self.target.value() or None,
        )

    def _save_cfg(self) -> None:
        self.hs.save_config(self.config())
        self.ctx.set("benchmark_name", self.bench.currentText().strip())
        self.app.status("Harvest configuration saved.")

    # ------------------------------------------------------------------ running
    def _preflight(self) -> bool:
        if self.ctx.current_entity_id is None:
            QMessageBox.information(self, "No entity", "Select or create a tax entity first.")
            return False
        if self.app.risk_service.active() is None:
            QMessageBox.information(self, "No risk model", "Fit a risk model on the Risk model tab first.")
            return False
        return True

    def run(self) -> None:
        if not self._preflight():
            return
        cfg = self.config()
        self.ctx.set("benchmark_name", self.bench.currentText().strip())
        self._busy(True, "Running harvest optimizer…")
        run_task(self.hs.run, self.ctx.current_entity_id, cfg, on_done=self._done, on_error=self._fail, wants_progress=False)

    def _done(self, out) -> None:
        self.run_id, self.result = out
        self._busy(False, f"Harvest run #{self.run_id} complete ({self.result.solver_status}).")
        self._show_result(self.result.summary, self.hs.load_run(self.run_id))
        self.data_changed.emit()

    def _fail(self, msg: str) -> None:
        self._busy(False, "Harvest failed.")
        self.app.error(msg)

    def _busy(self, on: bool, msg: str) -> None:
        for b in (self.run_btn, self.front_btn, self.prio_btn):
            b.setEnabled(not on)
        self.app.status(msg)

    def _show_result(self, s: dict, run: dict) -> None:
        self.k_loss.set(money(s.get("harvested_loss")), f"ST {money(s.get('harvested_loss_st'))} · LT {money(s.get('harvested_loss_lt'))}", sign_color(-(s.get("harvested_loss") or 0)) and None)
        self.k_benefit.set(money(s.get("tax_benefit")), "at marginal rates", sign_color(s.get("tax_benefit")))
        self.k_alpha.set(money(s.get("tax_alpha")), f"{s.get('tax_alpha_bps', 0):.1f} bps of portfolio", sign_color(s.get("tax_alpha")))
        te_b, te_a = s.get("te_before"), s.get("te_after")
        self.k_te.set(f"{pct(te_b)} → {pct(te_a)}", "before → after", sign_color((te_b or 0) - (te_a or 0)))
        self.k_turn.set(pct(s.get("turnover")), f"sell {money(s.get('sell_value'))} · buy {money(s.get('buy_value'))}")
        self.k_trades.set(f"{s.get('n_sells', 0)} sells / {s.get('n_buys', 0)} buys", f"{s.get('n_blocked_lots', 0)} lots wash-blocked · {s.get('solver_status', '')}")
        self.cfg_note.setText(f"Run #{run.get('id')} · {s.get('mode')} · " + " > ".join(PRIO_LABEL.get(p, p) for p in s.get("priority", [])))
        self.trades.set_frame(run["trades"])
        self.blocked.set_frame(run["blocked"])
        self.repl.set_frame(run["replacements"])
        ex = run["exposures"]
        if not ex.empty:
            tab = pd.DataFrame({"portfolio": ex["after"], "benchmark": ex["before"]})
            tab["active"] = tab["portfolio"] - tab["benchmark"]
            tab["kind"] = ["market" if f == "market" else "sector" if str(f).startswith(("sec:", "ind:")) else "macro" if str(f).startswith("macro:") else "style" for f in tab.index]
            fig = charts.exposure_bars(tab, "Style exposures: after (blue) vs before (grey)")
            fig.data[0].name, fig.data[1].name, fig.data[2].name = "After", "Before", "Change"
            self.exp_chart.set_figure(fig)
        for view, d, lbl in ((self.te_chart_b, run["te_before"], "TE decomposition — before"), (self.te_chart_a, run["te_after"], "TE decomposition — after")):
            if not d.empty:
                te = float((d["variance"].sum()) ** 0.5)
                d.attrs["tracking_error"] = te
                view.set_figure(charts.te_decomposition_bars(d, lbl))
        sec = run["sectors"]
        if not sec.empty:
            st = pd.DataFrame({"portfolio": sec["after"], "benchmark": sec["before"]})
            fig = charts.sector_bars(st, "Sector weights: after (blue) vs before (grey)")
            fig.data[0].name, fig.data[1].name = "After", "Before"
            self.sec_chart.set_figure(fig)
        self.refresh_runs()

    def _trade_selected(self, row: dict) -> None:
        self.explain.set_text(f"{row['side']} {row['quantity']:g} {row['symbol']} · {row.get('wash_status', '')}\n{row.get('wash_explanation', '')}")

    # ------------------------------------------------------------------ sweeps
    def run_frontier(self) -> None:
        if not self._preflight():
            return
        self._busy(True, "Sweeping TE budgets (12 optimisations)…")
        run_task(self.hs.frontier, self.ctx.current_entity_id, self.config(), on_done=self._frontier_done, on_error=self._fail, wants_progress=False)

    def _frontier_done(self, df: pd.DataFrame) -> None:
        self.frontier_df = df
        self._busy(False, "Frontier complete.")
        self.front_table.set_frame(df)
        cur = self.result.summary if self.result else None
        self.front_chart.set_figure(charts.frontier_chart(df, cur))
        self.tabs.setCurrentIndex(4)

    def run_priorities(self) -> None:
        if not self._preflight():
            return
        self._busy(True, "Comparing all 6 constraint orderings…")
        run_task(self.hs.priority_comparison, self.ctx.current_entity_id, self.config(), on_done=self._prio_done, on_error=self._fail, wants_progress=False)

    def _prio_done(self, out) -> None:
        table, _ = out
        self.prio_table_df = table
        self._busy(False, "Priority comparison complete.")
        self.prio_table.set_frame(table)
        self.prio_chart.set_figure(charts.priority_chart(table))
        self.tabs.setCurrentIndex(5)

    # ------------------------------------------------------------------ history & acting
    def refresh(self) -> None:
        self.refresh_runs()
        self._refresh_benchmarks()

    def _refresh_benchmarks(self) -> None:
        cur = self.bench.currentText().strip() or self.app.risk_service.benchmark_name()
        names = ["S&P 500", "SPY", "IVV", "VOO", "VTI", "Russell 1000", "S&P Composite 1500"]
        try:
            bk = self.ctx.baskets.list()
            names += [f"basket:{n}" for n in (bk["name"].tolist() if not bk.empty else [])]
        except Exception:
            pass
        self.bench.blockSignals(True)
        self.bench.clear()
        self.bench.addItems(names)
        self.bench.setCurrentText(cur)
        self.bench.blockSignals(False)

    def refresh_runs(self) -> None:
        df = self.ctx.runs.list(limit=200)
        if df.empty:
            self.runs.set_frame(pd.DataFrame())
            return
        rows = []
        for _, r in df.iterrows():
            s, p = r["summary"], r["params"]
            rows.append({"id": r["id"], "created_at": r["created_at"], "run_type": r["run_type"], "as_of_date": r["as_of_date"],
                         "harvested_loss": s.get("harvested_loss"), "tax_alpha": s.get("tax_alpha"), "te_before": s.get("te_before"),
                         "te_after": s.get("te_after"), "n_trades": (s.get("n_sells", 0) or 0) + (s.get("n_buys", 0) or 0),
                         "mode": p.get("mode"), "priority": " > ".join(PRIO_LABEL.get(x, x) for x in p.get("priority", []))})
        self.runs.set_frame(pd.DataFrame(rows))

    def _load_run(self, row: dict) -> None:
        if row.get("run_type") not in ("harvest", "ai_plan"):
            return
        run = self.hs.load_run(int(row["id"]))
        if run:
            self.run_id = int(row["id"])
            self.result = None
            self._show_result(run["summary"], run)
            self.tabs.setCurrentIndex(0)

    def _mark_acted(self) -> None:
        ids = [int(r["id"]) for r in self.trades.selected_rows() if "id" in r]
        if ids:
            self.hs.mark_acted(ids, True)
            self._show_result(self.hs.load_run(self.run_id)["summary"], self.hs.load_run(self.run_id))

    def _book(self) -> None:
        rows = self.trades.selected_rows()
        ids = [int(r["id"]) for r in rows if "id" in r and not r.get("acted_on")]
        if not ids:
            return
        if QMessageBox.question(self, "Book as executed (paper)",
                                f"Record {len(ids)} trade(s) as executed at their estimated prices? This updates lots and wash-sale state. "
                                "No orders are sent anywhere.") != QMessageBox.Yes:
            return
        try:
            done = self.hs.book_trades(self.run_id, ids)
        except Exception as e:
            self.app.error(str(e))
            return
        self.app.status("Booked: " + ", ".join(done))
        self._show_result(self.hs.load_run(self.run_id)["summary"], self.hs.load_run(self.run_id))
        self.data_changed.emit()
