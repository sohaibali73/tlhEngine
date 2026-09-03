"""Tactical overlay & levered beta: Potomac signals -> target beta -> leveraged / inverse ETF overlay on the core (no futures,
no shorts, margin-aware), plus the levered-beta model builder (S&P stocks + 2x/3x ETFs + optional margin to a 1.5 beta)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...optim.leverage import DEFAULT_LONG_LEVERED, INSTRUMENTS, MarginPolicy
from ...optim.strategies import StrategySpec
from ...optim.tactical import RULES, SignalSpec
from ...services.strategy_service import StrategyService
from ...services.tactical_service import TacticalService
from .. import charts, theme
from ..widgets import FrameTable, KpiCard, TextPanel, button, hbox, header, money, pct
from ..workers import run_task


class TacticalScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.svc = TacticalService(self.ctx)
        self.strat = StrategyService(self.ctx)
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([420, 1100])

    def _left(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(header("Tactical overlay & levered beta",
                              "Potomac's tactical signals set a target beta (0 to 1.5). The overlay reaches it with leveraged or inverse S&P ETFs on "
                              "top of the tax-sensitive core, never by selling stock, never with futures or shorts, and always inside the margin policy."))
        # ---- signal
        g = QGroupBox("Signal (target beta source)")
        f = QFormLayout(g)
        self.sig_name = QLineEdit("Potomac composite")
        self.sig_kind = QComboBox()
        self.sig_kind.addItem("manual target beta", "manual")
        self.sig_kind.addItem("Potomac strategy CSV (date + target_beta / state / score)", "csv")
        for k in RULES:
            self.sig_kind.addItem(f"{k} — example rule", k)
        self.sig_kind.addItem("blend of saved signals", "blend")
        self.sig_kind.currentIndexChanged.connect(self._kind_changed)
        self.manual_slider = QSlider(Qt.Horizontal)
        self.manual_slider.setRange(0, 150)
        self.manual_slider.setValue(100)
        self.manual_lbl = QLabel("1.00")
        self.manual_slider.valueChanged.connect(lambda v: self.manual_lbl.setText(f"{v / 100:.2f}"))
        self.csv_path = QLineEdit()
        self.csv_path.setPlaceholderText("exported signal file…")
        self.beta_min = QDoubleSpinBox()
        self.beta_min.setRange(0, 3)
        self.beta_min.setSingleStep(0.1)
        self.beta_min.setValue(0.0)
        self.beta_max = QDoubleSpinBox()
        self.beta_max.setRange(0, 3)
        self.beta_max.setSingleStep(0.1)
        self.beta_max.setValue(1.5)
        self.blend = QLineEdit()
        self.blend.setPlaceholderText("name=weight, name=weight")
        f.addRow("Name", self.sig_name)
        f.addRow("Source", self.sig_kind)
        self.row_manual = (QLabel("Target beta"), hbox(self.manual_slider, self.manual_lbl))
        f.addRow(*self.row_manual)
        self.row_csv = (QLabel("CSV"), hbox(self.csv_path, button("Browse…", self._browse)))
        f.addRow(*self.row_csv)
        self.row_blend = (QLabel("Components"), self.blend)
        f.addRow(*self.row_blend)
        f.addRow("Beta range (risk-off … risk-on)", hbox(self.beta_min, self.beta_max))
        f.addRow(hbox(button("Save signal", self._save_signal, primary=True), button("Set active", self._set_active), button("Delete", self._delete, danger=True)))
        self.signals = FrameTable(["name", "kind", "latest", "mean_beta", "changes_per_year", "start", "end", "active"], filter_box=False)
        self.signals.row_selected.connect(lambda r: setattr(self, "_sel_sig", r["name"]))
        f.addRow(self.signals)
        lay.addWidget(g)
        # ---- margin policy
        g2 = QGroupBox("Margin policy (Reg-T; leveraged ETFs marginable)")
        f2 = QFormLayout(g2)
        self.m_initial = self._pct(50)
        self.m_maint = self._pct(30)
        self.m_lev = self._pct(30)
        self.m_buffer = self._pct(25)
        self.m_rate = self._pct(6.5)
        self.m_max = self._pct(50)
        self.m_allow = QCheckBox("allow borrowing (else cash-only overlays)")
        self.m_allow.setChecked(True)
        f2.addRow("Initial margin", self.m_initial)
        f2.addRow("Maintenance (stocks)", self.m_maint)
        f2.addRow("Maintenance per 1x leverage", self.m_lev)
        f2.addRow("Buffer over maintenance", self.m_buffer)
        f2.addRow("Margin rate", self.m_rate)
        f2.addRow("Max loan (% of equity)", self.m_max)
        f2.addRow(self.m_allow)
        f2.addRow(button("Save margin policy", self._save_policy))
        lay.addWidget(g2)
        # ---- levered beta model
        g3 = QGroupBox("Levered-beta model (S&P stocks + 2x/3x ETFs, no futures)")
        f3 = QFormLayout(g3)
        self.lb_name = QLineEdit("Levered Beta 1.5")
        self.lb_beta = QDoubleSpinBox()
        self.lb_beta.setRange(0.5, 3.0)
        self.lb_beta.setSingleStep(0.1)
        self.lb_beta.setValue(1.5)
        self.lb_replicate = QCheckBox("Hold every index name (full replication, lowest tracking error)")
        self.lb_replicate.setChecked(True)
        self.lb_n = QSpinBox()
        self.lb_n.setRange(10, 500)
        self.lb_n.setValue(500)
        self.lb_n.setEnabled(False)
        self.lb_replicate.toggled.connect(lambda on: self.lb_n.setEnabled(not on))
        self.lb_maxw = self._pct(10)
        self.lb_etfw = self._pct(35)
        self.lb_margin = self._pct(50)
        self.lb_inst = QLineEdit("SSO, UPRO")
        f3.addRow("Basket name", self.lb_name)
        f3.addRow("Target beta", self.lb_beta)
        f3.addRow("", self.lb_replicate)
        f3.addRow("Max stocks (sampled)", self.lb_n)
        f3.addRow("Max stock weight", self.lb_maxw)
        f3.addRow("Max per ETF", self.lb_etfw)
        f3.addRow("Max margin loan", self.lb_margin)
        f3.addRow("Leveraged ETFs", self.lb_inst)
        self.lb_btn = button("Build levered-beta basket", self._build_lb, primary=True)
        f3.addRow(self.lb_btn)
        lay.addWidget(g3)
        lay.addStretch(1)
        scroll.setWidget(w)
        return scroll

    def _right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 0, 0)
        k = QHBoxLayout()
        self.k_target = KpiCard("Target beta today")
        self.k_now = KpiCard("Beta now")
        self.k_ticket = KpiCard("Overlay ticket")
        self.k_margin = KpiCard("Margin after")
        self.k_cost = KpiCard("Annual carry")
        for c in (self.k_target, self.k_now, self.k_ticket, self.k_margin, self.k_cost):
            k.addWidget(c)
        lay.addLayout(k)
        row = QHBoxLayout()
        self.override = QDoubleSpinBox()
        self.override.setRange(0, 3)
        self.override.setSingleStep(0.1)
        self.override.setValue(1.5)
        self.use_override = QCheckBox("override signal with this beta")
        self.cash = QDoubleSpinBox()
        self.cash.setRange(0, 1e9)
        self.cash.setPrefix("$ ")
        self.cash.setDecimals(0)
        row.addWidget(self.use_override)
        row.addWidget(self.override)
        row.addWidget(QLabel("Cash available"))
        row.addWidget(self.cash)
        row.addWidget(button("Recommend overlay", self.recommend, primary=True))
        row.addWidget(button("Backtest active signal", self.backtest, success=True))
        row.addStretch(1)
        lay.addLayout(row)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.explain = TextPanel("Recommendation")
        self.tickets = FrameTable(["side", "symbol", "shares", "weight", "beta_contribution", "annual_cost", "note"], pct_cols={"weight"}, filter_box=False)
        self.cands = FrameTable(["symbol", "leverage", "beta_used", "weight", "notional", "annual_cost", "loan", "feasible", "maintenance_excess", "market_drop_to_call"],
                                pct_cols={"weight", "loan", "maintenance_excess", "market_drop_to_call"}, filter_box=False)
        rec = QSplitter(Qt.Vertical)
        rec.addWidget(self.explain)
        rec.addWidget(self.tickets)
        rec.addWidget(self.cands)
        rec.setSizes([160, 160, 220])
        self.tabs.addTab(rec, "Overlay recommendation")
        self.sig_chart = charts.PlotlyView()
        self.tabs.addTab(self.sig_chart, "Signal")
        self.bt_chart = charts.PlotlyView()
        self.bt_beta = charts.PlotlyView()
        self.bt_metrics = TextPanel("Backtest metrics")
        bt = QSplitter(Qt.Vertical)
        bt.addWidget(self.bt_chart)
        bt.addWidget(self.bt_beta)
        bt.addWidget(self.bt_metrics)
        self.tabs.addTab(bt, "Backtest")
        self.lb_out = TextPanel("Levered-beta basket")
        self.lb_table = FrameTable(["symbol", "weight", "kind"], pct_cols={"weight"})
        lb = QSplitter(Qt.Vertical)
        lb.addWidget(self.lb_out)
        lb.addWidget(self.lb_table)
        self.tabs.addTab(lb, "Levered-beta model")
        self.inst = FrameTable(["symbol", "name", "leverage", "index", "expense_ratio", "kind"], pct_cols={"expense_ratio"}, filter_box=False)
        self.tabs.addTab(self.inst, "Instruments & custodians")
        self._kind_changed()
        return w

    def _pct(self, v: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0, 200)
        s.setDecimals(2)
        s.setSuffix(" %")
        s.setValue(v)
        return s

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        pol = self.svc.policy()
        self.m_initial.setValue(pol.initial * 100)
        self.m_maint.setValue(pol.maintenance_stock * 100)
        self.m_lev.setValue(pol.maintenance_per_leverage * 100)
        self.m_buffer.setValue(pol.buffer * 100)
        self.m_rate.setValue(pol.margin_rate * 100)
        self.m_max.setValue(pol.max_loan * 100)
        self.m_allow.setChecked(pol.allow_margin)
        self.signals.set_frame(self.svc.list_signals())
        tb = self.svc.target_beta_today()
        self.k_target.set(f"{tb:.2f}" if tb is not None else "—", f"signal: {self.svc.active_name() or 'none (set one or use override)'}")
        self.inst.set_frame(self.svc.instruments())
        s = self.svc.signal()
        if s is not None:
            fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=theme.ACCENT, shape="hv"), name="target beta"))
            fig.update_layout(title=f"Target beta — {self.svc.active_name()}", yaxis_title="beta", height=340)
            self.sig_chart.set_figure(fig)

    def _kind_changed(self, *_):
        k = self.sig_kind.currentData()
        for wdg in self.row_manual:
            wdg.setVisible(k == "manual")
        for wdg in self.row_csv:
            wdg.setVisible(k == "csv")
        for wdg in self.row_blend:
            wdg.setVisible(k == "blend")

    # ------------------------------------------------------------------ signal actions
    def _browse(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Signal CSV", "", "CSV (*.csv);;All files (*)")
        if p:
            self.csv_path.setText(p)

    def _spec(self) -> SignalSpec:
        comps = []
        for part in self.blend.text().split(","):
            if "=" in part:
                n, v = part.split("=", 1)
                try:
                    comps.append({"name": n.strip(), "weight": float(v)})
                except ValueError:
                    pass
            elif part.strip():
                comps.append({"name": part.strip(), "weight": 1.0})
        return SignalSpec(name=self.sig_name.text().strip() or "signal", kind=self.sig_kind.currentData(), beta_min=self.beta_min.value(),
                          beta_max=self.beta_max.value(), manual_beta=self.manual_slider.value() / 100, path=self.csv_path.text().strip() or None,
                          components=comps)

    def _save_signal(self) -> None:
        try:
            out = self.svc.save_signal(self._spec())
        except Exception as e:
            QMessageBox.critical(self, "Signal", str(e))
            return
        self.app.status(f"Signal '{out['name']}' saved: latest target beta {out.get('latest')}.")
        self.refresh()

    def _set_active(self) -> None:
        n = getattr(self, "_sel_sig", None)
        if n:
            self.svc.set_active(n)
            self.refresh()

    def _delete(self) -> None:
        n = getattr(self, "_sel_sig", None)
        if n and QMessageBox.question(self, "Delete signal", f"Delete '{n}'?") == QMessageBox.Yes:
            self.svc.delete_signal(n)
            self.refresh()

    def _save_policy(self) -> None:
        self.svc.save_policy(MarginPolicy(initial=self.m_initial.value() / 100, maintenance_stock=self.m_maint.value() / 100,
                                          maintenance_per_leverage=self.m_lev.value() / 100, buffer=self.m_buffer.value() / 100,
                                          margin_rate=self.m_rate.value() / 100, max_loan=self.m_max.value() / 100, allow_margin=self.m_allow.isChecked()))
        self.app.status("Margin policy saved.")

    # ------------------------------------------------------------------ overlay
    def recommend(self) -> None:
        tb = self.override.value() if self.use_override.isChecked() else None
        run_task(self.svc.recommend, tb, self.cash.value(), on_done=self._rec_done, on_error=self.app.error, wants_progress=False)

    def _rec_done(self, out: dict) -> None:
        self.k_target.set(f"{out['target_beta']:.2f}", f"signal: {out.get('signal') or 'override'}")
        self.k_now.set(f"{out['beta_now']:.2f}", f"core beta {out['core']['core_beta']:.2f} on {money(out['core']['core_value'])}")
        tk = out.get("tickets") or []
        if tk:
            main = tk[-1]
            self.k_ticket.set(f"{main['side']} {main.get('shares') or ''} {main['symbol']}".strip(), f"{pct(main['weight'])} of equity → beta {out.get('beta_after', 0):.2f}")
            ch = out.get("chosen", {})
            self.k_margin.set(pct(ch.get("loan", 0.0)), f"loan · drop to call {pct(ch.get('market_drop_to_call', 1.0), 0)}", theme.RED if ch.get("loan", 0) > 0.35 else None)
            self.k_cost.set(money(out.get("annual_cost", 0.0)), "drag + fees + interest per year")
        else:
            self.k_ticket.set("none", out.get("note", ""))
            self.k_margin.set("—", "")
            self.k_cost.set("—", "")
        self.tickets.set_frame(pd.DataFrame(tk))
        tbl = out.get("table")
        self.cands.set_frame(tbl if isinstance(tbl, pd.DataFrame) else pd.DataFrame())
        txt = (f"Target beta {out['target_beta']:.2f} vs {out['beta_now']:.2f} now (gap {out['gap']:+.2f}). " + (out.get("note") or "") + "\n\n")
        if tk:
            txt += f"Ticket: {tk[-1]['side']} {tk[-1]['symbol']} for {pct(tk[-1]['weight'])} of equity ({money(tk[-1]['weight'] * out['equity'])}). "
            txt += f"Beta after {out.get('beta_after', 0):.2f}. Annual carry about {money(out.get('annual_cost', 0.0))}.\n"
        if out.get("tax_avoided_vs_selling_core"):
            txt += f"\nCutting beta by selling core stock instead would have realised roughly {money(out['tax_avoided_vs_selling_core'])} of tax; the overlay avoids it.\n"
        txt += "\nNo futures, no short sales: inverse funds reduce beta, leveraged funds raise it, within Reg-T and the house maintenance rules. " \
               "Leveraged funds decay with volatility, so overlays are meant to be held for the signal's horizon, not years. Nothing here trades."
        self.explain.set_text(txt)
        self.tabs.setCurrentIndex(0)

    def backtest(self) -> None:
        if self.svc.active_name() is None:
            QMessageBox.information(self, "Backtest", "Save and activate a signal first.")
            return
        self.app.status("Backtesting the tactical overlay…")
        run_task(self.svc.backtest, None, on_done=self._bt_done, on_error=self.app.error, wants_progress=False)

    def _bt_done(self, res: dict) -> None:
        eq, bench, core = res["equity"], res["benchmark"], res["core_only"]
        fig = go.Figure()
        fig.add_scatter(x=eq.index, y=eq.values, name="core + tactical overlay", line=dict(color=theme.GREEN, width=2))
        fig.add_scatter(x=core.index, y=core.values, name=f"core only ({res['core_source']})", line=dict(color=theme.ACCENT))
        fig.add_scatter(x=bench.index, y=bench.values, name="index", line=dict(color=theme.MUTED))
        fig.update_layout(title=f"Growth of $1 — signal {res['signal']}", height=360, yaxis_title="$")
        self.bt_chart.set_figure(fig)
        f2 = go.Figure()
        f2.add_scatter(x=res["target_beta"].index, y=res["target_beta"].values, name="target beta", line=dict(color=theme.AMBER, shape="hv"))
        f2.add_scatter(x=res["realised_beta_series"].index, y=res["realised_beta_series"].values, name="book beta", line=dict(color=theme.PURPLE))
        f2.update_layout(title="Target vs book beta", height=260)
        self.bt_beta.set_figure(f2)
        m = res["metrics"]
        self.bt_metrics.set_text(
            f"CAGR {m['cagr']:.2%} (core {m['core_cagr']:.2%}, index {m['bench_cagr']:.2%}) · vol {m['vol']:.2%} · max drawdown {m['max_drawdown']:.1%} "
            f"(index {m['bench_max_drawdown']:.1%}) · realised beta {m['realised_beta']:.2f} vs avg target {m['avg_target_beta']:.2f} · "
            f"{m['n_rebalances']} overlay rebalances, turnover {m['annual_turnover']:.1%}/yr, costs {m['total_costs']:.2%} · "
            f"overlay booked losses {m['overlay_losses_booked']:.2%} of start equity (harvestable, short-term). Simulation, not a forecast.")
        self.tabs.setCurrentIndex(2)
        self.data_changed.emit()

    # ------------------------------------------------------------------ levered beta
    def _build_lb(self) -> None:
        if self.app.risk_service.active() is None:
            QMessageBox.information(self, "No risk model", "Fit a risk model first (the leveraged ETFs must be in it: refresh data if they are not).")
            return
        inst = tuple(s.strip().upper() for s in self.lb_inst.text().split(",") if s.strip().upper() in INSTRUMENTS)
        spec = StrategySpec(kind="levered_beta", target_beta=self.lb_beta.value(), n_max=self.lb_n.value(), max_weight=self.lb_maxw.value() / 100,
                            etf_max_weight=self.lb_etfw.value() / 100, margin_max=self.lb_margin.value() / 100, lev_instruments=inst or DEFAULT_LONG_LEVERED,
                            margin_rate=self.svc.policy().margin_rate, margin_buffer=self.svc.policy().buffer, sector_band=0.03,
                            replicate=self.lb_replicate.isChecked())
        self.lb_btn.setEnabled(False)
        self.app.status("Building levered-beta basket…")
        run_task(self.strat.build, self.lb_name.text().strip() or "Levered Beta", spec, None, None, None,
                 f"Target beta {spec.target_beta:.2f} with S&P stocks + leveraged ETFs, no futures", on_done=self._lb_done, on_error=self._lb_fail, wants_progress=False)

    @staticmethod
    def _realised_text(r: dict) -> str:
        if not r or not r.get("available"):
            return "n/a (no history for every holding)"
        return (f"{r.get('te_periodic', 0):.2%} monthly-rebalanced / {r.get('te_daily', 0):.2%} daily-rebalanced over {r.get('n_days')} days "
                f"({r.get('start')} to {r.get('end')}), realised beta {r.get('beta_periodic', 0):.2f}")

    def _lb_done(self, out: dict) -> None:
        self.lb_btn.setEnabled(True)
        d = out.get("diagnostics", {})
        mg = d.get("margin", {})
        self.lb_out.set_text(
            f"{out.get('saved_basket') or out.get('name')}: beta {d.get('beta', 0):.3f} (target {d.get('target_beta')}), gross {d.get('gross', 0):.2f}x, "
            f"loan {d.get('loan', 0):.1%} of equity, {d.get('n_stocks')} stocks + ETFs {d.get('etf_weights')} (ETF betas {d.get('etf_betas')}).\n"
            f"Model TE vs {d.get('target_beta')}x index {d.get('te_vs_levered_benchmark', 0):.2%} ({d.get('replication')}); "
            f"realised TE with actual ETF histories {self._realised_text(d.get('realised', {}))}.\n"
            f"Annual cost {d.get('annual_cost', 0):.2%} "
            f"(ETF drag+fees {d.get('cost_breakdown', {}).get('etf_drag_and_fees', 0):.2%}, interest {d.get('cost_breakdown', {}).get('margin_interest', 0):.2%}).\n"
            f"Margin: equity {mg.get('equity_pct_of_mv', 1):.0%} of market value, maintenance excess {mg.get('maintenance_excess', 0):.1%}, "
            f"market drop to a call {mg.get('market_drop_to_margin_call', 1):.0%}, initial-margin ok {mg.get('initial_margin_ok')}, buffer ok {mg.get('buffer_ok')}.\n"
            f"Status {out.get('status')}. Saved as a basket: set it as the benchmark to harvest toward it. Nothing here trades.")
        tw = out.get("top_weights", {})
        b = self.ctx.baskets.get(out.get("saved_basket") or "")
        wts = b["weights"] if b else pd.Series(tw)
        df = wts.rename("weight").to_frame()
        df["kind"] = ["leveraged ETF" if s in INSTRUMENTS else "stock" for s in df.index]
        self.lb_table.set_frame(df.reset_index().rename(columns={"index": "symbol"}))
        self.tabs.setCurrentIndex(3)
        self.app.status(f"Levered-beta basket built: beta {d.get('beta', 0):.2f}, loan {d.get('loan', 0):.0%}.")
        self.data_changed.emit()

    def _lb_fail(self, msg: str) -> None:
        self.lb_btn.setEnabled(True)
        self.app.error(msg)
