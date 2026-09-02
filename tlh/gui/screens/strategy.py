"""Strategy lab: build model portfolios with any construction strategy and backtest them walk-forward."""
from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
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

from ...optim.backtest import BacktestSpec
from ...optim.strategies import STRATEGIES
from ...services.strategy_service import StrategyService, spec_from_params
from .. import charts, theme
from ..widgets import FrameTable, KpiCard, TextPanel, button, header, pct, vbox
from ..workers import run_task

BT_COLS = ["id", "created_at", "name", "strategy", "cagr", "bench_cagr", "sharpe", "max_drawdown", "tracking_error", "information_ratio", "annual_turnover", "n_rebalances"]


def equity_chart(eq: pd.DataFrame, title: str = "Growth of $1 (log scale)") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq["equity"], name="Strategy", line=dict(color=theme.ACCENT))
    fig.add_scatter(x=eq.index, y=eq["benchmark"], name="Benchmark", line=dict(color=theme.MUTED))
    fig.update_layout(title=title, yaxis=dict(type="log", gridcolor=theme.BORDER), height=340)
    return fig


def drawdown_chart(eq: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color, name in (("equity", theme.RED, "Strategy"), ("benchmark", theme.MUTED, "Benchmark")):
        dd = eq[col] / eq[col].cummax() - 1
        fig.add_scatter(x=eq.index, y=dd * 100, name=name, line=dict(color=color), fill="tozeroy" if col == "equity" else None)
    fig.update_layout(title="Drawdown (%)", height=260)
    return fig


def weights_history_chart(W: pd.DataFrame, top: int = 20) -> go.Figure:
    if W.empty:
        return go.Figure()
    cols = W.mean().sort_values(ascending=False).index[:top]
    fig = go.Figure()
    for c in cols:
        fig.add_scatter(x=W.index, y=W[c] * 100, name=c, stackgroup="one", mode="lines", line=dict(width=0.5))
    fig.update_layout(title=f"Weights over time (top {top} names, stacked %)", height=340, showlegend=True)
    return fig


def active_return_chart(eq: pd.DataFrame) -> go.Figure:
    r = eq.pct_change().dropna()
    act = (r["equity"] - r["benchmark"]).resample("ME").sum() * 100
    fig = go.Figure(go.Bar(x=act.index, y=act, marker_color=[theme.GREEN if v >= 0 else theme.RED for v in act]))
    fig.update_layout(title="Monthly active return (pp)", height=260)
    return fig


class StrategyScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.svc = StrategyService(self.ctx)
        self._last_bt = None
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([380, 1100])

    def _left(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(header("Strategy lab", "Institutional construction methods on the risk model's covariance, saved as baskets; walk-forward backtests with costs."))
        g = QGroupBox("Strategy")
        f = QFormLayout(g)
        self.kind = QComboBox()
        for k in STRATEGIES:
            self.kind.addItem(k, k)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.desc = QLabel("")
        self.desc.setWordWrap(True)
        self.desc.setProperty("muted", True)
        self.name = QLineEdit()
        self.name.setPlaceholderText("basket name (leave blank to not save)")
        self.bench = QComboBox()
        self.bench.setEditable(True)
        self.cov_src = QComboBox()
        self.cov_src.addItems(["model", "sample"])
        self.n_max = QSpinBox()
        self.n_max.setRange(0, 1000)
        self.n_max.setValue(50)
        self.n_max.setSpecialValueText("no cap")
        self.max_w = QDoubleSpinBox()
        self.max_w.setRange(0.1, 100)
        self.max_w.setValue(8)
        self.max_w.setSuffix(" %")
        self.band = QDoubleSpinBox()
        self.band.setRange(0, 100)
        self.band.setValue(0)
        self.band.setSuffix(" %")
        self.band.setSpecialValueText("off")
        self.exclude = QLineEdit()
        self.exclude.setPlaceholderText("exclude symbols, comma-separated")
        f.addRow("Method", self.kind)
        f.addRow(self.desc)
        f.addRow("Save as basket", self.name)
        f.addRow("Benchmark", self.bench)
        f.addRow("Covariance", self.cov_src)
        f.addRow("Max names", self.n_max)
        f.addRow("Max weight", self.max_w)
        f.addRow("Sector band", self.band)
        f.addRow("Exclude", self.exclude)
        lay.addWidget(g)

        self.g_mv = QGroupBox("Alpha / mean-variance")
        f = QFormLayout(self.g_mv)
        self.signals = QLineEdit("momentum=1, quality=0.5")
        self.ic = QDoubleSpinBox()
        self.ic.setRange(0.001, 1)
        self.ic.setDecimals(3)
        self.ic.setValue(0.05)
        self.ra = QDoubleSpinBox()
        self.ra.setRange(0, 100)
        self.ra.setValue(5)
        self.bench_rel = QCheckBox("benchmark-relative risk")
        self.bench_rel.setChecked(True)
        f.addRow("Signal weights", self.signals)
        f.addRow("IC", self.ic)
        f.addRow("Risk aversion", self.ra)
        f.addRow(self.bench_rel)
        lay.addWidget(self.g_mv)

        self.g_bl = QGroupBox("Black-Litterman views (JSON list)")
        f = QFormLayout(self.g_bl)
        self.views = QPlainTextEdit('[{"assets": {"AAPL": 1, "MSFT": -1}, "return": 0.03, "confidence": 0.5}]')
        self.views.setMaximumHeight(80)
        self.tau = QDoubleSpinBox()
        self.tau.setRange(0.001, 1)
        self.tau.setDecimals(3)
        self.tau.setValue(0.05)
        f.addRow(self.views)
        f.addRow("tau", self.tau)
        lay.addWidget(self.g_bl)

        self.g_cvar = QGroupBox("CVaR")
        f = QFormLayout(self.g_cvar)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.8, 0.999)
        self.alpha.setDecimals(3)
        self.alpha.setValue(0.95)
        f.addRow("Confidence", self.alpha)
        lay.addWidget(self.g_cvar)

        self.g_tilt = QGroupBox("Factor tilts")
        f = QFormLayout(self.g_tilt)
        self.tilts = QLineEdit("quality=0.3, lowvol=0.2")
        f.addRow("Target active z", self.tilts)
        lay.addWidget(self.g_tilt)

        self.g_tr = QGroupBox("Tax-aware transition")
        f = QFormLayout(self.g_tr)
        self.target = QComboBox()
        self.gain_budget = QDoubleSpinBox()
        self.gain_budget.setRange(-100, 100)
        self.gain_budget.setDecimals(2)
        self.gain_budget.setValue(1.0)
        self.gain_budget.setSuffix(" % of value")
        self.turn_max = QDoubleSpinBox()
        self.turn_max.setRange(1, 100)
        self.turn_max.setValue(50)
        self.turn_max.setSuffix(" %")
        f.addRow("Target basket", self.target)
        f.addRow("Net realised-gain budget", self.gain_budget)
        f.addRow("Turnover cap", self.turn_max)
        lay.addWidget(self.g_tr)

        self.build_btn = button("Build basket", self.build, primary=True)
        lay.addWidget(self.build_btn)

        g = QGroupBox("Backtest")
        f = QFormLayout(g)
        self.bt_start = QLineEdit("")
        self.bt_start.setPlaceholderText("YYYY-MM-DD (default: earliest)")
        self.bt_end = QLineEdit("")
        self.bt_end.setPlaceholderText("YYYY-MM-DD (default: latest)")
        self.bt_freq = QComboBox()
        self.bt_freq.addItems(["M", "Q", "W"])
        self.bt_look = QSpinBox()
        self.bt_look.setRange(60, 1500)
        self.bt_look.setValue(252)
        self.bt_cost = QDoubleSpinBox()
        self.bt_cost.setRange(0, 100)
        self.bt_cost.setValue(5)
        self.bt_cost.setSuffix(" bps")
        self.bt_bench = QLineEdit("SPY")
        self.bt_bench.setPlaceholderText("ETF ticker or blank for cap-weighted proxy")
        self.bt_member = QCheckBox("Use point-in-time index membership when available")
        self.bt_member.setChecked(True)
        f.addRow("Start", self.bt_start)
        f.addRow("End", self.bt_end)
        f.addRow("Rebalance", self.bt_freq)
        f.addRow("Lookback (days)", self.bt_look)
        f.addRow("Cost", self.bt_cost)
        f.addRow("Benchmark ETF", self.bt_bench)
        f.addRow(self.bt_member)
        self.bt_btn = button("Run backtest", self.backtest)
        f.addRow(self.bt_btn)
        lay.addWidget(g)
        lay.addStretch(1)
        scroll.setWidget(w)
        self._kind_changed()
        return scroll

    def _right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 0, 0)
        k = QHBoxLayout()
        self.k_te = KpiCard("TE vs benchmark")
        self.k_vol = KpiCard("Volatility")
        self.k_n = KpiCard("Names")
        self.k_cagr = KpiCard("Backtest CAGR")
        self.k_sharpe = KpiCard("Sharpe")
        self.k_dd = KpiCard("Max drawdown")
        self.k_ir = KpiCard("Information ratio")
        for c in (self.k_te, self.k_vol, self.k_n, self.k_cagr, self.k_sharpe, self.k_dd, self.k_ir):
            k.addWidget(c)
        lay.addLayout(k)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.weights = FrameTable(["symbol", "weight", "benchmark_weight", "active"], pct_cols={"weight", "benchmark_weight", "active"})
        self.diag = TextPanel("Diagnostics")
        wsplit = QSplitter(Qt.Vertical)
        wsplit.addWidget(self.weights)
        wsplit.addWidget(self.diag)
        wsplit.setSizes([500, 160])
        self.tabs.addTab(wsplit, "Weights")
        self.eq_chart = charts.PlotlyView()
        self.dd_chart = charts.PlotlyView()
        col = QSplitter(Qt.Vertical)
        col.addWidget(self.eq_chart)
        col.addWidget(self.dd_chart)
        self.tabs.addTab(col, "Backtest: equity")
        self.wh_chart = charts.PlotlyView()
        self.ar_chart = charts.PlotlyView()
        col2 = QSplitter(Qt.Vertical)
        col2.addWidget(self.wh_chart)
        col2.addWidget(self.ar_chart)
        self.tabs.addTab(col2, "Backtest: weights & active")
        self.metrics = FrameTable(["metric", "strategy", "benchmark"], filter_box=False)
        self.warn = TextPanel("Caveats")
        msplit = QSplitter(Qt.Vertical)
        msplit.addWidget(self.metrics)
        msplit.addWidget(self.warn)
        self.tabs.addTab(msplit, "Backtest: metrics")
        self.history = FrameTable(BT_COLS)
        self.history.row_activated.connect(self._load_bt)
        self.tabs.addTab(vbox(QLabel("Double-click a backtest to load it."), self.history), "Backtest history")
        return w

    # ------------------------------------------------------------------ helpers
    def _kind_changed(self) -> None:
        k = self.kind.currentData()
        self.desc.setText(STRATEGIES.get(k, ""))
        self.g_mv.setVisible(k in ("mean_variance", "black_litterman"))
        self.g_bl.setVisible(k == "black_litterman")
        self.g_cvar.setVisible(k == "min_cvar")
        self.g_tilt.setVisible(k == "factor_tilt")
        self.g_tr.setVisible(k == "tax_aware_transition")

    @staticmethod
    def _kv(text: str) -> dict:
        out = {}
        for part in text.split(","):
            if "=" in part:
                a, b = part.split("=", 1)
                try:
                    out[a.strip()] = float(b)
                except ValueError:
                    pass
        return out

    def params(self) -> dict:
        p = {"n_max": self.n_max.value() or None, "max_weight": self.max_w.value() / 100,
             "sector_band": (self.band.value() / 100) if self.band.value() > 0 else None,
             "exclude": [s.strip().upper() for s in self.exclude.text().split(",") if s.strip()],
             "signal_weights": self._kv(self.signals.text()) or {"momentum": 1.0}, "ic": self.ic.value(), "risk_aversion": self.ra.value(),
             "benchmark_relative": self.bench_rel.isChecked(), "tau": self.tau.value(), "cvar_alpha": self.alpha.value(),
             "tilts": self._kv(self.tilts.text()), "gain_budget": self.gain_budget.value() / 100, "turnover_max": self.turn_max.value() / 100}
        if self.kind.currentData() == "black_litterman":
            try:
                p["views"] = json.loads(self.views.toPlainText() or "[]")
            except json.JSONDecodeError as e:
                raise ValueError(f"views JSON invalid: {e}") from e
        return p

    def refresh(self) -> None:
        bk = self.ctx.baskets.list()
        names = [f"basket:{n}" for n in (bk["name"].tolist() if not bk.empty else [])]
        for combo, base in ((self.bench, ["S&P 500", "SPY", "VTI", "Russell 1000"]), (self.target, [])):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(base + names)
            if cur:
                combo.setCurrentText(cur)
            combo.blockSignals(False)
        if not self.bench.currentText():
            self.bench.setCurrentText(self.app.risk_service.benchmark_name())
        self.history.set_frame(self.svc.list_backtests())

    # ------------------------------------------------------------------ actions
    def build(self) -> None:
        if self.app.risk_service.active() is None:
            QMessageBox.information(self, "No risk model", "Fit a risk model first.")
            return
        try:
            spec = spec_from_params(self.kind.currentData(), self.params())
            if spec.kind == "tax_aware_transition":
                spec = self.svc.resolve_target(spec, self.target.currentText())
        except Exception as e:
            QMessageBox.warning(self, "Parameters", str(e))
            return
        name = self.name.text().strip() or None
        self.build_btn.setEnabled(False)
        self.app.status(f"Building {spec.kind}…")
        run_task(self.svc.build, name, spec, self.bench.currentText().strip() or None, None, self.ctx.current_entity_id, None,
                 bool(name), self.cov_src.currentText(), on_done=self._built, on_error=self._fail, wants_progress=False)

    def _built(self, out: dict) -> None:
        self.build_btn.setEnabled(True)
        d = out.get("diagnostics", {})
        self.k_te.set(pct(out.get("tracking_error_model", d.get("tracking_error"))), f"vs {out.get('benchmark')}")
        self.k_vol.set(pct(d.get("volatility")), "annualised")
        self.k_n.set(str(out.get("n_names")), out.get("status", ""))
        w = pd.Series(out.get("top_weights", {}))
        full = None
        if out.get("saved_basket"):
            b = self.ctx.baskets.get(out["saved_basket"])
            full = b["weights"] if b else None
        ws = full if full is not None else w
        snap = self.app.data_service.latest_snapshot()
        act = self.app.risk_service.active()
        bench = self.app.risk_service.benchmark_weights(snap, act[1], out.get("benchmark")) if (snap is not None and act) else pd.Series(dtype=float)
        df = ws.rename("weight").to_frame()
        df["benchmark_weight"] = bench.reindex(df.index).fillna(0.0)
        df["active"] = df["weight"] - df["benchmark_weight"]
        self.weights.set_frame(df.reset_index().rename(columns={"index": "symbol"}))
        lines = [f"{k}: {v}" for k, v in d.items() if not isinstance(v, dict)]
        lines += [f"active style: {out.get('active_style_exposures')}", f"sector active (pp): {out.get('sector_active_pp')}"]
        self.diag.set_text("\n".join(lines))
        self.tabs.setCurrentIndex(0)
        self.app.status(f"Built {out['strategy']}: {out.get('n_names')} names, TE {pct(out.get('tracking_error_model', d.get('tracking_error')))}"
                        + (f" · saved as basket '{out['saved_basket']}'" if out.get("saved_basket") else ""))
        if out.get("saved_basket"):
            self.data_changed.emit()

    def _fail(self, msg: str) -> None:
        self.build_btn.setEnabled(True)
        self.bt_btn.setEnabled(True)
        self.app.error(msg)

    def backtest(self) -> None:
        try:
            spec = spec_from_params(self.kind.currentData(), self.params())
            if spec.kind == "tax_aware_transition":
                QMessageBox.information(self, "Backtest", "Tax-aware transition is a one-off rebalance, not a repeatable strategy; backtest another method.")
                return
        except Exception as e:
            QMessageBox.warning(self, "Parameters", str(e))
            return
        bspec = BacktestSpec(start=self.bt_start.text().strip() or None, end=self.bt_end.text().strip() or None, rebalance=self.bt_freq.currentText(),
                             lookback_days=self.bt_look.value(), cost_bps=self.bt_cost.value(),
                             benchmark_symbol=self.bt_bench.text().strip().upper() or None, use_membership=self.bt_member.isChecked())
        self.bt_btn.setEnabled(False)
        self.app.status("Backtesting…")
        run_task(self.svc.backtest, spec, bspec, self.name.text().strip() or f"{spec.kind} backtest", self.ctx.current_entity_id,
                 on_done=self._bt_done, on_error=self._fail, on_progress=self.app.status)

    def _bt_done(self, out) -> None:
        self.bt_btn.setEnabled(True)
        rid, res = out
        self._show_bt(pd.DataFrame({"equity": res.equity, "benchmark": res.bench_equity}), res.weights, res.metrics, res.warnings, asdict(res.strategy))
        self.refresh()
        self.app.status(f"Backtest run #{rid}: CAGR {res.metrics['cagr']:.2%} vs {res.metrics['bench_cagr']:.2%}, IR {res.metrics['information_ratio']:.2f}")
        self.tabs.setCurrentIndex(1)

    def _show_bt(self, eq: pd.DataFrame, W: pd.DataFrame, m: dict, warns: list, strat: dict) -> None:
        self.k_cagr.set(pct(m.get("cagr")), f"benchmark {pct(m.get('bench_cagr'))}")
        self.k_sharpe.set(f"{m.get('sharpe', 0):.2f}", f"benchmark {m.get('bench_sharpe', 0):.2f}")
        self.k_dd.set(pct(m.get("max_drawdown")), f"benchmark {pct(m.get('bench_max_drawdown'))}")
        self.k_ir.set(f"{m.get('information_ratio', 0):.2f}", f"TE {pct(m.get('tracking_error'))}")
        self.eq_chart.set_figure(equity_chart(eq, f"{strat.get('kind')} · growth of $1"))
        self.dd_chart.set_figure(drawdown_chart(eq))
        self.wh_chart.set_figure(weights_history_chart(W))
        self.ar_chart.set_figure(active_return_chart(eq))
        pairs = [("CAGR", "cagr", "bench_cagr"), ("Total return", "total_return", "bench_total_return"), ("Ann. vol", "ann_vol", "bench_ann_vol"),
                 ("Sharpe", "sharpe", "bench_sharpe"), ("Max drawdown", "max_drawdown", "bench_max_drawdown")]
        rows = [{"metric": lbl, "strategy": m.get(a), "benchmark": m.get(b)} for lbl, a, b in pairs]
        rows += [{"metric": k, "strategy": m.get(k), "benchmark": None} for k in ("tracking_error", "information_ratio", "active_return_ann",
                                                                                   "annual_turnover", "avg_turnover_per_rebalance", "avg_names", "hit_rate_monthly", "n_rebalances", "start", "end")]
        self.metrics.set_frame(pd.DataFrame(rows))
        self.warn.set_text("\n".join(warns) if warns else "none")

    def _load_bt(self, row: dict) -> None:
        bt = self.svc.load_backtest(int(row["id"]))
        if not bt or bt["equity"].empty:
            return
        s = bt["summary"]
        self._show_bt(bt["equity"], bt["weights"], s, s.get("warnings", []), bt["params"].get("strategy", {}))
        self.tabs.setCurrentIndex(1)
