"""Risk lab: decomposition, factor stress tests, historical scenarios, VaR, estimator diagnostics, bias-test validation."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...risk.analytics import PRESET_SHOCKS
from .. import charts, theme
from ..widgets import FrameTable, KpiCard, TextPanel, button, header, pct
from ..workers import run_task


def heatmap(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index), zmin=-1, zmax=1,
                               colorscale=[[0, theme.RED], [0.5, theme.BG3], [1, theme.GREEN]], hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title=title, height=max(320, 18 * len(df) + 120), xaxis=dict(tickangle=-45))
    return fig


def group_bars(groups_pct: dict, title: str) -> go.Figure:
    s = pd.Series(groups_pct).sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=s.index, y=s.values * 100, marker_color=[theme.ACCENT, theme.AMBER, theme.PURPLE, theme.GREEN, theme.MUTED][: len(s)]))
    fig.update_layout(title=title, yaxis_title="% of variance", height=300)
    return fig


def contrib_bars(contrib: dict, title: str, unit: str = "bps") -> go.Figure:
    s = pd.Series(contrib)
    s = s[s.abs() > 0].sort_values()
    scale = 1e4 if unit == "bps" else 100
    fig = go.Figure(go.Bar(x=s.values * scale, y=s.index, orientation="h", marker_color=[theme.RED if v < 0 else theme.ACCENT for v in s.values]))
    fig.update_layout(title=title, xaxis_title=unit, height=max(300, 20 * len(s) + 100))
    return fig


def r2_chart(series: list) -> go.Figure:
    fig = go.Figure()
    if series:
        df = pd.DataFrame(series, columns=["date", "r2"])
        fig.add_scatter(x=pd.to_datetime(df["date"]), y=df["r2"], mode="lines", line=dict(color=theme.ACCENT), name="R²")
        fig.add_scatter(x=pd.to_datetime(df["date"]), y=df["r2"].rolling(12, min_periods=1).mean(), mode="lines", line=dict(color=theme.AMBER), name="rolling mean")
    fig.update_layout(title="Cross-sectional R² by regression date", height=300)
    return fig


class RiskLabScreen(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.rs = app.risk_service
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        top.addWidget(header("Risk lab", "Where does my risk come from, what happens under factor shocks, and can the model be trusted? All on the active model and current holdings."))
        top.addStretch(1)
        self.active = QCheckBox("active (vs benchmark)")
        self.active.setChecked(True)
        top.addWidget(self.active)
        top.addWidget(button("Recompute", self.refresh))
        root.addLayout(top)
        k = QHBoxLayout()
        self.k_sigma = KpiCard("Risk")
        self.k_factor = KpiCard("Factor share")
        self.k_var = KpiCard("VaR 99% · 1m")
        self.k_es = KpiCard("Expected shortfall")
        self.k_top = KpiCard("Largest contributor")
        for c in (self.k_sigma, self.k_factor, self.k_var, self.k_es, self.k_top):
            k.addWidget(c)
        root.addLayout(k)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # decomposition
        self.grp_chart = charts.PlotlyView()
        self.fac_chart = charts.PlotlyView()
        self.hold_table = FrameTable(["symbol", "weight", "mctr", "ctr", "pct_of_risk"], pct_cols={"weight", "pct_of_risk"})
        row = QSplitter(Qt.Horizontal)
        row.addWidget(self.grp_chart)
        row.addWidget(self.fac_chart)
        col = QSplitter(Qt.Vertical)
        col.addWidget(row)
        col.addWidget(self.hold_table)
        self.tabs.addTab(col, "Decomposition")

        # stress
        st = QWidget()
        sl = QHBoxLayout(st)
        left = QGroupBox("Shocks (factor = sigma units; add ':raw' for a return)")
        lf = QFormLayout(left)
        self.preset = QComboBox()
        self.preset.addItem("Presets…")
        self.preset.addItems(list(PRESET_SHOCKS))
        self.preset.currentIndexChanged.connect(self._preset)
        self.shocks = QPlainTextEdit("market = -2\nmomentum = -1")
        self.shocks.setMaximumHeight(140)
        self.propagate = QCheckBox("propagate to correlated factors")
        self.propagate.setChecked(True)
        lf.addRow(self.preset)
        lf.addRow(self.shocks)
        lf.addRow(self.propagate)
        lf.addRow(button("Run stress test", self.run_stress, primary=True))
        self.hist_start = QLineEdit("2025-01-01")
        self.hist_end = QLineEdit("2025-03-31")
        lf.addRow(QLabel("Historical replay of the model's factor returns:"))
        lf.addRow("Start", self.hist_start)
        lf.addRow("End", self.hist_end)
        lf.addRow(button("Replay window", self.run_scenario))
        self.stress_kpi = TextPanel("Result")
        lf.addRow(self.stress_kpi)
        left.setMaximumWidth(420)
        sl.addWidget(left)
        self.stress_fac = charts.PlotlyView()
        self.stress_hold = FrameTable(["symbol", "weight", "shock_return", "contribution"], pct_cols={"weight", "shock_return", "contribution"})
        rcol = QSplitter(Qt.Vertical)
        rcol.addWidget(self.stress_fac)
        rcol.addWidget(self.stress_hold)
        sl.addWidget(rcol, 1)
        self.tabs.addTab(st, "Stress tests")

        # diagnostics
        self.tstat = FrameTable(["factor", "vol", "mean_abs_t", "pct_significant"], pct_cols={"vol", "pct_significant"})
        self.r2 = charts.PlotlyView()
        self.fcorr = charts.PlotlyView()
        self.xcorr = charts.PlotlyView()
        self.coverage = FrameTable(["descriptor", "coverage"])
        d1 = QSplitter(Qt.Horizontal)
        d1.addWidget(self.tstat)
        d1.addWidget(self.r2)
        d2 = QSplitter(Qt.Horizontal)
        d2.addWidget(self.fcorr)
        d2.addWidget(self.xcorr)
        d3 = QSplitter(Qt.Vertical)
        d3.addWidget(d1)
        d3.addWidget(d2)
        d3.addWidget(self.coverage)
        d3.setSizes([300, 400, 160])
        self.tabs.addTab(d3, "Estimator diagnostics")

        # validation
        v = QWidget()
        vl = QVBoxLayout(v)
        form = QHBoxLayout()
        self.n_periods = QSpinBox()
        self.n_periods.setRange(2, 36)
        self.n_periods.setValue(6)
        self.period_days = QSpinBox()
        self.period_days.setRange(5, 63)
        self.period_days.setValue(21)
        form.addWidget(QLabel("Periods"))
        form.addWidget(self.n_periods)
        form.addWidget(QLabel("Days per period"))
        form.addWidget(self.period_days)
        self.bias_btn = button("Run out-of-sample bias test (refits the model once per period)", self.run_bias, primary=True)
        form.addWidget(self.bias_btn)
        form.addStretch(1)
        vl.addLayout(form)
        note = QLabel("Bias statistic = std(realised return / predicted vol) across periods. ≈1 is calibrated; >1 the model under-forecasts risk; "
                      "<1 it over-forecasts. Band = 1 ± √(2/n).")
        note.setWordWrap(True)
        note.setProperty("muted", True)
        vl.addWidget(note)
        self.bias_summary = FrameTable(["portfolio", "n", "bias_stat", "band_low", "band_high", "mean_z", "verdict"])
        self.bias_detail = FrameTable(["period_end", "portfolio", "predicted_vol", "realised_return", "z"], pct_cols={"predicted_vol", "realised_return"})
        vs = QSplitter(Qt.Vertical)
        vs.addWidget(self.bias_summary)
        vs.addWidget(self.bias_detail)
        vl.addWidget(vs, 1)
        self.tabs.addTab(v, "Validation")

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        if self.rs.active() is None:
            self.k_sigma.set("—", "fit a model")
            return
        run_task(self._load, self.active.isChecked(), on_done=self._loaded, on_error=lambda m: self.app.status(m.splitlines()[0][:160]), wants_progress=False)

    def _load(self, active: bool) -> dict:
        out = {"dec": self.rs.decomposition(self.ctx.current_entity_id, active=active),
               "var": self.rs.var(self.ctx.current_entity_id, 21, 0.99, active=active)}
        act = self.rs.active()
        out["model"] = act[1]
        return out

    def _loaded(self, d: dict) -> None:
        dec, var, m = d["dec"], d["var"], d["model"]
        lbl = "active" if dec["is_active"] else "total"
        self.k_sigma.set(pct(dec["sigma"]), f"{lbl} annualised")
        self.k_factor.set(pct(dec["pct_factor"]), "of variance from factors")
        self.k_var.set(pct(var["var"]), f"{lbl} · parametric")
        self.k_es.set(pct(var["es"]), "99% · 1 month")
        h = dec["holdings"]
        if not h.empty:
            top = h.iloc[0]
            self.k_top.set(str(h.index[0]), f"{top['pct_of_risk']:.0%} of risk")
        self.grp_chart.set_figure(group_bars(dec["groups_pct"], f"{lbl.title()} risk by factor group"))
        self.fac_chart.set_figure(contrib_bars(dec["factor_contrib_sigma"], "Factor contributions to risk (bps of σ)"))
        self.hold_table.set_frame(h.reset_index().rename(columns={"index": "symbol"}))
        dg = m.diagnostics
        vols = m.factor_vols()
        ts = dg.get("t_stats", {})
        rows = [{"factor": f, "vol": float(vols.get(f, float("nan"))), "mean_abs_t": (ts.get(f) or {}).get("mean_abs_t"),
                 "pct_significant": (ts.get(f) or {}).get("pct_significant")} for f in m.factors]
        self.tstat.set_frame(pd.DataFrame(rows))
        self.r2.set_figure(r2_chart(dg.get("r2_series", [])))
        F = m.factor_cov
        sd = pd.Series(F.values.diagonal() ** 0.5, index=F.index)
        corr = F.div(sd, axis=0).div(sd, axis=1)
        keep = [f for f in corr.index if not str(f).startswith(("sec:", "ind:"))]
        self.fcorr.set_figure(heatmap(corr.loc[keep, keep].round(2), "Factor return correlation (market, styles, macro)"))
        xc = dg.get("style_exposure_corr")
        if xc:
            self.xcorr.set_figure(heatmap(pd.DataFrame(xc), "Style exposure correlation (cross-section)"))
        else:
            styles = [c for c in m.factors if c in m.spec.styles]
            if styles:
                self.xcorr.set_figure(heatmap(m.exposures[styles].corr().round(2), "Style exposure correlation (cross-section)"))
        cov = dg.get("descriptor_coverage", {})
        self.coverage.set_frame(pd.DataFrame([{"descriptor": k, "coverage": v} for k, v in cov.items()]) if cov else pd.DataFrame())

    # ------------------------------------------------------------------ stress
    def _preset(self, i: int) -> None:
        if i <= 0:
            return
        sh = PRESET_SHOCKS[self.preset.currentText()]
        self.shocks.setPlainText("\n".join(f"{k} = {v}" for k, v in sh.items()))
        self.preset.setCurrentIndex(0)

    def _parse_shocks(self) -> dict:
        out = {}
        for line in self.shocks.toPlainText().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    out[k.strip()] = float(v)
                except ValueError:
                    pass
        return out

    def run_stress(self) -> None:
        sh = self._parse_shocks()
        if not sh:
            return
        run_task(self.rs.stress, sh, self.ctx.current_entity_id, self.active.isChecked(), self.propagate.isChecked(),
                 on_done=self._stress_done, on_error=self.app.error, wants_progress=False)

    def _stress_done(self, d: dict) -> None:
        txt = f"Portfolio {'active ' if d['is_active'] else ''}return: {d['portfolio_return']:+.2%}"
        if d.get("ignored"):
            txt += f"\nIgnored (not in model): {d['ignored']}"
        txt += "\nShocked: " + ", ".join(f"{k} {v:+.2%}" for k, v in d["shocked"].items())
        self.stress_kpi.set_text(txt)
        self.stress_fac.set_figure(contrib_bars(d["factor_contrib"], "Factor contributions to shock return", unit="%"))
        self.stress_hold.set_frame(d["holdings"].reset_index().rename(columns={"index": "symbol"}))

    def run_scenario(self) -> None:
        run_task(self.rs.scenario, self.hist_start.text().strip(), self.hist_end.text().strip(), self.ctx.current_entity_id, self.active.isChecked(),
                 on_done=self._scenario_done, on_error=self.app.error, wants_progress=False)

    def _scenario_done(self, d: dict) -> None:
        if "error" in d:
            self.stress_kpi.set_text(f"{d['error']} (available {d['available'][0]} → {d['available'][1]})")
            return
        self.stress_kpi.set_text(f"Replay {d['start']} → {d['end']} ({d['n_days']} days): portfolio {d['portfolio_return']:+.2%} (factor part only)")
        self.stress_fac.set_figure(contrib_bars(d["factor_contrib"], "Factor contributions over the window", unit="%"))

    # ------------------------------------------------------------------ validation
    def run_bias(self) -> None:
        self.bias_btn.setEnabled(False)
        self.app.status("Running bias test…")
        run_task(self.rs.bias_test, None, self.n_periods.value(), self.period_days.value(), self.ctx.current_entity_id,
                 on_done=self._bias_done, on_error=lambda m: (self.bias_btn.setEnabled(True), self.app.error(m)), on_progress=self.app.status)

    def _bias_done(self, df: pd.DataFrame) -> None:
        self.bias_btn.setEnabled(True)
        self.bias_summary.set_frame(df)
        self.bias_detail.set_frame(df.attrs.get("detail", pd.DataFrame()))
        self.app.status("Bias test complete.")
