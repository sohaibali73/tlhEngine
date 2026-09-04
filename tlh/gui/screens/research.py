"""TLH research: the due-diligence backtesting laboratory. Design a study (base case + sweeps + rolling windows), run it on
every core, read the summaries, curves and the concentrated-position grid, export the write-up. GUI computes nothing."""
from __future__ import annotations

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
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...research.spec import APPROACHES, ResearchSpec, StudySpec
from ...services.research_service import ResearchService
from .. import charts, theme
from ..widgets import FrameTable, KpiCard, TextPanel, button, hbox, header
from ..workers import run_task

SWEEPS = [("account_size", "Account size ($10k … $1m)"), ("basket_size", "Basket size (50 … 300 names)"), ("trigger", "Harvest trigger (0.01% … 1%)"),
          ("approach", "Harvesting approach (4)"), ("concentrated", "Concentrated start (size x gain grid, 90 combos)")]


class ResearchScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.svc = ResearchService(self.ctx)
        self._cancel = False
        self._build()
        self.refresh()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([430, 1100])

    def _left(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(header("TLH research", "Backtest the harvesting rules over rolling windows since 2000 on the point-in-time S&P 500, and defend every parameter with numbers."))

        g0 = QGroupBox("1 · Data store (Norgate, every S&P 500 member since 1999)")
        f0 = QFormLayout(g0)
        self.store_lbl = QLabel("—")
        self.store_lbl.setWordWrap(True)
        f0.addRow(self.store_lbl)
        self.build_btn = button("Build / refresh store", self._build_store)
        f0.addRow(self.build_btn)
        lay.addWidget(g0)

        g1 = QGroupBox("2 · Base case")
        f1 = QFormLayout(g1)
        self.name = QLineEdit("MVP")
        self.horizon = QComboBox()
        self.horizon.addItems(["10", "5"])
        self.account = QDoubleSpinBox()
        self.account.setRange(1_000, 100_000_000)
        self.account.setDecimals(0)
        self.account.setSingleStep(50_000)
        self.account.setValue(500_000)
        self.account.setPrefix("$")
        self.basket = QSpinBox()
        self.basket.setRange(10, 500)
        self.basket.setValue(150)
        self.trigger = QDoubleSpinBox()
        self.trigger.setRange(0.0, 5.0)
        self.trigger.setDecimals(3)
        self.trigger.setSingleStep(0.05)
        self.trigger.setValue(0.25)
        self.trigger.setSuffix(" % of account")
        self.approach = QComboBox()
        for k in APPROACHES:
            self.approach.addItem(k, k)
        self.approach.setCurrentText("optimizer")
        self.te = self._pct(2.0)
        self.sector = self._pct(2.0)
        self.factors = QCheckBox("factor alignment (size, momentum, vol, beta)")
        self.factors.setChecked(True)
        self.whole = QCheckBox("whole shares (account size matters)")
        self.whole.setChecked(True)
        f1.addRow("Study name", self.name)
        f1.addRow("Window (years)", self.horizon)
        f1.addRow("Account", self.account)
        f1.addRow("Basket size", self.basket)
        f1.addRow("Harvest trigger", self.trigger)
        f1.addRow("Approach", self.approach)
        f1.addRow("TE target", self.te)
        f1.addRow("Sector band", self.sector)
        f1.addRow("", self.factors)
        f1.addRow("", self.whole)
        lay.addWidget(g1)

        g2 = QGroupBox("3 · Sweeps and windows")
        f2 = QFormLayout(g2)
        self.sweep_boxes: dict[str, QCheckBox] = {}
        for key, label in SWEEPS:
            cb = QCheckBox(label)
            cb.setChecked(key != "concentrated")
            self.sweep_boxes[key] = cb
            f2.addRow(cb)
        self.first_year = QSpinBox()
        self.first_year.setRange(2000, 2025)
        self.first_year.setValue(2000)
        self.every = QComboBox()
        self.every.addItem("every calendar year", 1)
        self.every.addItem("every 2nd year", 2)
        self.every.addItem("every 3rd year (quick)", 3)
        f2.addRow("First start year", self.first_year)
        f2.addRow("Window starts", self.every)
        self.est_lbl = QLabel("")
        self.est_lbl.setWordWrap(True)
        f2.addRow(self.est_lbl)
        row = hbox(button("Estimate", self._estimate), button("Run study", self._run, primary=True), button("Cancel", self._cancel_run, danger=True))
        f2.addRow(row)
        lay.addWidget(g2)

        g3 = QGroupBox("4 · Single run (look inside one window)")
        f3 = QFormLayout(g3)
        self.single_year = QSpinBox()
        self.single_year.setRange(2000, 2025)
        self.single_year.setValue(2015)
        self.conc_pct = self._pct(0.0)
        self.conc_gain = self._pct(0.0)
        f3.addRow("Start year", self.single_year)
        f3.addRow("Concentrated %", self.conc_pct)
        f3.addRow("Embedded gain %", self.conc_gain)
        f3.addRow(button("Run one window", self._single))
        lay.addWidget(g3)
        lay.addStretch(1)
        scroll.setWidget(w)
        return scroll

    def _right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel("Study"))
        self.study_combo = QComboBox()
        self.study_combo.currentTextChanged.connect(lambda _t: self._load_study())
        top.addWidget(self.study_combo, 1)
        top.addWidget(button("Export write-up (xlsx + md)", self._export))
        top.addWidget(button("Refresh", self.refresh))
        lay.addLayout(top)
        kpis = QHBoxLayout()
        self.k_runs = KpiCard("Runs")
        self.k_harv = KpiCard("Base harvest / yr")
        self.k_te = KpiCard("Base TE")
        self.k_life = KpiCard("Harvest life")
        for k in (self.k_runs, self.k_harv, self.k_te, self.k_life):
            kpis.addWidget(k)
        lay.addLayout(kpis)
        self.tabs = QTabWidget()
        # summary tab
        st = QWidget()
        sl = QVBoxLayout(st)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Sweep"))
        self.sweep_combo = QComboBox()
        for key, label in [("base", "Base case")] + SWEEPS:
            self.sweep_combo.addItem(label, key)
        self.sweep_combo.currentIndexChanged.connect(lambda _i: self._show_sweep())
        r1.addWidget(self.sweep_combo, 1)
        sl.addLayout(r1)
        self.sum_table = FrameTable(pct_cols={"harvested_per_year_pct", "harvested_pct_of_start", "tax_value_pct_of_start", "te_realised", "te_forecast_avg",
                                              "excess_return_annual", "turnover_annual", "unrealised_gain_end_pct", "harvested_per_year_pct_iqr", "te_realised_iqr"})
        sl.addWidget(self.sum_table, 1)
        self.tabs.addTab(st, "Summary tables")
        # charts
        ct = QWidget()
        cl = QVBoxLayout(ct)
        self.chart_tradeoff = charts.PlotlyView()
        self.chart_curve = charts.PlotlyView()
        cl.addWidget(self.chart_tradeoff, 1)
        cl.addWidget(self.chart_curve, 1)
        self.tabs.addTab(ct, "Trade-off charts")
        # concentrated
        cc = QWidget()
        ccl = QVBoxLayout(cc)
        self.chart_conc = charts.PlotlyView()
        ccl.addWidget(self.chart_conc, 1)
        self.conc_table = FrameTable()
        ccl.addWidget(self.conc_table, 1)
        self.tabs.addTab(cc, "Concentrated grid")
        # single run
        sr = QWidget()
        srl = QVBoxLayout(sr)
        self.chart_single = charts.PlotlyView()
        srl.addWidget(self.chart_single, 2)
        self.single_out = TextPanel("Single window")
        srl.addWidget(self.single_out, 1)
        self.tabs.addTab(sr, "Single run")
        # write-up
        self.report = TextPanel("Due-diligence write-up")
        self.tabs.addTab(self.report, "Write-up")
        # all runs
        self.runs_table = FrameTable()
        self.tabs.addTab(self.runs_table, "All runs")
        lay.addWidget(self.tabs, 1)
        return w

    @staticmethod
    def _pct(v: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.0, 100.0)
        s.setDecimals(1)
        s.setSuffix(" %")
        s.setValue(v)
        return s

    # ------------------------------------------------------------------ study spec from the form
    def study_spec(self) -> StudySpec:
        base = ResearchSpec(horizon_years=int(self.horizon.currentText()), account_size=float(self.account.value()), basket_size=int(self.basket.value()),
                            trigger=self.trigger.value() / 100.0, approach=self.approach.currentData(), te_limit=self.te.value() / 100.0,
                            sector_band=self.sector.value() / 100.0, factor_alignment=self.factors.isChecked(), whole_shares=self.whole.isChecked())
        sweeps = [k for k, cb in self.sweep_boxes.items() if cb.isChecked()]
        return StudySpec(name=self.name.text().strip() or "study", base=base, sweeps=sweeps, horizons=[int(self.horizon.currentText())],
                         first_start_year=int(self.first_year.value()), every_n_years=int(self.every.currentData()))

    # ------------------------------------------------------------------ actions
    def refresh(self) -> None:
        st = self.svc.store_status()
        if st.get("ready"):
            self.store_lbl.setText(f"Ready: {st['symbols']} symbols ({st['delisted']} delisted), {st['start']} to {st['end']}, {st['members_now']} members today. Built {st.get('built', '')}.")
        else:
            self.store_lbl.setText("Not built yet. Norgate Data Updater must be running; the pull takes about a minute.")
        cur = self.study_combo.currentText()
        self.study_combo.blockSignals(True)
        self.study_combo.clear()
        for s in self.svc.list_studies():
            self.study_combo.addItem(f"{s['name']}  ({s['runs']} runs)", s["name"])
        i = next((k for k in range(self.study_combo.count()) if self.study_combo.itemText(k) == cur), -1)
        self.study_combo.setCurrentIndex(i if i >= 0 else max(self.study_combo.count() - 1, 0))
        self.study_combo.blockSignals(False)
        self._load_study()

    def _build_store(self) -> None:
        if not self.app.data_service.norgate_ok():
            QMessageBox.information(self, "Norgate", "Norgate Data Updater is not running.")
            return
        self.build_btn.setEnabled(False)
        self.app.status("Building research store from Norgate…")
        run_task(self.svc.build_store, on_done=lambda s: (self.build_btn.setEnabled(True), self.app.status(f"Research store ready: {s['symbols']} symbols."), self.refresh()),
                 on_error=lambda e: (self.build_btn.setEnabled(True), self.app.error(str(e))), on_progress=self.app.status)

    def _estimate(self) -> None:
        try:
            est = self.svc.estimate(self.study_spec())
        except Exception as e:  # noqa: BLE001
            self.est_lbl.setText(str(e))
            return
        self.est_lbl.setText(f"{est['runs']} runs ({est['optimizer_runs']} optimizer); roughly {est['approx_minutes_with_workers']} min on this machine. "
                             "Re-running a study resumes where it stopped.")

    def _run(self) -> None:
        if not self.svc.store_status().get("ready"):
            QMessageBox.information(self, "Research store", "Build the data store first (step 1).")
            return
        study = self.study_spec()
        self._cancel = False
        self.app.status(f"Running study {study.name}…")
        run_task(self.svc.run_study, study, cancel=lambda: self._cancel, on_done=self._run_done, on_error=self.app.error, on_progress=self.app.status)

    def _cancel_run(self) -> None:
        self._cancel = True
        self.app.status("Cancelling after the runs in flight finish…")

    def _run_done(self, out: dict) -> None:
        self.app.status(f"Study {out['study']} done: {out['runs']} runs ({out['failed']} failed).")
        self.refresh()
        i = self.study_combo.findData(out["study"])
        if i >= 0:
            self.study_combo.setCurrentIndex(i)
        self.data_changed.emit()

    def _single(self) -> None:
        if not self.svc.store_status().get("ready"):
            QMessageBox.information(self, "Research store", "Build the data store first (step 1).")
            return
        spec = self.study_spec().base.with_(start_year=int(self.single_year.value()), concentrated_pct=self.conc_pct.value() / 100.0,
                                             concentrated_gain=self.conc_gain.value() / 100.0)
        self.app.status("Running one window…")
        run_task(self.svc.run_single, spec, on_done=self._single_done, on_error=self.app.error, on_progress=self.app.status)

    def _single_done(self, res) -> None:
        m = res.metrics
        self.single_out.set_text(
            f"{res.spec.approach} · {m['start']} to {m['end']} · ${res.spec.account_size:,.0f} · {res.spec.basket_size} names · trigger {res.spec.trigger:.2%}\n"
            f"Harvested ${m['harvested_total']:,.0f} = {m['harvested_pct_of_start']:.1%} of start ({m['harvested_per_year_pct']:.2%}/yr; ST ${m['harvested_short_term']:,.0f}, LT ${m['harvested_long_term']:,.0f}); "
            f"tax value ${m['tax_value_of_losses']:,.0f}. Harvest life {m['harvest_life_months']} months, half-life {m['harvest_half_life_months']} months.\n"
            f"Realised TE {m['te_realised']:.2%} (forecast avg {m['te_forecast_avg']:.2%}, worst year {m['te_realised_max_year']:.2%}); excess return {m['excess_return_annual']:+.2%}/yr; "
            f"turnover {m['turnover_annual']:.0%}/yr; {m['trades']} trades; {m['wash_blocked']} harvests blocked by the wash window; names avg {m['names_avg']:.0f} (min {m['names_min']}); "
            f"ending embedded gain {m['unrealised_gain_end_pct']:.1%}."
            + (f"\nConcentrated position: diversified in {m['conc_months_to_diversify']} months; weight at end {m['conc_weight_end']:.1%}." if res.spec.concentrated_pct > 0 else "")
            + ("\n" + "; ".join(res.warnings) if res.warnings else ""))
        d = res.daily
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d["value"], name="account", line=dict(color=theme.ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=d.index, y=d["index"], name="S&P 500 TR", line=dict(color=theme.MUTED, width=1.5)))
        mo = res.monthly
        fig.add_trace(go.Bar(x=mo.index, y=mo["harvested"].cumsum(), name="cumulative harvested", yaxis="y2", marker_color=theme.GREEN, opacity=0.45))
        fig.update_layout(yaxis=dict(title="value"), yaxis2=dict(title="harvested $", overlaying="y", side="right", showgrid=False), height=420)
        self.chart_single.set_figure(fig)
        self.tabs.setCurrentIndex(3)
        self.app.status("Single window done.")

    def _export(self) -> None:
        name = self.study_combo.currentData()
        if not name:
            return
        try:
            p = self.svc.export(name)
            self.app.status(f"Exported {p}")
        except Exception as e:  # noqa: BLE001
            self.app.error(str(e))

    # ------------------------------------------------------------------ results
    def _load_study(self) -> None:
        name = self.study_combo.currentData()
        if not name:
            self.k_runs.set("—")
            return
        try:
            _study, res, _mon = self.svc.load(name)
        except Exception as e:  # noqa: BLE001
            self.app.error(str(e))
            return
        self._res = res
        self.k_runs.set(str(len(res)))
        base = res[res["sweep"] == "base"] if "sweep" in res else res
        if len(base) and "harvested_per_year_pct" in base:
            self.k_harv.set(f"{base['harvested_per_year_pct'].median():.2%}")
            self.k_te.set(f"{base['te_realised'].median():.2%}")
            self.k_life.set(f"{base['harvest_life_months'].median():.0f} mo")
        cols = [c for c in res.columns if c not in ("harvested_by_year", "te_by_year", "run_id")]
        self.runs_table.set_frame(res[cols])
        self._show_sweep()
        try:
            self.report.set_text(self.svc.report(name))
        except Exception as e:  # noqa: BLE001
            self.report.set_text(str(e))
        self._show_conc(name)

    def _show_sweep(self) -> None:
        name = self.study_combo.currentData()
        sweep = self.sweep_combo.currentData()
        if not name or not sweep:
            return
        try:
            s = self.svc.summary(name, sweep)
        except Exception:
            s = pd.DataFrame()
        self.sum_table.set_frame(s)
        if s.empty or sweep == "concentrated":
            return
        x = s["level"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=s["harvested_per_year_pct"], name="harvest / yr (% of start)", marker_color=theme.GREEN,
                             error_y=dict(type="data", array=s.get("harvested_per_year_pct_iqr", pd.Series(0, index=s.index)) / 2, visible=True)))
        fig.add_trace(go.Scatter(x=x, y=s["te_realised"], name="realised TE", yaxis="y2", mode="lines+markers", line=dict(color=theme.RED, width=2)))
        if "harvest_life_months" in s:
            fig.add_trace(go.Scatter(x=x, y=s["harvest_life_months"], name="harvest life (months)", yaxis="y3", mode="lines+markers", line=dict(color=theme.ACCENT, dash="dot")))
        fig.update_layout(title=f"Trade-off by {self.sweep_combo.currentText().lower()} (median across windows, bars ± half IQR)",
                          yaxis=dict(title="harvest / yr", tickformat=".1%"), yaxis2=dict(title="TE", overlaying="y", side="right", tickformat=".1%", showgrid=False),
                          yaxis3=dict(title="months", overlaying="y", side="right", position=0.97, showgrid=False, showticklabels=False), height=380)
        self.chart_tradeoff.set_figure(fig)
        try:
            cur = self.svc.curves(name, sweep)
        except Exception:
            cur = pd.DataFrame()
        fig2 = go.Figure()
        for c in cur.columns:
            fig2.add_trace(go.Scatter(x=cur.index, y=cur[c], name=str(c), mode="lines"))
        fig2.update_layout(title="Cumulative harvested losses (% of start value) by month since inception, median across windows",
                           xaxis=dict(title="months"), yaxis=dict(tickformat=".0%"), height=360)
        self.chart_curve.set_figure(fig2)

    def _show_conc(self, name: str) -> None:
        try:
            cg = self.svc.concentrated(name)
            hv = self.svc.concentrated(name, "harvested_per_year_pct")
        except Exception:
            cg = pd.DataFrame()
            hv = pd.DataFrame()
        if cg.empty:
            self.conc_table.set_frame(pd.DataFrame({"note": ["run a study with the concentrated sweep to fill this grid"]}))
            return
        fig = go.Figure(data=go.Heatmap(z=cg.values, x=[f"gain {c:.0%}" for c in cg.columns], y=[f"{i:.0%} position" for i in cg.index],
                                        colorscale="YlOrRd", colorbar=dict(title="months"), hovertemplate="%{y}, %{x}: %{z:.0f} months<extra></extra>"))
        fig.update_layout(title="Months to diversify a concentrated position tax-neutrally (median across windows)", height=420)
        self.chart_conc.set_figure(fig)
        t = cg.copy()
        t.index = [f"{i:.0%} position" for i in t.index]
        t.columns = [f"gain {c:.0%}: months" for c in t.columns]
        if not hv.empty:
            h2 = hv.copy()
            h2.index = t.index
            h2.columns = [f"gain {c:.0%}: harvest/yr" for c in hv.columns]
            t = pd.concat([t, h2], axis=1)
        self.conc_table.set_frame(t.reset_index().rename(columns={"index": "start"}))
