"""Risk model workbench: spec editor (barra_lite or full ERM), fit, versions, exposures vs benchmark, TE decomposition,
factor returns."""
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
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...risk.descriptors import ERM_DEFAULT_STYLES, STYLE_DESCRIPTIONS
from ...risk.factors import STYLE_DEFINITIONS
from ...risk.model import RiskModelSpec
from .. import charts
from ..widgets import FrameTable, KpiCard, button, hbox, header, pct
from ..workers import run_task

VER_COLS = ["id", "name", "created_at", "as_of_date", "universe_name", "lookback_days", "is_active", "snapshot_id", "notes"]


class RiskScreen(QWidget):
    model_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.rs = app.risk_service
        self.model = None
        self.model_id = None
        self._build()
        self._load_spec(self.rs.default_spec())

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
        lay.addWidget(header("Risk model", "Two estimators: barra_lite (fast, six styles, sectors) and the full equity risk model ERM "
                                           "(multi-descriptor styles, industry groups, Newey-West, eigen-adjusted and regime-adjusted covariance, "
                                           "shrunk specific risk). Each fit is a versioned artifact; YANG can propose changes to factors, "
                                           "descriptors and the estimator."))
        g = QGroupBox("Specification")
        f = QFormLayout(g)
        self.kind = QComboBox()
        self.kind.addItems(["barra_lite", "erm"])
        self.kind.currentTextChanged.connect(self._kind_changed)
        self.lookback = QSpinBox()
        self.lookback.setRange(120, 2520)
        self.halflife = QSpinBox()
        self.halflife.setRange(10, 1000)
        self.refresh_days = QSpinBox()
        self.refresh_days.setRange(1, 126)
        self.cov_shrink = QDoubleSpinBox()
        self.cov_shrink.setRange(0, 1)
        self.cov_shrink.setSingleStep(0.05)
        self.spec_shrink = QDoubleSpinBox()
        self.spec_shrink.setRange(0, 1)
        self.spec_shrink.setSingleStep(0.05)
        self.use_sectors = QCheckBox("GICS sector block (barra_lite)")
        self.use_macro = QCheckBox("Macro overlay (rates, slope, credit, USD; barra_lite)")
        f.addRow("Estimator", self.kind)
        f.addRow("Lookback (days)", self.lookback)
        f.addRow("Exposure refresh (days)", self.refresh_days)
        self.row_hl = (QLabel("EWMA half-life"), self.halflife)
        f.addRow(*self.row_hl)
        self.row_cs = (QLabel("Factor cov shrink"), self.cov_shrink)
        f.addRow(*self.row_cs)
        f.addRow("Specific shrink", self.spec_shrink)
        f.addRow(self.use_sectors)
        f.addRow(self.use_macro)
        # barra_lite styles
        self.styles: dict[str, QCheckBox] = {}
        self.style_box = QWidget()
        sb = QVBoxLayout(self.style_box)
        sb.setContentsMargins(0, 0, 0, 0)
        for name, d in STYLE_DEFINITIONS.items():
            cb = QCheckBox(f"{name} — {d.description}")
            self.styles[name] = cb
            sb.addWidget(cb)
        f.addRow("Style factors", self.style_box)
        # ERM options
        self.erm_box = QGroupBox("ERM options")
        ef = QFormLayout(self.erm_box)
        self.industry = QComboBox()
        self.industry.addItems(["gics_sector", "gics_industry_group", "gics_industry"])
        self.robust = QCheckBox("Huber robust regression (slower)")
        self.nw = QSpinBox()
        self.nw.setRange(0, 10)
        self.hl_vol = QSpinBox()
        self.hl_vol.setRange(10, 1000)
        self.hl_corr = QSpinBox()
        self.hl_corr.setRange(20, 2000)
        self.eigen = QCheckBox("Eigenfactor risk adjustment")
        self.vra = QCheckBox("Volatility regime adjustment")
        self.spec_hl = QSpinBox()
        self.spec_hl.setRange(10, 1000)
        self.weight_cap = QDoubleSpinBox()
        self.weight_cap.setRange(50, 100)
        ef.addRow("Industry level", self.industry)
        ef.addRow(self.robust)
        ef.addRow("Newey-West lags", self.nw)
        ef.addRow("Vol half-life", self.hl_vol)
        ef.addRow("Corr half-life", self.hl_corr)
        ef.addRow("Specific half-life", self.spec_hl)
        ef.addRow("Reg. weight cap (pct)", self.weight_cap)
        ef.addRow(self.eigen)
        ef.addRow(self.vra)
        self.erm_styles: dict[str, QCheckBox] = {}
        esb = QVBoxLayout()
        for name in ERM_DEFAULT_STYLES:
            cb = QCheckBox(f"{name} — {STYLE_DESCRIPTIONS.get(name, '')}")
            cb.setChecked(True)
            self.erm_styles[name] = cb
            esb.addWidget(cb)
        esw = QWidget()
        esw.setLayout(esb)
        ef.addRow("ERM styles", esw)
        f.addRow(self.erm_box)
        lay.addWidget(g)
        self.fit_btn = button("Fit model on latest snapshot", self.fit, primary=True)
        lay.addWidget(self.fit_btn)
        lay.addWidget(button("Save spec as default", lambda: (self.rs.save_spec(self.spec()), self.app.status("Spec saved."))))
        g2 = QGroupBox("Model versions")
        gl = QVBoxLayout(g2)
        self.versions = FrameTable(VER_COLS, filter_box=False)
        self.versions.row_selected.connect(self._version_selected)
        gl.addWidget(self.versions)
        gl.addWidget(hbox(button("Activate selected", self._activate), button("Compare with active", self._compare)))
        lay.addWidget(g2, 1)
        scroll.setWidget(w)
        return scroll

    def _right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 0, 0)
        k = QHBoxLayout()
        self.k_id = KpiCard("Active model")
        self.k_r2 = KpiCard("Avg cross-sectional R²")
        self.k_n = KpiCard("Universe")
        self.k_mkt = KpiCard("Market factor vol")
        self.k_te = KpiCard("Portfolio TE")
        for c in (self.k_id, self.k_r2, self.k_n, self.k_mkt, self.k_te):
            k.addWidget(c)
        lay.addLayout(k)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
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
        self.te_chart = charts.PlotlyView()
        self.te_table = FrameTable(["factor", "active_exposure", "variance", "share", "te_contrib"], pct_cols={"share"})
        tsplit = QSplitter(Qt.Horizontal)
        tsplit.addWidget(self.te_chart)
        tsplit.addWidget(self.te_table)
        self.tabs.addTab(tsplit, "TE decomposition")
        self.fr_chart = charts.PlotlyView()
        self.tabs.addTab(self.fr_chart, "Factor returns")
        self.exp_table = FrameTable(None)
        self.tabs.addTab(self.exp_table, "Holdings exposures")
        self.cmp_chart = charts.PlotlyView()
        self.cmp_table = FrameTable(None)
        csplit = QSplitter(Qt.Vertical)
        csplit.addWidget(self.cmp_chart)
        csplit.addWidget(self.cmp_table)
        self.tabs.addTab(csplit, "Version comparison")
        self.diag = QPlainTextEdit()
        self.diag.setReadOnly(True)
        self.tabs.addTab(self.diag, "Diagnostics")
        return w

    # ------------------------------------------------------------------ spec
    def _kind_changed(self, kind: str) -> None:
        erm = kind == "erm"
        self.erm_box.setVisible(erm)
        self.style_box.setVisible(not erm)
        self.use_sectors.setVisible(not erm)
        self.use_macro.setVisible(not erm)
        for wdg in self.row_hl + self.row_cs:
            wdg.setVisible(not erm)

    def _load_spec(self, s: RiskModelSpec) -> None:
        self.kind.setCurrentText(getattr(s, "model_kind", "barra_lite"))
        self.lookback.setValue(s.lookback_days)
        self.halflife.setValue(s.halflife_days)
        self.refresh_days.setValue(s.exposure_refresh_days)
        self.cov_shrink.setValue(s.cov_shrink)
        self.spec_shrink.setValue(s.specific_shrink)
        self.use_sectors.setChecked(s.use_sectors)
        self.use_macro.setChecked(s.use_macro)
        erm = getattr(s, "model_kind", "barra_lite") == "erm"
        for n, cb in self.styles.items():
            cb.setChecked((n in s.styles) if not erm else n in ("value", "momentum", "quality", "size", "lowvol", "growth"))
        for n, cb in self.erm_styles.items():
            cb.setChecked((n in s.styles) if erm else True)
        self.industry.setCurrentText(getattr(s, "industry_level", "gics_industry_group"))
        self.robust.setChecked(getattr(s, "robust", False))
        self.nw.setValue(getattr(s, "nw_lags", 2))
        self.hl_vol.setValue(getattr(s, "hl_vol", 84))
        self.hl_corr.setValue(getattr(s, "hl_corr", 504))
        self.spec_hl.setValue(getattr(s, "specific_hl", 84))
        self.weight_cap.setValue(getattr(s, "weight_cap_pct", 95.0))
        self.eigen.setChecked(getattr(s, "eigen_adjust", True))
        self.vra.setChecked(getattr(s, "vra", True))
        self._kind_changed(self.kind.currentText())

    def spec(self) -> RiskModelSpec:
        erm = self.kind.currentText() == "erm"
        styles = [n for n, cb in (self.erm_styles if erm else self.styles).items() if cb.isChecked()]
        return RiskModelSpec(lookback_days=self.lookback.value(), halflife_days=self.halflife.value(),
                             exposure_refresh_days=self.refresh_days.value(), cov_shrink=self.cov_shrink.value(),
                             specific_shrink=self.spec_shrink.value(), use_sectors=self.use_sectors.isChecked(),
                             use_macro=self.use_macro.isChecked() and not erm, styles=styles, model_kind=self.kind.currentText(),
                             industry_level=self.industry.currentText(), robust=self.robust.isChecked(), nw_lags=self.nw.value(),
                             hl_vol=self.hl_vol.value(), hl_corr=self.hl_corr.value(), specific_hl=self.spec_hl.value(),
                             weight_cap_pct=self.weight_cap.value(), eigen_adjust=self.eigen.isChecked(), vra=self.vra.isChecked())

    # ------------------------------------------------------------------ fit
    def fit(self) -> None:
        snap = self.app.data_service.latest_snapshot()
        if snap is None:
            QMessageBox.information(self, "No data", "Refresh data first (toolbar).")
            return
        self.fit_btn.setEnabled(False)
        self.app.status("Fitting risk model…")
        run_task(self.rs.fit, snap, self.spec(), on_done=self._fit_done, on_error=self._fit_fail, on_progress=self.app.status)

    def _fit_done(self, out) -> None:
        self.fit_btn.setEnabled(True)
        mid, _model = out
        self.app.status(f"Model #{mid} fitted and activated.")
        self.refresh()
        self.model_changed.emit()

    def _fit_fail(self, msg: str) -> None:
        self.fit_btn.setEnabled(True)
        self.app.error(msg)

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        self.versions.set_frame(self.ctx.models.list())
        run_task(self._load, on_done=self._loaded, on_error=self.app.error, wants_progress=False)

    def _load(self) -> dict | None:
        act = self.rs.active()
        if act is None:
            return None
        mid, model = act
        out = {"id": mid, "model": model}
        snap = self.app.data_service.latest_snapshot()
        eid = self.ctx.current_entity_id
        if snap is not None and eid is not None:
            w = self.rs.holdings_weights(eid, snap, model)
            if w is not None:
                bench = self.rs.benchmark_weights(snap, model)
                out["table"] = self.rs.exposure_table(model, w, bench)
                wa = w.reindex(model.symbols).fillna(0.0)
                ba = bench.reindex(model.symbols).fillna(0.0)
                out["dec"] = model.te_decomposition(wa, ba)
                sector_cols = [c for c in model.factors if str(c).startswith(("sec:", "ind:"))]
                X = model.exposures.loc[model.symbols, sector_cols]
                out["sectors"] = pd.DataFrame({"portfolio": X.T @ wa, "benchmark": X.T @ ba})
                out["sectors"].index = [c.split(":", 1)[1] for c in out["sectors"].index]
                ex = model.exposures.loc[list(w.index)].copy()
                ex.insert(0, "weight", w)
                out["holdings"] = ex.reset_index().rename(columns={"index": "symbol"})
        return out

    def _loaded(self, d: dict | None) -> None:
        if d is None:
            self.k_id.set("none", "fit a model")
            return
        self.model_id, self.model = d["id"], d["model"]
        m = self.model
        dg = m.diagnostics
        self.k_id.set(f"#{self.model_id} · {dg.get('model_kind', 'barra_lite')}", f"as of {m.as_of} · {m.spec.lookback_days}d · {len(m.factors)} factors")
        self.k_r2.set(f"{dg.get('avg_r2', 0):.2f}", f"{dg.get('n_dates')} daily regressions")
        self.k_n.set(f"{dg.get('n_symbols')}", f"{dg.get('n_filled_by_regression')} via time-series (ETFs)")
        self.k_mkt.set(pct(m.factor_vols().get("market")), f"median specific vol {pct(dg.get('median_specific_vol'))}")
        skip = {"r2_series", "style_exposure_corr", "t_stats", "descriptor_coverage", "factor_vol_annual", "style_descriptions"}
        self.diag.setPlainText(pd.Series({k: v for k, v in dg.items() if not isinstance(v, dict | list) or k not in skip}).to_string()
                               + "\n\nFactor vols (annualised):\n" + m.factor_vols().round(4).to_string() + f"\n\nSpec: {m.spec}")
        self.fr_chart.set_figure(charts.factor_returns_chart(m.factor_returns))
        if "table" in d:
            self.exp_chart.set_figure(charts.exposure_bars(d["table"]))
            self.radar.set_figure(charts.radar(d["table"]))
            if not d["sectors"].empty:
                self.sec_chart.set_figure(charts.sector_bars(d["sectors"], "Industry weights: portfolio vs benchmark"))
            dec = d["dec"]
            self.k_te.set(pct(dec.attrs["tracking_error"]), "vs " + self.rs.benchmark_name())
            self.te_chart.set_figure(charts.te_decomposition_bars(dec))
            self.te_table.set_frame(dec.reset_index().rename(columns={"index": "factor"}))
            self.exp_table.set_frame(d["holdings"])
        else:
            self.k_te.set("—", "no holdings in model universe")

    # ------------------------------------------------------------------ versions
    def _version_selected(self, row: dict) -> None:
        self._sel_version = int(row["id"])

    def _activate(self) -> None:
        vid = getattr(self, "_sel_version", None)
        if vid is None:
            return
        self.ctx.models.set_active(vid)
        self.refresh()
        self.model_changed.emit()

    def _compare(self) -> None:
        vid = getattr(self, "_sel_version", None)
        if vid is None or self.model_id is None:
            return
        df = self.rs.compare_versions(vid, self.model_id)
        if df.empty:
            return
        self.cmp_table.set_frame(df.reset_index().rename(columns={"index": "factor"}))
        self.cmp_chart.set_figure(charts.factor_vol_compare(df))
        self.tabs.setCurrentIndex(4)
