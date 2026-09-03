"""Risk model workbench: model library (13 presets), spec editor (barra_lite, ERM, hybrid, statistical, PCA, dynamic covariance),
fit, versions, exposures vs benchmark, TE decomposition, factor returns."""
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
from ...risk.model import COV_METHODS, MODEL_KINDS, RISK_MODEL_PRESETS, RiskModelSpec, preset_spec
from .. import charts
from ..widgets import FrameTable, KpiCard, button, hbox, header, pct
from ..workers import run_task

VER_COLS = ["id", "name", "created_at", "as_of_date", "universe_name", "lookback_days", "is_active", "snapshot_id", "notes"]
KIND_BLURB = {
    "barra_lite": "Fast Barra-style model: market, six styles, GICS sectors, EWMA covariance.",
    "erm": "Full equity risk model: 10 multi-descriptor styles, industry groups, Newey-West, eigen and regime adjustments.",
    "hybrid": "ERM plus principal components of its residuals (captures themes the descriptors miss).",
    "statistical": "Potomac calibrated covariance: fixed window, equal/exponential weights, Ledoit-Wolf or sample; eigen-factor form.",
    "pca": "Asymptotic principal components with automatic factor count; needs prices only.",
}


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
        split.setSizes([380, 1100])

    def _left(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(header("Risk model", "Pick a model from the library or build your own: fundamental (barra_lite, ERM, hybrid), statistical "
                                           "(Potomac calibrated covariance, PCA) and dynamic covariances (GARCH, regime). Each fit is a versioned artifact."))
        # ---- library
        gl = QGroupBox("Model library")
        ll = QVBoxLayout(gl)
        self.preset = QComboBox()
        self.preset.addItem("— custom specification —", "")
        for name in RISK_MODEL_PRESETS:
            self.preset.addItem(name, name)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        ll.addWidget(self.preset)
        self.preset_desc = QLabel("")
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setProperty("muted", True)
        ll.addWidget(self.preset_desc)
        ll.addWidget(hbox(button("Fit this preset", self.fit, primary=True), button("Fit whole library (compare)", self.fit_library,
                                                                                      tooltip="Fits every preset without changing the active model and tabulates the results")))
        lay.addWidget(gl)

        g = QGroupBox("Specification")
        f = QFormLayout(g)
        self.kind = QComboBox()
        self.kind.addItems(list(MODEL_KINDS))
        self.kind.currentTextChanged.connect(self._kind_changed)
        self.kind_blurb = QLabel("")
        self.kind_blurb.setWordWrap(True)
        self.kind_blurb.setProperty("muted", True)
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
        f.addRow(self.kind_blurb)
        self.row_lb = (QLabel("Lookback (days)"), self.lookback)
        f.addRow(*self.row_lb)
        self.row_rf = (QLabel("Exposure refresh (days)"), self.refresh_days)
        f.addRow(*self.row_rf)
        self.row_hl = (QLabel("EWMA half-life"), self.halflife)
        f.addRow(*self.row_hl)
        self.row_cs = (QLabel("Factor cov shrink"), self.cov_shrink)
        f.addRow(*self.row_cs)
        self.row_ss = (QLabel("Specific shrink"), self.spec_shrink)
        f.addRow(*self.row_ss)
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
        self.row_styles = (QLabel("Style factors"), self.style_box)
        f.addRow(*self.row_styles)
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
        self.hybrid_k = QSpinBox()
        self.hybrid_k.setRange(1, 15)
        ef.addRow("Industry level", self.industry)
        ef.addRow(self.robust)
        ef.addRow("Newey-West lags", self.nw)
        ef.addRow("Vol half-life", self.hl_vol)
        ef.addRow("Corr half-life", self.hl_corr)
        ef.addRow("Specific half-life", self.spec_hl)
        ef.addRow("Reg. weight cap (pct)", self.weight_cap)
        ef.addRow(self.eigen)
        ef.addRow(self.vra)
        self.row_hyb = (QLabel("Statistical factors (hybrid)"), self.hybrid_k)
        ef.addRow(*self.row_hyb)
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
        # statistical options
        self.stat_box = QGroupBox("Statistical options")
        sf = QFormLayout(self.stat_box)
        self.stat_lookback = QSpinBox()
        self.stat_lookback.setRange(21, 1512)
        self.stat_weighting = QComboBox()
        self.stat_weighting.addItems(["equal", "exponential"])
        self.stat_estimator = QComboBox()
        self.stat_estimator.addItems(["ledoit_wolf", "sample"])
        self.stat_factors = QSpinBox()
        self.stat_factors.setRange(0, 60)
        self.stat_factors.setSpecialValueText("auto")
        sf.addRow("Window (days)", self.stat_lookback)
        sf.addRow("Weighting", self.stat_weighting)
        self.row_est = (QLabel("Estimator"), self.stat_estimator)
        sf.addRow(*self.row_est)
        sf.addRow("Factors (0 = auto)", self.stat_factors)
        note = QLabel("Calibration study (Risk lab › Calibration) recommends 126d equal Ledoit-Wolf for 3–6 month horizons and the sample "
                      "matrix for tight substitute pairs.")
        note.setWordWrap(True)
        note.setProperty("muted", True)
        sf.addRow(note)
        f.addRow(self.stat_box)
        # dynamic covariance
        self.dyn_box = QGroupBox("Factor covariance dynamics")
        df_ = QFormLayout(self.dyn_box)
        self.cov_method = QComboBox()
        self.cov_method.addItems(list(COV_METHODS))
        self.horizon = QSpinBox()
        self.horizon.setRange(1, 252)
        df_.addRow("Method", self.cov_method)
        df_.addRow("Decision horizon (days)", self.horizon)
        dn = QLabel("ewma: half-life covariance (default). garch: GARCH(1,1) variance forecasts per factor over the horizon with EWMA correlations. "
                    "regime: calm/stress covariances blended by today's stress probability.")
        dn.setWordWrap(True)
        dn.setProperty("muted", True)
        df_.addRow(dn)
        f.addRow(self.dyn_box)
        lay.addWidget(g)
        self.fit_btn = button("Fit model on latest snapshot", self.fit, primary=True)
        lay.addWidget(self.fit_btn)
        lay.addWidget(button("Save spec as default", lambda: (self.rs.save_spec(self.spec()), self.app.status("Spec saved."))))
        g2 = QGroupBox("Model versions")
        gl2 = QVBoxLayout(g2)
        self.versions = FrameTable(VER_COLS, filter_box=False)
        self.versions.row_selected.connect(self._version_selected)
        gl2.addWidget(self.versions)
        gl2.addWidget(hbox(button("Activate selected", self._activate), button("Compare with active", self._compare)))
        lay.addWidget(g2, 1)
        scroll.setWidget(w)
        return scroll

    def _right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 0, 0)
        k = QHBoxLayout()
        self.k_id = KpiCard("Active model")
        self.k_r2 = KpiCard("Explained / R²")
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
        self.lib_table = FrameTable(["preset", "model_id", "kind", "factors", "symbols", "avg_r2", "median_specific_vol", "market_vol", "status"],
                                    pct_cols={"median_specific_vol", "market_vol"})
        self.tabs.addTab(self.lib_table, "Library comparison")
        self.diag = QPlainTextEdit()
        self.diag.setReadOnly(True)
        self.tabs.addTab(self.diag, "Diagnostics")
        return w

    # ------------------------------------------------------------------ spec
    def _preset_changed(self, _i: int) -> None:
        name = self.preset.currentData()
        if not name:
            self.preset_desc.setText("")
            return
        self.preset_desc.setText(RISK_MODEL_PRESETS[name]["description"])
        self._load_spec(preset_spec(name), keep_preset=True)

    def _kind_changed(self, kind: str) -> None:
        self.kind_blurb.setText(KIND_BLURB.get(kind, ""))
        fundamental = kind in ("barra_lite", "erm", "hybrid")
        erm = kind in ("erm", "hybrid")
        stat = kind in ("statistical", "pca")
        self.erm_box.setVisible(erm)
        self.stat_box.setVisible(stat)
        self.dyn_box.setVisible(fundamental)
        self.use_sectors.setVisible(kind == "barra_lite")
        self.use_macro.setVisible(kind == "barra_lite")
        for wdg in self.row_hl + self.row_cs + self.row_styles:
            wdg.setVisible(kind == "barra_lite")
        for wdg in self.row_lb + self.row_rf + self.row_ss:
            wdg.setVisible(fundamental)
        for wdg in self.row_hyb:
            wdg.setVisible(kind == "hybrid")
        for wdg in self.row_est:
            wdg.setVisible(kind == "statistical")

    def _load_spec(self, s: RiskModelSpec, keep_preset: bool = False) -> None:
        if not keep_preset:
            self.preset.blockSignals(True)
            i = self.preset.findData(getattr(s, "preset", "") or "")
            self.preset.setCurrentIndex(max(i, 0))
            self.preset_desc.setText(RISK_MODEL_PRESETS[s.preset]["description"] if getattr(s, "preset", "") in RISK_MODEL_PRESETS else "")
            self.preset.blockSignals(False)
        self.kind.setCurrentText(getattr(s, "model_kind", "barra_lite"))
        self.lookback.setValue(s.lookback_days)
        self.halflife.setValue(s.halflife_days)
        self.refresh_days.setValue(s.exposure_refresh_days)
        self.cov_shrink.setValue(s.cov_shrink)
        self.spec_shrink.setValue(s.specific_shrink)
        self.use_sectors.setChecked(s.use_sectors)
        self.use_macro.setChecked(s.use_macro)
        erm = getattr(s, "model_kind", "barra_lite") in ("erm", "hybrid")
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
        self.hybrid_k.setValue(getattr(s, "hybrid_stat_factors", 5))
        self.stat_lookback.setValue(getattr(s, "stat_lookback", 126))
        self.stat_weighting.setCurrentText(getattr(s, "stat_weighting", "equal"))
        self.stat_estimator.setCurrentText(getattr(s, "stat_estimator", "ledoit_wolf"))
        self.stat_factors.setValue(int(getattr(s, "stat_factors", None) or 0))
        self.cov_method.setCurrentText(getattr(s, "cov_method", "ewma"))
        self.horizon.setValue(getattr(s, "horizon_days", 21))
        self._kind_changed(self.kind.currentText())

    def spec(self) -> RiskModelSpec:
        kind = self.kind.currentText()
        erm = kind in ("erm", "hybrid")
        styles = [n for n, cb in (self.erm_styles if erm else self.styles).items() if cb.isChecked()]
        return RiskModelSpec(name=kind, lookback_days=self.lookback.value(), halflife_days=self.halflife.value(),
                             exposure_refresh_days=self.refresh_days.value(), cov_shrink=self.cov_shrink.value(),
                             specific_shrink=self.spec_shrink.value(), use_sectors=self.use_sectors.isChecked(),
                             use_macro=self.use_macro.isChecked() and kind == "barra_lite", styles=styles, model_kind=kind,
                             industry_level=self.industry.currentText(), robust=self.robust.isChecked(), nw_lags=self.nw.value(),
                             hl_vol=self.hl_vol.value(), hl_corr=self.hl_corr.value(), specific_hl=self.spec_hl.value(),
                             weight_cap_pct=self.weight_cap.value(), eigen_adjust=self.eigen.isChecked(), vra=self.vra.isChecked(),
                             hybrid_stat_factors=self.hybrid_k.value(), stat_lookback=self.stat_lookback.value(),
                             stat_weighting=self.stat_weighting.currentText(), stat_estimator=self.stat_estimator.currentText(),
                             stat_factors=self.stat_factors.value() or None, cov_method=self.cov_method.currentText(),
                             horizon_days=self.horizon.value(), preset=self.preset.currentData() or "")

    # ------------------------------------------------------------------ fit
    def fit(self) -> None:
        snap = self.app.data_service.latest_snapshot()
        if snap is None:
            QMessageBox.information(self, "No data", "Refresh data first (toolbar).")
            return
        self.fit_btn.setEnabled(False)
        self.app.status("Fitting risk model…")
        run_task(self.rs.fit, snap, self.spec(), on_done=self._fit_done, on_error=self._fit_fail, on_progress=self.app.status)

    def fit_library(self) -> None:
        snap = self.app.data_service.latest_snapshot()
        if snap is None:
            QMessageBox.information(self, "No data", "Refresh data first (toolbar).")
            return
        if QMessageBox.question(self, "Fit library", f"Fit all {len(RISK_MODEL_PRESETS)} presets? The ERM variants take ~20–60 s each. "
                                                     "The active model is not changed.") != QMessageBox.Yes:
            return
        self.fit_btn.setEnabled(False)
        run_task(self.rs.fit_library, snap, None, on_done=self._lib_done, on_error=self._fit_fail, on_progress=self.app.status)

    def _lib_done(self, df: pd.DataFrame) -> None:
        self.fit_btn.setEnabled(True)
        self.lib_table.set_frame(df)
        self.tabs.setCurrentWidget(self.lib_table)
        self.app.status(f"Library fitted: {int((df['status'] == 'ok').sum())} of {len(df)} presets succeeded.")
        self.refresh()

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
                if sector_cols:
                    X = model.exposures.loc[model.symbols, sector_cols]
                    out["sectors"] = pd.DataFrame({"portfolio": X.T @ wa, "benchmark": X.T @ ba})
                    out["sectors"].index = [c.split(":", 1)[1] for c in out["sectors"].index]
                else:
                    out["sectors"] = pd.DataFrame()
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
        kind = dg.get("model_kind", "barra_lite")
        label = dg.get("preset") or kind
        self.k_id.set(f"#{self.model_id} · {kind}", f"{label} · as of {m.as_of} · {len(m.factors)} factors" + (f" · {dg['cov_method']}" if dg.get("cov_method") else ""))
        r2 = dg.get("avg_r2")
        self.k_r2.set(f"{r2:.2f}" if r2 is not None else "—", f"{dg.get('n_dates')} days" + (" · explained variance" if kind in ("statistical", "pca") else " · cross-sectional R²"))
        self.k_n.set(f"{dg.get('n_symbols')}", f"{dg.get('n_filled_by_regression', 0)} via time-series (ETFs)" if kind not in ("statistical", "pca") else "prices only")
        mv = m.factor_vols().get("market") if "market" in m.factors else None
        self.k_mkt.set(pct(mv) if mv is not None else "—", f"median specific vol {pct(dg.get('median_specific_vol'))}")
        skip = {"r2_series", "style_exposure_corr", "t_stats", "descriptor_coverage", "factor_vol_annual", "style_descriptions", "garch_params", "eigenvalues_top"}
        self.diag.setPlainText(pd.Series({k: v for k, v in dg.items() if not isinstance(v, dict | list) or k not in skip}).to_string()
                               + "\n\nFactor vols (annualised):\n" + m.factor_vols().round(4).to_string() + f"\n\nSpec: {m.spec}")
        self.fr_chart.set_figure(charts.factor_returns_chart(m.factor_returns))
        if "table" in d:
            tab = d["table"]
            if (tab["kind"] == "style").any() or (tab["kind"] == "market").any():
                self.exp_chart.set_figure(charts.exposure_bars(tab))
                self.radar.set_figure(charts.radar(tab))
            else:
                self.exp_chart.set_message("Statistical model: no named style exposures. See TE decomposition and holdings exposures.")
                self.radar.set_message("")
            if not d["sectors"].empty:
                self.sec_chart.set_figure(charts.sector_bars(d["sectors"], "Industry weights: portfolio vs benchmark"))
            else:
                self.sec_chart.set_message("No industry factors in this model.")
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
