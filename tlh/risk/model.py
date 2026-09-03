"""Barra-style multi-factor equity risk model and the model library.

Fit pipeline (barra_lite)
-------------------------
1. Daily total returns for the fit universe over `lookback_days`.
2. Exposures rebuilt every `exposure_refresh_days` trading days (held constant in between).
3. Per day: weighted cross-sectional regression r_t = X_t f_t + u_t with sqrt(cap) weights and the Barra
   constraint that cap-weighted sector factor returns sum to zero (so `market` is the cap-weighted market).
4. Factor covariance: EWMA (half-life `halflife_days`) with shrinkage toward its diagonal; annualised.
5. Specific variance: EWMA of squared residuals, shrunk toward the cross-sectional median; annualised.
6. Names without fundamental/sector exposures (ETFs, funds) get exposures by ridge regression of their returns on
   the factor returns; their residual variance becomes specific variance.
7. Optional macro block: time-series betas of every asset to macro shocks (rates, slope, credit, USD).

Other estimators share the same `FittedRiskModel` output (see `RiskModelSpec.model_kind`):
    erm          full equity risk model (risk/erm.py)
    hybrid       ERM + statistical factors on the residuals (risk/statistical.py)
    statistical  Potomac calibrated covariance (lookback / weighting / Ledoit-Wolf) in eigen-factor form
    pca          asymptotic principal components
and any fundamental model can carry a dynamic factor covariance (`cov_method`: ewma | garch | regime).
`RISK_MODEL_PRESETS` is the model library shown in the GUI and to YANG.

This module is AI-editable (ai/registry.py). Keep `FittedRiskModel`'s public surface stable: the optimizer and
GUI depend on `exposures`, `factor_cov`, `specific_var`, `covariance()`, `tracking_error()`, `te_decomposition()`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import STYLE_DEFINITIONS, ExposureBuild, FactorInputs, build_exposures

TRADING_DAYS = 252

MODEL_KINDS = ("barra_lite", "erm", "hybrid", "statistical", "pca")
COV_METHODS = ("ewma", "garch", "regime")


@dataclass
class RiskModelSpec:
    name: str = "barra_lite"
    lookback_days: int = 504
    halflife_days: int = 126
    exposure_refresh_days: int = 21
    styles: list[str] = field(default_factory=lambda: ["value", "momentum", "quality", "size", "lowvol", "growth"])
    use_sectors: bool = True
    use_macro: bool = False
    macro_columns: list[str] = field(default_factory=lambda: ["rate_10y", "slope_10y2y", "credit_spread", "usd"])
    cov_shrink: float = 0.10
    specific_shrink: float = 0.25
    min_obs_per_symbol: int = 120
    ridge_lambda: float = 0.05
    winsor_returns: float = 0.25      # clip daily returns at +/-25% in the regression
    # ---- equity risk model (ERM) options; used when model_kind in (erm, hybrid)
    model_kind: str = "barra_lite"    # barra_lite | erm | hybrid | statistical | pca
    industry_level: str = "gics_industry_group"
    robust: bool = False
    weight_cap_pct: float = 95.0
    nw_lags: int = 2
    hl_vol: int = 84
    hl_corr: int = 504
    eigen_adjust: bool = True
    vra: bool = True
    specific_hl: int = 84
    # ---- statistical / PCA options (model_kind statistical | pca) and hybrid extension
    stat_lookback: int = 126
    stat_weighting: str = "equal"     # equal | exponential (half-life = 0.35 x lookback)
    stat_estimator: str = "ledoit_wolf"   # sample | ledoit_wolf
    stat_factors: int | None = None   # PCA components / eigen factors (None = automatic)
    hybrid_stat_factors: int = 5
    # ---- dynamic factor covariance for fundamental models
    cov_method: str = "ewma"          # ewma | garch | regime
    horizon_days: int = 21
    preset: str = ""

    def factor_names(self, sector_cols: list[str]) -> list[str]:
        cols = ["market"] + list(self.styles) + list(sector_cols)
        if self.use_macro:
            cols += [f"macro:{c}" for c in self.macro_columns]
        return cols

    @property
    def is_fundamental(self) -> bool:
        return self.model_kind in ("barra_lite", "erm", "hybrid")


# ====================================================================================== model library
RISK_MODEL_PRESETS: dict[str, dict] = {
    "ERM · Standard": {
        "description": "Full equity risk model in the Barra USE4 tradition: 10 multi-descriptor styles, GICS industry groups, Newey-West, "
                       "eigenfactor and volatility-regime adjustments, Bayesian-shrunk specific risk. The default for harvesting and baskets.",
        "spec": dict(name="erm", model_kind="erm")},
    "ERM · Short horizon (1 month)": {
        "description": "Same factor structure with fast half-lives (vol 21d, corr 126d, specific 42d) for a one-month decision horizon: "
                       "reacts to regime changes quickly at the cost of noisier levels.",
        "spec": dict(name="erm_short", model_kind="erm", hl_vol=21, hl_corr=126, specific_hl=42, horizon_days=21)},
    "ERM · Long horizon (6 months)": {
        "description": "Slow half-lives (vol 252d, corr 756d, specific 252d) over a four-year window for strategic tracking-error budgets "
                       "and multi-year glide paths.",
        "spec": dict(name="erm_long", model_kind="erm", lookback_days=1008, hl_vol=252, hl_corr=756, specific_hl=252, horizon_days=126)},
    "ERM · Robust (Huber)": {
        "description": "ERM with Huber M-estimation of the daily cross-sections so single-name blow-ups and data errors do not drive factor returns.",
        "spec": dict(name="erm_robust", model_kind="erm", robust=True)},
    "ERM · GARCH-dynamic covariance": {
        "description": "ERM exposures and factor returns; factor variances forecast by GARCH(1,1) over the decision horizon and combined with "
                       "EWMA correlations. Sharper after volatility shocks than a fixed half-life.",
        "spec": dict(name="erm_garch", model_kind="erm", cov_method="garch", horizon_days=21)},
    "ERM · Regime-conditional covariance": {
        "description": "Two-state (calm/stress) factor covariance blended by the current probability of stress from market volatility. "
                       "Conservative when the tape is fragile.",
        "spec": dict(name="erm_regime", model_kind="erm", cov_method="regime")},
    "Hybrid · ERM + statistical factors": {
        "description": "ERM plus five principal components extracted from its residual returns, capturing co-movement the descriptors miss "
                       "(themes, crowded trades). Lower specific risk, more explained variance.",
        "spec": dict(name="hybrid", model_kind="hybrid", hybrid_stat_factors=5)},
    "Potomac Calibrated · 126d equal Ledoit-Wolf": {
        "description": "The 2026 calibration study's winner for 3- and 6-month horizons: 126-day window, equal weights, Ledoit-Wolf "
                       "constant-correlation shrinkage. Best composite forecast of basket tracking error out of sample.",
        "spec": dict(name="calibrated_126_lw", model_kind="statistical", stat_lookback=126, stat_weighting="equal", stat_estimator="ledoit_wolf", horizon_days=126)},
    "Potomac Calibrated · 189d exponential Ledoit-Wolf (1 month)": {
        "description": "Calibration winner at the one-month horizon: 189-day window, exponential weights (half-life 66d), Ledoit-Wolf. Ranks "
                       "risk fastest after a regime change.",
        "spec": dict(name="calibrated_189_exp_lw", model_kind="statistical", stat_lookback=189, stat_weighting="exponential", stat_estimator="ledoit_wolf", horizon_days=21)},
    "Tight-pair · 126d sample covariance": {
        "description": "Unshrunk sample covariance for substitute-pair tracking error (IVV vs SPY): shrinkage drags a 0.997 correlation toward "
                       "the universe average and triples the forecast TE of near-identical pairs.",
        "spec": dict(name="pair_sample_126", model_kind="statistical", stat_lookback=126, stat_weighting="equal", stat_estimator="sample")},
    "Statistical · PCA (auto factors)": {
        "description": "Asymptotic principal components on one year of exponentially weighted returns; the number of factors is chosen by the "
                       "eigenvalue-ratio test. No fundamentals needed, so it covers every asset with a price history.",
        "spec": dict(name="pca", model_kind="pca", stat_lookback=252, stat_weighting="exponential", stat_factors=None)},
    "barra_lite · Fast six-style": {
        "description": "Market, six styles and GICS sectors with EWMA covariance; fits ~640 names in about three seconds. Good for quick "
                       "what-ifs and the walk-forward backtester.",
        "spec": dict(name="barra_lite", model_kind="barra_lite")},
    "barra_lite · with macro block": {
        "description": "barra_lite plus time-series betas to rates, curve slope, credit spreads and the dollar, for macro stress tests.",
        "spec": dict(name="barra_lite_macro", model_kind="barra_lite", use_macro=True)},
}


def preset_spec(name: str, **overrides) -> RiskModelSpec:
    if name not in RISK_MODEL_PRESETS:
        raise KeyError(f"unknown risk-model preset '{name}'; choose from {list(RISK_MODEL_PRESETS)}")
    kw = dict(RISK_MODEL_PRESETS[name]["spec"])
    kw.update(overrides)
    kw.setdefault("preset", name)
    return RiskModelSpec(**kw)


def preset_table() -> pd.DataFrame:
    rows = [{"preset": k, "model_kind": v["spec"].get("model_kind", "barra_lite"), "cov_method": v["spec"].get("cov_method", "ewma"),
             "description": v["description"]} for k, v in RISK_MODEL_PRESETS.items()]
    return pd.DataFrame(rows)


@dataclass
class FittedRiskModel:
    spec: RiskModelSpec
    as_of: date
    exposures: pd.DataFrame          # symbol x factor
    factor_cov: pd.DataFrame         # factor x factor, annualised
    specific_var: pd.Series          # symbol, annualised variance
    factor_returns: pd.DataFrame     # date x factor (daily)
    diagnostics: dict = field(default_factory=dict)
    universe_name: str = ""
    snapshot_id: str | None = None

    # ------------------------------------------------------------------ core algebra
    @property
    def factors(self) -> list[str]:
        return list(self.factor_cov.columns)

    @property
    def symbols(self) -> list[str]:
        return list(self.exposures.index)

    def align(self, weights: pd.Series) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        """Restrict to symbols the model knows; unknown symbols raise."""
        unknown = [s for s in weights.index if s not in self.exposures.index]
        if unknown:
            raise KeyError(f"symbols not in risk model: {unknown[:10]}")
        w = weights.astype(float)
        X = self.exposures.loc[w.index]
        D = self.specific_var.loc[w.index]
        return w, X, D

    def covariance(self, symbols: list[str]) -> pd.DataFrame:
        X = self.exposures.loc[symbols].values
        F = self.factor_cov.values
        D = np.diag(self.specific_var.loc[symbols].values)
        return pd.DataFrame(X @ F @ X.T + D, index=symbols, columns=symbols)

    def portfolio_exposures(self, weights: pd.Series) -> pd.Series:
        w, X, _ = self.align(weights)
        return X.T @ w

    def portfolio_variance(self, weights: pd.Series) -> float:
        w, X, D = self.align(weights)
        b = X.T @ w
        return float(b.values @ self.factor_cov.values @ b.values + (w.values ** 2 * D.values).sum())

    def portfolio_risk(self, weights: pd.Series) -> float:
        return float(np.sqrt(max(self.portfolio_variance(weights), 0.0)))

    def tracking_error(self, weights: pd.Series, benchmark: pd.Series) -> float:
        active = _active(weights, benchmark)
        return self.portfolio_risk(active)

    def te_decomposition(self, weights: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
        """Contribution of each factor and of specific risk to tracking variance (sums to TE^2)."""
        active = _active(weights, benchmark)
        a, X, D = self.align(active)
        b = X.T @ a
        Fb = self.factor_cov.values @ b.values
        contrib = pd.Series(b.values * Fb, index=self.factors, name="variance")
        spec = float((a.values ** 2 * D.values).sum())
        rows = contrib.to_frame()
        rows.loc["specific", "variance"] = spec
        total = rows["variance"].sum()
        rows["share"] = rows["variance"] / total if total > 0 else 0.0
        rows["active_exposure"] = b.reindex(rows.index)
        rows["te_contrib"] = rows["variance"] / np.sqrt(total) if total > 0 else 0.0
        rows.attrs["tracking_error"] = float(np.sqrt(max(total, 0.0)))
        return rows

    def factor_vols(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.factor_cov.values)), index=self.factors)

    @property
    def kind(self) -> str:
        return self.diagnostics.get("model_kind", self.spec.model_kind)

    # ------------------------------------------------------------------ persistence
    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.exposures.to_parquet(folder / "exposures.parquet")
        self.factor_cov.to_parquet(folder / "factor_cov.parquet")
        self.specific_var.rename("specific_var").to_frame().to_parquet(folder / "specific_var.parquet")
        self.factor_returns.to_parquet(folder / "factor_returns.parquet")
        meta = {"spec": asdict(self.spec), "as_of": self.as_of.isoformat(), "diagnostics": self.diagnostics,
                "universe_name": self.universe_name, "snapshot_id": self.snapshot_id}
        (folder / "model.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: Path) -> FittedRiskModel:
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        return cls(
            spec=RiskModelSpec(**{k: v for k, v in meta["spec"].items() if k in RiskModelSpec.__dataclass_fields__}), as_of=date.fromisoformat(meta["as_of"]),
            exposures=pd.read_parquet(folder / "exposures.parquet"),
            factor_cov=pd.read_parquet(folder / "factor_cov.parquet"),
            specific_var=pd.read_parquet(folder / "specific_var.parquet")["specific_var"],
            factor_returns=pd.read_parquet(folder / "factor_returns.parquet"),
            diagnostics=meta.get("diagnostics", {}), universe_name=meta.get("universe_name", ""),
            snapshot_id=meta.get("snapshot_id"),
        )


def _active(weights: pd.Series, benchmark: pd.Series) -> pd.Series:
    idx = weights.index.union(benchmark.index)
    return weights.reindex(idx).fillna(0.0) - benchmark.reindex(idx).fillna(0.0)


# ====================================================================================== fitting
class FactorRiskModel:
    def __init__(self, spec: RiskModelSpec | None = None):
        self.spec = spec or RiskModelSpec()

    def fit(self, prices: pd.DataFrame, securities: pd.DataFrame, fundamentals: pd.DataFrame,
            macro_levels: pd.DataFrame | None = None, as_of: date | None = None,
            universe_name: str = "", snapshot_id: str | None = None, progress=None,
            volume: pd.DataFrame | None = None) -> FittedRiskModel:
        """`prices`: date x symbol total-return close. `securities`/`fundamentals`: indexed by symbol."""
        spec = self.spec
        say = progress or (lambda m: None)
        prices = prices.sort_index()
        if as_of is not None:
            prices = prices.loc[: pd.Timestamp(as_of)]
        if spec.model_kind not in MODEL_KINDS:
            raise ValueError(f"unknown model_kind {spec.model_kind}; choose from {MODEL_KINDS}")
        if spec.model_kind in ("statistical", "pca"):
            out = self._fit_statistical(prices, say)
            return self._finalise(out, spec, universe_name, snapshot_id, say)
        if spec.model_kind in ("erm", "hybrid"):
            out = self._fit_erm(prices, securities, fundamentals, volume, say)
            return self._finalise(out, spec, universe_name, snapshot_id, say)
        out = self._fit_barra_lite(prices, securities, fundamentals, macro_levels, say)
        return self._finalise(out, spec, universe_name, snapshot_id, say)

    # ------------------------------------------------------------------ shared post-processing
    def _finalise(self, out: dict, spec: RiskModelSpec, universe_name: str, snapshot_id: str | None, say) -> FittedRiskModel:
        X, F, D, FR, diag = out["exposures"], out["factor_cov"], out["specific_var"], out["factor_returns"], dict(out["diagnostics"])
        style_cols = list(out.get("style_cols", spec.styles))
        if spec.model_kind == "hybrid":
            from .statistical import add_statistical_factors
            say("Extracting statistical factors from residuals…")
            U = out.get("residuals")
            if U is not None:
                X, F, D, FR, info = add_statistical_factors(X, F, D, FR, U, spec.hybrid_stat_factors or None, halflife=spec.hl_corr // 4 or 126)
                diag.update(info)
                diag["model_kind"] = "hybrid"
        if spec.is_fundamental and spec.cov_method in ("garch", "regime"):
            from .statistical import garch_factor_cov, regime_factor_cov
            say(f"Dynamic factor covariance ({spec.cov_method})…")
            fr = FR.reindex(columns=F.columns).dropna(how="all")
            F_dyn, info = garch_factor_cov(fr, spec.horizon_days, spec.hl_corr) if spec.cov_method == "garch" else regime_factor_cov(fr)
            # keep the eigen/VRA-adjusted level from the base fit: rescale the dynamic matrix to the base total factor variance ratio
            diag["cov_method"] = spec.cov_method
            diag.update(info)
            diag["dynamic_vs_base_vol_ratio"] = float(np.sqrt(np.trace(F_dyn.values) / max(np.trace(F.values), 1e-18)))
            F = F_dyn
        diag.setdefault("model_kind", spec.model_kind)
        if spec.preset:
            diag["preset"] = spec.preset
        diag["factor_vol_annual"] = {str(k): float(v) for k, v in zip(F.columns, np.sqrt(np.maximum(np.diag(F.values), 0.0)), strict=True)}
        fitted_spec = replace(spec, styles=style_cols) if spec.model_kind in ("erm", "hybrid") else spec
        return FittedRiskModel(spec=fitted_spec, as_of=out["as_of"], exposures=X, factor_cov=F, specific_var=D, factor_returns=FR,
                               diagnostics=diag, universe_name=universe_name, snapshot_id=snapshot_id)

    # ------------------------------------------------------------------ estimators
    def _fit_statistical(self, prices: pd.DataFrame, say) -> dict:
        from .statistical import StatOptions, fit_calibrated, fit_pca
        spec = self.spec
        opts = StatOptions(lookback=spec.stat_lookback, weighting=spec.stat_weighting, estimator=spec.stat_estimator,
                           n_factors=spec.stat_factors, min_obs=min(spec.min_obs_per_symbol, max(spec.stat_lookback - 5, 20)))
        return fit_pca(prices, opts, say) if spec.model_kind == "pca" else fit_calibrated(prices, opts, say)

    def _fit_erm(self, prices, securities, fundamentals, volume, say) -> dict:
        from .erm import ERMOptions, fit_erm
        spec = self.spec
        opts = ERMOptions(industry_level=spec.industry_level, styles=list(spec.styles), robust=spec.robust,
                          weight_cap_pct=spec.weight_cap_pct, nw_lags=spec.nw_lags, hl_vol=spec.hl_vol, hl_corr=spec.hl_corr,
                          eigen_adjust=spec.eigen_adjust, vra=spec.vra, specific_hl=spec.specific_hl, min_obs=spec.min_obs_per_symbol)
        out = fit_erm(prices, securities.reindex(prices.columns), fundamentals.reindex(prices.columns), volume,
                      spec.lookback_days, spec.exposure_refresh_days, opts, progress=say)
        out["diagnostics"] = dict(out["diagnostics"])
        out["diagnostics"].setdefault("model_kind", "erm")
        return out

    def _fit_barra_lite(self, prices, securities, fundamentals, macro_levels, say) -> dict:
        spec = self.spec
        prices = prices.dropna(axis=1, thresh=spec.min_obs_per_symbol)
        securities = securities.reindex(prices.columns)
        fundamentals = fundamentals.reindex(prices.columns)
        rets = prices.pct_change().iloc[1:]
        rets = rets.clip(-spec.winsor_returns, spec.winsor_returns)
        fit_dates = rets.index[-spec.lookback_days:]
        if len(fit_dates) < spec.min_obs_per_symbol:
            raise ValueError(f"only {len(fit_dates)} return observations; need >= {spec.min_obs_per_symbol}")
        t_end = fit_dates[-1]

        # ---- exposures on a monthly grid -----------------------------------------------------
        say("Building factor exposures...")
        grid = list(fit_dates[:: spec.exposure_refresh_days])
        if grid[-1] != t_end:
            grid.append(t_end)
        builds: dict[pd.Timestamp, ExposureBuild] = {}
        for t in grid:
            fi = FactorInputs(prices=prices, t=t, fundamentals=fundamentals, securities=securities)
            builds[t] = build_exposures(fi, spec.styles, spec.use_sectors)
        sector_cols = builds[t_end].sector_cols
        core_factors = ["market"] + list(spec.styles) + sector_cols

        # ---- daily cross-sectional regressions --------------------------------------------------
        say("Estimating factor returns (cross-sectional WLS)...")
        f_rows, resid_rows, r2s = [], [], []
        grid_arr = np.array(grid, dtype="datetime64[ns]")
        for t in fit_dates:
            k = int(np.searchsorted(grid_arr, np.datetime64(t), side="right")) - 1
            b = builds[grid[max(k, 0)]]
            X = b.exposures[core_factors]
            r = rets.loc[t]
            ok = X.notna().all(axis=1) & r.notna() & (b.cap_weights > 0)
            if ok.sum() < len(core_factors) + 5:
                continue
            Xo, ro, wo = X[ok].values, r[ok].values, np.sqrt(b.cap_weights[ok].values)
            f, u, r2 = _constrained_wls(Xo, ro, wo, X.columns.tolist(), sector_cols, b.cap_weights[ok].values)
            f_rows.append(pd.Series(f, index=core_factors, name=t))
            resid_rows.append(pd.Series(u, index=X.index[ok], name=t))
            r2s.append(r2)
        F_ret = pd.DataFrame(f_rows)
        U = pd.DataFrame(resid_rows).reindex(columns=prices.columns)
        if F_ret.empty:
            raise ValueError("no regression dates succeeded; check exposures/fundamentals coverage")

        # ---- covariance & specific risk -------------------------------------------------------------
        say("Estimating factor covariance and specific risk...")
        wts = _ewma_weights(len(F_ret), spec.halflife_days)
        F_cov = _weighted_cov(F_ret.values, wts) * TRADING_DAYS
        F_cov = (1 - spec.cov_shrink) * F_cov + spec.cov_shrink * np.diag(np.diag(F_cov))
        factor_cov = pd.DataFrame(F_cov, index=core_factors, columns=core_factors)
        spec_var = _ewma_var(U, wts) * TRADING_DAYS
        med = float(np.nanmedian(spec_var.values)) if spec_var.notna().any() else 0.04
        spec_var = (1 - spec.specific_shrink) * spec_var + spec.specific_shrink * med

        # ---- final exposures + regression-based fill for ETFs / missing names -------------------------
        X_end = builds[t_end].exposures[core_factors].copy()
        missing = X_end.index[X_end.isna().any(axis=1)]
        filled = 0
        if len(missing):
            say(f"Filling exposures for {len(missing)} names by time-series regression...")
            R = rets.loc[F_ret.index, missing]
            for sym in missing:
                y = R[sym].dropna()
                if len(y) < spec.min_obs_per_symbol:
                    continue
                Fm = F_ret.loc[y.index].values
                beta, resid = _ridge(Fm, y.values, spec.ridge_lambda)
                X_end.loc[sym] = beta
                spec_var.loc[sym] = float(np.average(resid ** 2, weights=_ewma_weights(len(resid), spec.halflife_days))) * TRADING_DAYS
                filled += 1
        keep = X_end.notna().all(axis=1) & spec_var.reindex(X_end.index).notna()
        X_end = X_end[keep]
        spec_var = spec_var.reindex(X_end.index)

        # ---- optional macro block -----------------------------------------------------------------
        if spec.use_macro and macro_levels is not None and not macro_levels.empty:
            say("Estimating macro betas...")
            from ..data.macro import macro_shocks
            shocks = macro_shocks(macro_levels, spec.macro_columns).reindex(F_ret.index).ffill().dropna()
            common = shocks.index.intersection(F_ret.index)
            if len(common) >= spec.min_obs_per_symbol:
                # orthogonalise stock residuals to macro shocks: beta of residual returns to shocks
                Ures = U.loc[common].fillna(0.0)
                M = shocks.loc[common].values
                M = (M - M.mean(0)) / (M.std(0) + 1e-12)
                betas = np.linalg.lstsq(M, Ures.values, rcond=None)[0]        # k x n
                mcols = [f"macro:{c}" for c in shocks.columns]
                Bm = pd.DataFrame(betas.T, index=Ures.columns, columns=mcols).reindex(X_end.index).fillna(0.0)
                X_end = pd.concat([X_end, Bm], axis=1)
                Mcov = np.cov(M.T) * TRADING_DAYS
                big = np.zeros((len(core_factors) + len(mcols),) * 2)
                big[: len(core_factors), : len(core_factors)] = factor_cov.values
                big[len(core_factors):, len(core_factors):] = Mcov
                factor_cov = pd.DataFrame(big, index=list(core_factors) + mcols, columns=list(core_factors) + mcols)
                F_ret = pd.concat([F_ret, pd.DataFrame(M, index=common, columns=mcols)], axis=1)

        diagnostics = {
            "model_kind": "barra_lite",
            "n_dates": int(len(F_ret)), "n_symbols": int(len(X_end)), "n_filled_by_regression": filled,
            "avg_r2": float(np.mean(r2s)) if r2s else None,
            "fit_start": str(F_ret.index.min().date()), "fit_end": str(F_ret.index.max().date()),
            "median_specific_vol": float(np.sqrt(np.nanmedian(spec_var.values))),
            "styles": list(spec.styles), "sectors": sector_cols, "use_macro": bool(spec.use_macro),
            "style_descriptions": {s: STYLE_DEFINITIONS[s].description for s in spec.styles if s in STYLE_DEFINITIONS},
        }
        say("Risk model fit complete.")
        return {"exposures": X_end, "factor_cov": factor_cov, "specific_var": spec_var, "factor_returns": F_ret, "diagnostics": diagnostics,
                "as_of": t_end.date(), "style_cols": list(spec.styles), "residuals": U}


# ====================================================================================== numerics
def _constrained_wls(X: np.ndarray, r: np.ndarray, sqrt_w: np.ndarray, cols: list[str], sector_cols: list[str],
                     capw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """WLS with the constraint sum_s (capweight of sector s) * f_s = 0 via reparameterisation."""
    k = X.shape[1]
    if sector_cols:
        idx = [cols.index(c) for c in sector_cols]
        sw = np.array([capw[X[:, i] > 0].sum() for i in idx])
        sw = sw / sw.sum() if sw.sum() > 0 else np.ones(len(idx)) / len(idx)
        # Reduced parameter vector drops the last sector; f_last = -sum_j(sw_j f_j) / sw_last.
        # R maps reduced coefficients (k-1) back to the full vector (k).
        last = idx[-1]
        R = np.delete(np.eye(k), last, axis=1)
        for j, i in enumerate(idx[:-1]):
            R[last, _col_after_delete(i, last)] = -sw[j] / sw[-1]
        Xr = X @ R
    else:
        R = np.eye(k)
        Xr = X
    Xw = Xr * sqrt_w[:, None]
    rw = r * sqrt_w
    beta_r, *_ = np.linalg.lstsq(Xw, rw, rcond=None)
    f = R @ beta_r
    u = r - X @ f
    tss = float(((rw - rw.mean()) ** 2).sum())
    r2 = 1.0 - float(((u * sqrt_w) ** 2).sum()) / tss if tss > 0 else 0.0
    return f, u, r2


def _col_after_delete(i: int, deleted: int) -> int:
    return i if i < deleted else i - 1


def _ewma_weights(n: int, halflife: int) -> np.ndarray:
    lam = 0.5 ** (1.0 / max(halflife, 1))
    w = lam ** np.arange(n)[::-1]
    return w / w.sum()


def _weighted_cov(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = np.average(F, axis=0, weights=w)
    Fc = F - mu
    return (Fc * w[:, None]).T @ Fc


def _ewma_var(U: pd.DataFrame, w: np.ndarray) -> pd.Series:
    out = {}
    for sym in U.columns:
        u = U[sym].values
        ok = ~np.isnan(u)
        if ok.sum() < 20:
            out[sym] = np.nan
            continue
        ww = w[ok] / w[ok].sum()
        out[sym] = float((ww * u[ok] ** 2).sum())
    return pd.Series(out)


def _ridge(F: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Relative ridge: penalty scaled by each factor's own sum of squares, so `lam` is a shrinkage fraction
    (0.05 -> ~5% shrink toward zero) regardless of factor-return scale. The market beta is never shrunk."""
    FtF = F.T @ F
    pen = lam * np.diag(np.diag(FtF))
    pen[0, 0] = 0.0
    beta = np.linalg.solve(FtF + pen + 1e-12 * np.eye(F.shape[1]), F.T @ y)
    return beta, y - F @ beta
