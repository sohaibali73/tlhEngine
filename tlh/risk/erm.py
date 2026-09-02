"""Equity Risk Model (ERM): Barra-style multi-descriptor model with production covariance adjustments.

Pipeline
  1. Descriptor composites -> standardized style exposures (monthly grid, held between refreshes), orthogonalised
     where Barra does (resvol on beta+size, liquidity on size, non-linear size on size); industry dummies at the
     chosen GICS level; market factor with the cap-weighted-industry-sum-to-zero constraint.
  2. Daily cross-sectional WLS (sqrt-cap weights capped at a percentile), optional Huber robust regression;
     factor return t-stats and R² recorded.
  3. Factor covariance: EWMA volatility (short half-life) x EWMA correlation (long half-life), Newey-West
     autocorrelation correction, eigenfactor risk adjustment (Monte Carlo), volatility regime adjustment (VRA).
  4. Specific risk: EWMA residual variance with VRA, Bayesian shrinkage toward the cap-decile mean, structural
     (characteristic) model for names with too little history.
  5. Names without characteristics (ETFs) are filled by ridge time-series regression on factor returns.

Output is the same `FittedRiskModel` as barra_lite, so the optimizer, baskets and strategies are unchanged.
This module is AI-editable (ai/registry.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .descriptors import (
    DESCRIPTORS,
    ERM_DEFAULT_STYLES,
    ORTHOGONALISE,
    STYLE_COMPOSITES,
    STYLE_DESCRIPTIONS,
    DescriptorInputs,
    orthogonalise,
    standardize,
)

log = logging.getLogger(__name__)
TRADING_DAYS = 252


@dataclass
class ERMOptions:
    industry_level: str = "gics_industry_group"     # gics_sector | gics_industry_group | gics_industry
    styles: list[str] = field(default_factory=lambda: list(ERM_DEFAULT_STYLES))
    robust: bool = False                            # Huber M-estimation instead of plain WLS
    weight_cap_pct: float = 95.0                    # cap on sqrt-cap regression weights
    nw_lags: int = 2
    hl_vol: int = 84
    hl_corr: int = 504
    eigen_adjust: bool = True
    eigen_sims: int = 200
    eigen_scale: float = 1.2
    vra: bool = True
    hl_vra: int = 42
    specific_hl: int = 84
    specific_shrink: float = 0.3
    min_obs: int = 120


def _ew(n: int, hl: int) -> np.ndarray:
    w = 0.5 ** (np.arange(n)[::-1] / max(hl, 1))
    return w / w.sum()


# ====================================================================================== exposures
def build_erm_exposures(d: DescriptorInputs, opts: ERMOptions, capw: pd.Series) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    symbols = d.prices.columns
    raw_cache: dict[str, pd.Series] = {}
    coverage: dict[str, int] = {}

    def raw(name: str) -> pd.Series:
        if name not in raw_cache:
            try:
                raw_cache[name] = DESCRIPTORS[name](d).reindex(symbols)
            except Exception as e:  # a broken descriptor must not kill the fit
                log.warning("descriptor %s failed: %s", name, e)
                raw_cache[name] = pd.Series(np.nan, index=symbols)
            coverage[name] = int(raw_cache[name].notna().sum())
        return raw_cache[name]

    is_etp = d.securities.get("subtype1", pd.Series(index=symbols, dtype=object)).reindex(symbols).fillna("").str.lower().str.startswith("exchange traded")
    capw_std = capw.where(~is_etp, 0.0)
    X = pd.DataFrame(index=symbols)
    X["market"] = 1.0
    styles: dict[str, pd.Series] = {}
    for s in opts.styles:
        if s == "midcap":
            continue
        comp = STYLE_COMPOSITES.get(s)
        if not comp:
            continue
        parts, wts = [], []
        for desc, wt in comp.items():
            z = standardize(raw(desc).where(~is_etp), capw_std)
            if z.notna().sum() >= 10:
                parts.append(z.fillna(0.0) * wt)
                wts.append((z.notna(), wt))
        if not parts:
            styles[s] = pd.Series(np.nan, index=symbols)
            continue
        num = sum(parts)
        den = sum(m.astype(float) * w for m, w in wts)
        comp_z = (num / den.where(den > 0)).where(den > 0)
        styles[s] = standardize(comp_z, capw_std)
    if "midcap" in opts.styles and "size" in styles:
        cube = styles["size"] ** 3
        styles["midcap"] = standardize(orthogonalise(cube, [styles["size"]], capw_std), capw_std)
    for s, on in ORTHOGONALISE.items():
        if s in styles and all(o in styles for o in on):
            styles[s] = standardize(orthogonalise(styles[s], [styles[o] for o in on], capw_std), capw_std)
    # drop styles with no usable descriptors in this dataset (e.g. liquidity without volume data)
    style_cols = [s for s in opts.styles if s in styles and styles[s].notna().sum() >= 10]
    for s in style_cols:
        X[s] = styles[s]
    # industries
    level = opts.industry_level if opts.industry_level in d.securities else ("gics_sector" if "gics_sector" in d.securities else None)
    ind_cols: list[str] = []
    if level:
        ind = d.securities[level].reindex(symbols)
        for name in sorted(ind.dropna().unique()):
            col = f"ind:{name}"
            X[col] = (ind == name).astype(float).where(ind.notna())
            ind_cols.append(col)
    return X, style_cols, ind_cols, coverage


# ====================================================================================== regression
def _constrained_design(X: np.ndarray, cols: list[str], ind_cols: list[str], capw: np.ndarray) -> np.ndarray:
    """Reparameterisation matrix R (k x k-1) enforcing sum_j sw_j f_j = 0 over industries."""
    k = X.shape[1]
    if not ind_cols:
        return np.eye(k)
    idx = [cols.index(c) for c in ind_cols]
    sw = np.array([capw[X[:, i] > 0].sum() for i in idx])
    sw = sw / sw.sum() if sw.sum() > 0 else np.ones(len(idx)) / len(idx)
    last = idx[-1]
    R = np.delete(np.eye(k), last, axis=1)
    for j, i in enumerate(idx[:-1]):
        R[last, i if i < last else i - 1] = -sw[j] / sw[-1]
    return R


def _wls(X: np.ndarray, r: np.ndarray, w: np.ndarray, R: np.ndarray, robust: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return factor returns f (k), residuals u (n), t-stats (k), R²."""
    Xr = X @ R
    sw = np.sqrt(w)
    Xw, rw = Xr * sw[:, None], r * sw
    if robust:
        try:
            import statsmodels.api as sm
            res = sm.RLM(rw, Xw, M=sm.robust.norms.HuberT()).fit(maxiter=30)
            beta_r = np.asarray(res.params)
        except Exception:
            beta_r, *_ = np.linalg.lstsq(Xw, rw, rcond=None)
    else:
        beta_r, *_ = np.linalg.lstsq(Xw, rw, rcond=None)
    f = R @ beta_r
    u = r - X @ f
    dof = max(len(r) - Xr.shape[1], 1)
    s2 = float(((u * sw) ** 2).sum() / dof)
    try:
        cov_r = s2 * np.linalg.pinv(Xw.T @ Xw)
        cov_f = R @ cov_r @ R.T
        se = np.sqrt(np.clip(np.diag(cov_f), 1e-18, None))
        t = f / se
    except Exception:
        t = np.full(len(f), np.nan)
    tss = float(((rw - rw.mean()) ** 2).sum())
    r2 = 1.0 - float(((u * sw) ** 2).sum()) / tss if tss > 0 else 0.0
    return f, u, t, r2


# ====================================================================================== covariance machinery
def ewma_cov(F: np.ndarray, hl: int, nw_lags: int = 0) -> np.ndarray:
    T = F.shape[0]
    w = _ew(T, hl)
    mu = np.average(F, axis=0, weights=w)
    Fc = F - mu
    C0 = (Fc * w[:, None]).T @ Fc
    if nw_lags <= 0:
        return C0
    C = C0.copy()
    for lag in range(1, nw_lags + 1):
        wl = w[lag:] / w[lag:].sum()
        Cl = (Fc[lag:] * wl[:, None]).T @ Fc[:-lag]
        C += (1 - lag / (nw_lags + 1)) * (Cl + Cl.T)
    return C


def factor_covariance(F: np.ndarray, opts: ERMOptions) -> tuple[np.ndarray, dict]:
    diag_info: dict = {}
    Cv = ewma_cov(F, opts.hl_vol, opts.nw_lags)
    Cc = ewma_cov(F, opts.hl_corr, opts.nw_lags)
    vol = np.sqrt(np.clip(np.diag(Cv), 1e-14, None))
    sd_c = np.sqrt(np.clip(np.diag(Cc), 1e-14, None))
    corr = Cc / np.outer(sd_c, sd_c)
    corr = np.clip(0.5 * (corr + corr.T), -1, 1)
    np.fill_diagonal(corr, 1.0)
    Fcov = corr * np.outer(vol, vol)
    Fcov = _psd(Fcov)
    if opts.vra:
        lam = vra_multiplier(F, vol_hl=opts.hl_vol, hl=opts.hl_vra)
        Fcov = Fcov * lam ** 2
        diag_info["vra_lambda"] = float(lam)
    if opts.eigen_adjust:
        Fcov, gammas = eigen_adjust(Fcov, T=F.shape[0], sims=opts.eigen_sims, scale=opts.eigen_scale)
        diag_info["eigen_gamma_min"], diag_info["eigen_gamma_max"] = float(np.min(gammas)), float(np.max(gammas))
    return Fcov, diag_info


def vra_multiplier(F: np.ndarray, vol_hl: int, hl: int) -> float:
    """Volatility regime adjustment: EWMA of the cross-sectional bias statistic of standardized factor returns."""
    T, k = F.shape
    if T < 60:
        return 1.0
    lam_v = 0.5 ** (1 / max(vol_hl, 1))
    var = np.var(F[:30], axis=0) + 1e-14
    B = []
    for t in range(30, T):
        z = F[t] / np.sqrt(var)
        B.append(np.mean(z ** 2))
        var = lam_v * var + (1 - lam_v) * F[t] ** 2
    w = _ew(len(B), hl)
    return float(np.sqrt(np.clip((w * np.array(B)).sum(), 0.25, 4.0)))


def eigen_adjust(Fcov: np.ndarray, T: int, sims: int = 200, scale: float = 1.2, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Menchero-Orr-Wang eigenfactor risk adjustment: inflate variances of eigenfactors that sample covariance
    systematically under-estimates (small eigenvalues) using Monte Carlo from the fitted covariance."""
    vals, vecs = np.linalg.eigh(Fcov)
    vals = np.clip(vals, 1e-14, None)
    rng = np.random.default_rng(seed)
    k = len(vals)
    ratios = np.zeros(k)
    L = vecs @ np.diag(np.sqrt(vals))
    for _ in range(sims):
        Z = rng.standard_normal((T, k))
        Fs = Z @ L.T
        Cs = np.cov(Fs, rowvar=False)
        vs, us = np.linalg.eigh(Cs)
        vs = np.clip(vs, 1e-14, None)
        true_var = np.einsum("ij,jk,ki->i", us.T, Fcov, us)   # variance of simulated eigenportfolios under the true cov
        ratios += true_var / vs
    ratios /= sims
    gamma = scale * (np.sqrt(ratios) - 1.0) + 1.0
    gamma = np.clip(gamma, 0.8, 3.0)
    adj = vecs @ np.diag(gamma ** 2 * vals) @ vecs.T
    return _psd(adj), gamma


def _psd(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    return vecs @ np.diag(np.clip(vals, 1e-12, None)) @ vecs.T


# ====================================================================================== specific risk
def specific_risk(U: pd.DataFrame, X: pd.DataFrame, capw: pd.Series, opts: ERMOptions) -> tuple[pd.Series, dict]:
    """Annualised specific variance per symbol with VRA, Bayesian shrinkage toward cap-decile mean, and a
    structural fallback for names with < min_obs residuals."""
    T = len(U)
    w = _ew(T, opts.specific_hl)
    raw_var = {}
    nobs = U.notna().sum()
    for s in U.columns:
        u = U[s].values
        ok = ~np.isnan(u)
        if ok.sum() < 20:
            continue
        ww = w[ok] / w[ok].sum()
        raw_var[s] = float((ww * u[ok] ** 2).sum())
    sv = pd.Series(raw_var)
    info: dict = {}
    # VRA for specific returns: bias statistic across stocks with a valid residual that day (NaNs excluded, not zero-filled)
    if opts.vra and T > 90:
        lam_v = 0.5 ** (1 / max(opts.specific_hl, 1))
        Uv = U.values
        import warnings
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            var = np.nanvar(Uv[:60], axis=0)
        var = np.where(np.isfinite(var) & (var > 1e-10), var, np.nan)
        B = []
        for t in range(60, T):
            u = Uv[t]
            ok = np.isfinite(u) & np.isfinite(var) & (var > 1e-10)
            if ok.sum() >= 20:
                B.append(float(np.mean(u[ok] ** 2 / var[ok])))
            upd = np.isfinite(u)
            var = np.where(upd, lam_v * np.where(np.isfinite(var), var, u ** 2) + (1 - lam_v) * u ** 2, var)
        if B:
            lam_s = float(np.sqrt(np.clip((_ew(len(B), opts.hl_vra) * np.array(B)).sum(), 0.25, 4.0)))
            lam_s = float(np.clip(lam_s, 0.5, 2.0))
            sv = sv * lam_s ** 2
            info["specific_vra_lambda"] = lam_s
    sigma = np.sqrt(sv)
    # Bayesian shrinkage toward cap-decile mean
    caps = capw.reindex(sigma.index).fillna(0.0)
    deciles = pd.qcut(caps.rank(method="first"), 10, labels=False) if len(caps) >= 20 else pd.Series(0, index=caps.index)
    prior = sigma.groupby(deciles).transform("mean")
    delta = sigma.groupby(deciles).transform("std").fillna(sigma.std())
    q = opts.specific_shrink
    v = (q * (sigma - prior).abs()) / (delta + q * (sigma - prior).abs() + 1e-12)
    sigma_shr = v * prior + (1 - v) * sigma
    # structural model for names with poor history
    good = sigma_shr.index[nobs.reindex(sigma_shr.index) >= opts.min_obs]
    poor = [s for s in X.index if s not in set(good)]
    n_struct = 0
    if len(good) >= 30 and poor:
        Xg = X.loc[good].fillna(0.0).values
        y = np.log(sigma_shr.loc[good].values)
        A = np.column_stack([np.ones(len(good)), Xg])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        for s in poor:
            xs = X.loc[s].fillna(0.0).values
            sigma_shr.loc[s] = float(np.exp(b[0] + xs @ b[1:]) * 1.05)
            n_struct += 1
    info["n_structural_specific"] = n_struct
    return (sigma_shr ** 2 * TRADING_DAYS).astype(float), info


# ====================================================================================== fit
def fit_erm(prices: pd.DataFrame, securities: pd.DataFrame, fundamentals: pd.DataFrame, volume: pd.DataFrame | None,
            lookback_days: int, refresh_days: int, opts: ERMOptions, progress=None) -> dict:
    """Return the ingredients of a FittedRiskModel (exposures, factor_cov, specific_var, factor_returns, diagnostics)."""
    from .model import _ridge  # shared helper

    say = progress or (lambda m: None)
    prices = prices.sort_index().dropna(axis=1, thresh=opts.min_obs)
    securities = securities.reindex(prices.columns)
    fundamentals = fundamentals.reindex(prices.columns)
    if volume is not None:
        volume = volume.reindex(columns=prices.columns)
    rets = prices.pct_change().iloc[1:].clip(-0.25, 0.25)
    fit_dates = rets.index[-lookback_days:]
    t_end = fit_dates[-1]
    shares = pd.to_numeric(securities.get("shares_outstanding"), errors="coerce").reindex(prices.columns) if "shares_outstanding" in securities else pd.Series(np.nan, index=prices.columns)
    is_etp = securities.get("subtype1", pd.Series(index=prices.columns, dtype=object)).fillna("").str.lower().str.startswith("exchange traded")
    # cap-weighted market return (stocks only)
    capm = (shares.where(~is_etp).values[None, :] * prices.values)
    capm = np.nan_to_num(capm)
    wprev = capm[:-1] / np.clip(capm[:-1].sum(axis=1, keepdims=True), 1e-12, None)
    market_ret = pd.Series((wprev * np.nan_to_num(rets.values)).sum(axis=1), index=rets.index)

    say("ERM: building descriptor exposures on the monthly grid…")
    grid = list(fit_dates[::refresh_days])
    if grid[-1] != t_end:
        grid.append(t_end)
    builds: dict[pd.Timestamp, tuple[pd.DataFrame, pd.Series]] = {}
    style_cols: list[str] = []
    ind_cols: list[str] = []
    coverage: dict = {}
    for t in grid:
        d = DescriptorInputs(prices=prices, t=t, fundamentals=fundamentals, securities=securities, volume=volume, market_ret=market_ret)
        capw = d.mktcap().where(~is_etp, 0.0).fillna(0.0).clip(lower=0.0)
        X, style_cols, ind_cols, cov = build_erm_exposures(d, opts, capw)
        builds[t] = (X, capw)
        coverage = cov
    core = ["market"] + style_cols + ind_cols

    say(f"ERM: estimating factor returns over {len(fit_dates)} days ({'Huber' if opts.robust else 'WLS'})…")
    grid_arr = np.array(grid, dtype="datetime64[ns]")
    f_rows, u_rows, t_rows, r2s = [], [], [], []
    for t in fit_dates:
        k = int(np.searchsorted(grid_arr, np.datetime64(t), side="right")) - 1
        X, capw = builds[grid[max(k, 0)]]
        Xc = X[core]
        r = rets.loc[t]
        ok = Xc.notna().all(axis=1) & r.notna() & (capw > 0)
        if ok.sum() < len(core) + 5:
            continue
        w = np.sqrt(capw[ok].values)
        w = np.minimum(w, np.percentile(w, opts.weight_cap_pct))
        w = w / w.sum()
        R = _constrained_design(Xc[ok].values, core, ind_cols, capw[ok].values)
        f, u, tst, r2 = _wls(Xc[ok].values, r[ok].values, w, R, opts.robust)
        f_rows.append(pd.Series(f, index=core, name=t))
        u_rows.append(pd.Series(u, index=Xc.index[ok], name=t))
        t_rows.append(pd.Series(tst, index=core, name=t))
        r2s.append((t, r2))
    F_ret = pd.DataFrame(f_rows)
    U = pd.DataFrame(u_rows).reindex(columns=prices.columns)
    Tst = pd.DataFrame(t_rows)
    if F_ret.empty:
        raise ValueError("ERM: no regression dates succeeded")

    say("ERM: covariance (EWMA vol x corr, Newey-West, eigen-adjust, VRA)…")
    Fcov, cov_info = factor_covariance(F_ret.values, opts)
    Fcov = Fcov * TRADING_DAYS
    factor_cov = pd.DataFrame(Fcov, index=core, columns=core)

    say("ERM: specific risk (VRA, Bayesian shrinkage, structural fallback)…")
    X_end, capw_end = builds[t_end]
    X_end = X_end[core].copy()
    spec_var, spec_info = specific_risk(U, X_end.fillna(0.0), capw_end, opts)

    # ETFs / missing characteristics: ridge on factor returns
    missing = X_end.index[X_end.isna().any(axis=1)]
    filled = 0
    for sym in missing:
        y = rets.loc[F_ret.index, sym].dropna()
        if len(y) < opts.min_obs:
            continue
        Fm = F_ret.loc[y.index].values
        b, resid = _ridge(Fm, y.values, 0.05)
        X_end.loc[sym] = b
        spec_var.loc[sym] = float(np.average(resid ** 2, weights=_ew(len(resid), opts.specific_hl))) * TRADING_DAYS
        filled += 1
    keep = X_end.notna().all(axis=1) & spec_var.reindex(X_end.index).notna()
    X_end = X_end[keep]
    spec_var = spec_var.reindex(X_end.index)

    r2_series = pd.Series(dict(r2s))
    tabs = Tst.abs()
    diagnostics = {
        "model_kind": "erm", "n_dates": int(len(F_ret)), "n_symbols": int(len(X_end)), "n_filled_by_regression": filled,
        "avg_r2": float(r2_series.mean()), "r2_series": [[str(k.date()), round(float(v), 4)] for k, v in r2_series.iloc[::5].items()],
        "fit_start": str(F_ret.index.min().date()), "fit_end": str(F_ret.index.max().date()),
        "factor_vol_annual": {c: float(v) for c, v in zip(core, np.sqrt(np.diag(Fcov)), strict=True)},
        "t_stats": {c: {"mean_abs_t": float(tabs[c].mean()), "pct_significant": float((tabs[c] > 2).mean())} for c in core if c in tabs},
        "median_specific_vol": float(np.sqrt(np.nanmedian(spec_var.values))),
        "styles": style_cols, "industries": ind_cols, "industry_level": opts.industry_level, "descriptor_coverage": coverage,
        "style_descriptions": {s: STYLE_DESCRIPTIONS.get(s, "") for s in style_cols}, "robust": opts.robust, "nw_lags": opts.nw_lags,
        "hl_vol": opts.hl_vol, "hl_corr": opts.hl_corr, **cov_info, **spec_info,
        "style_exposure_corr": X_end[style_cols].corr().round(3).to_dict() if style_cols else {},
    }
    say("ERM fit complete.")
    return {"exposures": X_end, "factor_cov": factor_cov, "specific_var": spec_var, "factor_returns": F_ret, "diagnostics": diagnostics,
            "as_of": t_end.date(), "style_cols": style_cols}
