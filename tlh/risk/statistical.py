"""Statistical and dynamic risk models that plug into the same `FittedRiskModel` surface as barra_lite and the ERM.

Estimators
----------
calibrated   The Potomac calibrated covariance from the 2026 calibration study: a fixed lookback window (126 days
             recommended), equal or exponential weighting (half-life = 0.35 x lookback), sample or Ledoit-Wolf
             constant-correlation shrinkage. Represented as a factor model through its eigen-decomposition
             (`stat:k` factors) so the optimizer, baskets and analytics need no special case. The exact shrunk
             covariance is reproduced on the diagonal and to within the truncated eigen-tail off-diagonal.
pca          Asymptotic principal components on weighted returns; the number of factors is chosen by the
             Ahn-Horenstein eigenvalue-ratio test (or fixed). Specific variance = weighted residual variance.
hybrid       Any fundamental model (ERM / barra_lite) plus PCA factors extracted from its residual returns, which
             picks up co-movement the descriptors miss (Axioma-style hybrid). Used by model.py.

Dynamic covariance post-processors (applied to a fitted model's factor returns)
--------------------------------------------------------------------------------
garch        GARCH(1,1) per factor (variance-targeted maximum likelihood) forecast over `horizon_days`, combined with
             an EWMA correlation matrix -> forward-looking factor covariance for the chosen horizon.
regime       Two-state (calm / stress) factor covariance mixed by the probability of the stress state given the
             current market-volatility z-score.

All covariances are annualised (x252). This module is AI-editable (ai/registry.py).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252


# ====================================================================================== weights & estimators
def obs_weights(n: int, weighting: str = "equal", halflife: float | None = None, halflife_ratio: float = 0.35) -> np.ndarray:
    """Observation weights (oldest first, sum to one). Exponential half-life defaults to 0.35 x n (calibration convention)."""
    if weighting == "exponential":
        hl = halflife or max(halflife_ratio * n, 1.0)
        lam = 0.5 ** (1.0 / hl)
        w = lam ** np.arange(n)[::-1]
    else:
        w = np.ones(n)
    return w / w.sum()


def effective_n(w: np.ndarray) -> float:
    """Kish effective sample size 1 / sum(w^2)."""
    return float(1.0 / np.sum(np.asarray(w) ** 2))


def weighted_cov(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = np.average(R, axis=0, weights=w)
    Rc = R - mu
    return (Rc * w[:, None]).T @ Rc


def ledoit_wolf_cc(R: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage toward the constant-correlation target, with optional observation weights.

    Returns (shrunk covariance, shrinkage intensity delta in [0, 1]). The diagonal is untouched."""
    T, n = R.shape
    w = np.ones(T) / T if w is None else np.asarray(w) / np.sum(w)
    mu = np.average(R, axis=0, weights=w)
    X = R - mu
    S = (X * w[:, None]).T @ X
    var = np.diag(S).copy()
    sd = np.sqrt(np.maximum(var, 1e-18))
    corr = S / np.outer(sd, sd)
    rbar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)
    # pi-hat: asymptotic variance of the sample covariance entries (weighted analogue of the LW estimator)
    Xw = X * np.sqrt(w)[:, None] * math.sqrt(T)          # rescale so sums behave like the equal-weight formula
    X2 = Xw ** 2
    piMat = (X2.T @ X2) / T - S ** 2
    pihat = piMat.sum()
    # rho-hat
    term1 = ((Xw ** 3).T @ Xw) / T
    help_ = (Xw.T @ Xw) / T
    helpDiag = np.diag(help_)
    term2 = helpDiag[:, None] * S
    term3 = help_ * var[:, None]
    term4 = var[:, None] * S
    thetaMat = term1 - term2 - term3 + term4
    np.fill_diagonal(thetaMat, 0.0)
    ratio = np.divide(sd[:, None], sd[None, :], out=np.ones_like(S), where=sd[None, :] > 0)
    rhohat = np.trace(piMat) + rbar * (ratio * thetaMat).sum()
    gammahat = np.linalg.norm(S - F, "fro") ** 2
    kappa = (pihat - rhohat) / gammahat if gammahat > 0 else 0.0
    n_eff = effective_n(w)
    delta = float(min(1.0, max(0.0, kappa / n_eff)))
    Sigma = delta * F + (1.0 - delta) * S
    return Sigma, delta


def covariance_estimate(R: pd.DataFrame, lookback: int, weighting: str = "equal", estimator: str = "ledoit_wolf",
                        halflife_ratio: float = 0.35) -> tuple[pd.DataFrame, dict]:
    """Daily covariance of the last `lookback` rows of `R` (date x symbol daily returns, no NaN)."""
    Rw = R.iloc[-lookback:]
    w = obs_weights(len(Rw), weighting, halflife_ratio=halflife_ratio)
    X = Rw.values.astype(float)
    if estimator == "ledoit_wolf":
        S, delta = ledoit_wolf_cc(X, w)
    elif estimator == "sample":
        S, delta = weighted_cov(X, w), 0.0
    else:
        raise ValueError(f"unknown estimator {estimator}")
    info = {"lookback": int(len(Rw)), "weighting": weighting, "estimator": estimator, "shrinkage_delta": float(delta),
            "effective_n": effective_n(w), "halflife": (halflife_ratio * len(Rw)) if weighting == "exponential" else None}
    return pd.DataFrame(S, index=R.columns, columns=R.columns), info


# ====================================================================================== eigen / factor representation
def eigen_factorise(Sigma: pd.DataFrame, n_factors: int | None = None, explained: float = 0.95, floor_frac: float = 0.05,
                    prefix: str = "stat") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Write Sigma ~= X F X' + D with X = V_k sqrt(L_k), F = I_k, D = diag(Sigma) - diag(X X') (floored).

    `n_factors` None -> the smallest k that explains `explained` of total variance (at most n/2, at least 1)."""
    vals, vecs = np.linalg.eigh(Sigma.values)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.maximum(vals[order], 0.0), vecs[:, order]
    n = len(vals)
    if n_factors is None:
        cum = np.cumsum(vals) / max(vals.sum(), 1e-18)
        k = int(np.searchsorted(cum, explained) + 1)
        k = max(1, min(k, max(1, n // 2)))
    else:
        k = max(1, min(int(n_factors), n))
    X = vecs[:, :k] * np.sqrt(vals[:k])
    cols = [f"{prefix}:{i + 1}" for i in range(k)]
    diag = np.diag(Sigma.values)
    D = diag - (X ** 2).sum(axis=1)
    D = np.maximum(D, floor_frac * diag)
    F = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    info = {"n_factors": k, "explained_variance": float(vals[:k].sum() / max(vals.sum(), 1e-18)), "eigenvalues_top": vals[:min(k, 10)].tolist()}
    return pd.DataFrame(X, index=Sigma.index, columns=cols), F, pd.Series(D, index=Sigma.index), info


def choose_n_factors(eigvals: np.ndarray, kmax: int | None = None) -> int:
    """Ahn & Horenstein (2013) eigenvalue-ratio estimator of the number of factors."""
    v = np.sort(np.maximum(np.asarray(eigvals, dtype=float), 1e-18))[::-1]
    kmax = kmax or max(1, min(len(v) - 2, 30))
    ratios = v[:kmax] / v[1:kmax + 1]
    return int(np.argmax(ratios) + 1)


# ====================================================================================== fits
@dataclass
class StatOptions:
    lookback: int = 126
    weighting: str = "equal"          # equal | exponential
    estimator: str = "ledoit_wolf"    # sample | ledoit_wolf
    halflife_ratio: float = 0.35
    n_factors: int | None = None      # None: auto
    explained: float = 0.95           # for eigen truncation of the calibrated covariance
    min_obs: int = 60
    winsor: float = 0.25


def _clean_returns(prices: pd.DataFrame, lookback: int, min_obs: int, winsor: float) -> pd.DataFrame:
    rets = prices.sort_index().pct_change().iloc[1:].iloc[-lookback:]
    rets = rets.clip(-winsor, winsor)
    ok = rets.notna().sum() >= max(min_obs, int(0.9 * len(rets)))
    rets = rets.loc[:, ok]
    return rets.fillna(0.0)


def fit_calibrated(prices: pd.DataFrame, opts: StatOptions, progress=None) -> dict:
    say = progress or (lambda m: None)
    say(f"Calibrated covariance: {opts.lookback}d {opts.weighting} {opts.estimator}…")
    R = _clean_returns(prices, opts.lookback, opts.min_obs, opts.winsor)
    if R.shape[1] < 3:
        raise ValueError("too few symbols with complete history for the statistical model")
    Sigma, info = covariance_estimate(R, opts.lookback, opts.weighting, opts.estimator, opts.halflife_ratio)
    Sigma_ann = Sigma * TRADING_DAYS
    X, F, D, einfo = eigen_factorise(Sigma_ann, opts.n_factors, opts.explained)
    # factor "returns": projections of returns on the eigenvectors (scores), for charts / dynamic post-processing
    V = X.values / np.sqrt(np.maximum(np.diag(F.values), 1e-18))      # unit eigenvectors
    scores = pd.DataFrame(R.values @ V / np.sqrt(np.maximum((V ** 2).sum(axis=0), 1e-18)), index=R.index, columns=X.columns)
    vol_med = float(np.sqrt(np.nanmedian(np.diag(Sigma_ann.values))))
    diag = {"model_kind": "statistical", **info, **einfo, "n_symbols": int(R.shape[1]), "n_dates": int(len(R)),
            "median_total_vol": vol_med, "median_specific_vol": float(np.sqrt(np.nanmedian(D.values))),
            "avg_r2": float(einfo["explained_variance"]), "fit_start": str(R.index.min().date()), "fit_end": str(R.index.max().date()),
            "factor_vol_annual": {c: 1.0 for c in X.columns}, "styles": [], "sectors": [],
            "note": "Eigen-factor representation of the calibrated covariance; stat:k factors have unit variance by construction."}
    say("Calibrated covariance ready.")
    return {"exposures": X, "factor_cov": F, "specific_var": D, "factor_returns": scores, "diagnostics": diag,
            "as_of": R.index[-1].date(), "style_cols": [], "covariance": Sigma_ann}


def fit_pca(prices: pd.DataFrame, opts: StatOptions, progress=None) -> dict:
    say = progress or (lambda m: None)
    say("PCA risk model: extracting principal components…")
    R = _clean_returns(prices, opts.lookback, opts.min_obs, opts.winsor)
    if R.shape[1] < 3:
        raise ValueError("too few symbols with complete history for the PCA model")
    w = obs_weights(len(R), opts.weighting, halflife_ratio=opts.halflife_ratio)
    X = R.values.astype(float)
    mu = np.average(X, axis=0, weights=w)
    Xc = (X - mu) * np.sqrt(w)[:, None]
    S = Xc.T @ Xc                                     # weighted covariance (daily)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.maximum(vals[order], 0.0), vecs[:, order]
    k = opts.n_factors or choose_n_factors(vals)
    k = max(1, min(k, R.shape[1] - 1))
    V = vecs[:, :k]
    f = (X - mu) @ V                                  # factor returns (scores), date x k
    cols = [f"stat:{i + 1}" for i in range(k)]
    B = pd.DataFrame(V, index=R.columns, columns=cols)   # loadings (unit-norm eigenvectors)
    F = pd.DataFrame(np.diag(vals[:k]) * TRADING_DAYS, index=cols, columns=cols)
    resid = (X - mu) - f @ V.T
    D = pd.Series((resid ** 2 * w[:, None]).sum(axis=0) * TRADING_DAYS, index=R.columns)
    D = D.clip(lower=0.05 * pd.Series(np.diag(S) * TRADING_DAYS, index=R.columns))
    diag = {"model_kind": "pca", "n_factors": int(k), "lookback": int(len(R)), "weighting": opts.weighting,
            "explained_variance": float(vals[:k].sum() / max(vals.sum(), 1e-18)), "avg_r2": float(vals[:k].sum() / max(vals.sum(), 1e-18)),
            "n_symbols": int(R.shape[1]), "n_dates": int(len(R)), "median_specific_vol": float(np.sqrt(np.nanmedian(D.values))),
            "fit_start": str(R.index.min().date()), "fit_end": str(R.index.max().date()),
            "factor_vol_annual": {c: float(np.sqrt(F.loc[c, c])) for c in cols}, "styles": [], "sectors": [],
            "eigenvalue_ratio_k": int(choose_n_factors(vals))}
    say(f"PCA model ready ({k} components).")
    return {"exposures": B, "factor_cov": F, "specific_var": D, "factor_returns": pd.DataFrame(f, index=R.index, columns=cols),
            "diagnostics": diag, "as_of": R.index[-1].date(), "style_cols": []}


def add_statistical_factors(exposures: pd.DataFrame, factor_cov: pd.DataFrame, specific_var: pd.Series, factor_returns: pd.DataFrame,
                            residuals: pd.DataFrame, n_stat: int | None = 5, halflife: int = 126) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """Hybrid model: extract `n_stat` principal components from a fundamental model's residual returns and append them
    as `stat:k` factors (block-diagonal covariance, since residuals are orthogonal to the fitted factors by construction)."""
    U = residuals.reindex(columns=exposures.index).dropna(axis=1, thresh=int(0.8 * len(residuals))).fillna(0.0)
    if U.shape[1] < 10 or len(U) < 60:
        return exposures, factor_cov, specific_var, factor_returns, {"stat_factors": 0}
    w = obs_weights(len(U), "exponential", halflife=halflife)
    X = U.values
    mu = np.average(X, axis=0, weights=w)
    Xc = (X - mu) * np.sqrt(w)[:, None]
    S = Xc.T @ Xc
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.maximum(vals[order], 0.0), vecs[:, order]
    k = n_stat or choose_n_factors(vals, kmax=15)
    k = max(1, min(k, 15))
    V = vecs[:, :k]
    f = (X - mu) @ V
    cols = [f"stat:{i + 1}" for i in range(k)]
    B = pd.DataFrame(V, index=U.columns, columns=cols).reindex(exposures.index).fillna(0.0)
    X_new = pd.concat([exposures, B], axis=1)
    F_new = pd.DataFrame(0.0, index=list(factor_cov.index) + cols, columns=list(factor_cov.columns) + cols)
    F_new.loc[factor_cov.index, factor_cov.columns] = factor_cov.values
    F_new.loc[cols, cols] = np.diag(vals[:k]) * TRADING_DAYS
    explained = pd.Series((V ** 2 * vals[:k]).sum(axis=1) * TRADING_DAYS, index=U.columns).reindex(exposures.index).fillna(0.0)
    D_new = (specific_var - explained).clip(lower=0.2 * specific_var)
    fr = pd.DataFrame(f, index=U.index, columns=cols)
    FR_new = pd.concat([factor_returns, fr.reindex(factor_returns.index)], axis=1)
    info = {"stat_factors": int(k), "stat_explained_of_residual": float(vals[:k].sum() / max(vals.sum(), 1e-18))}
    return X_new, F_new, D_new, FR_new, info


# ====================================================================================== dynamic covariance
def garch11_fit(x: np.ndarray) -> tuple[float, float, float]:
    """GARCH(1,1) by maximum likelihood with variance targeting (omega = var * (1 - a - b)). Returns (omega, alpha, beta)."""
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = float(np.var(x)) or 1e-8

    def nll(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        om = var * (1 - a - b)
        h = np.empty_like(x)
        h[0] = var
        for t in range(1, len(x)):
            h[t] = om + a * x[t - 1] ** 2 + b * h[t - 1]
        h = np.maximum(h, 1e-12)
        return 0.5 * np.sum(np.log(h) + x ** 2 / h)

    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.02, 0.95)):
        res = minimize(nll, np.array([a0, b0]), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    a, b = float(best.x[0]), float(best.x[1])
    a, b = max(a, 0.0), max(b, 0.0)
    if a + b >= 0.999:
        a, b = 0.05, 0.90
    return var * (1 - a - b), a, b


def garch_forecast_var(x: np.ndarray, horizon_days: int = 21) -> tuple[float, dict]:
    """Average daily variance forecast over the next `horizon_days` from a GARCH(1,1) fit to `x`."""
    om, a, b = garch11_fit(x)
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = float(np.var(x)) or 1e-8
    h = var
    for t in range(1, len(x)):
        h = om + a * x[t - 1] ** 2 + b * h
    h_next = om + a * x[-1] ** 2 + b * h
    lr = om / max(1 - a - b, 1e-6)
    path = [lr + (a + b) ** s * (h_next - lr) for s in range(horizon_days)]
    return float(np.mean(path)), {"omega": om, "alpha": a, "beta": b, "persistence": a + b, "long_run_var": lr, "h_next": h_next}


def garch_factor_cov(F_ret: pd.DataFrame, horizon_days: int = 21, corr_halflife: int = 504) -> tuple[pd.DataFrame, dict]:
    """Forward-looking factor covariance: GARCH(1,1) variances per factor x EWMA correlations, annualised."""
    F = F_ret.dropna(how="all").fillna(0.0)
    n = F.shape[1]
    var_f = np.zeros(n)
    params = {}
    for j, c in enumerate(F.columns):
        try:
            var_f[j], params[c] = garch_forecast_var(F[c].values, horizon_days)
        except Exception as e:  # degenerate factor
            log.debug("garch failed for %s: %s", c, e)
            var_f[j] = float(np.var(F[c].values))
    w = obs_weights(len(F), "exponential", halflife=corr_halflife)
    S = weighted_cov(F.values, w)
    sd = np.sqrt(np.maximum(np.diag(S), 1e-18))
    C = S / np.outer(sd, sd)
    np.fill_diagonal(C, 1.0)
    sd_f = np.sqrt(var_f)
    cov = C * np.outer(sd_f, sd_f) * TRADING_DAYS
    ratio = float(np.median(var_f / np.maximum(np.diag(S), 1e-18)))
    return pd.DataFrame(cov, index=F.columns, columns=F.columns), {"garch_params": {k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in list(params.items())[:12]},
                                                                    "garch_vs_ewma_variance_ratio": ratio, "horizon_days": horizon_days}


def regime_factor_cov(F_ret: pd.DataFrame, market_col: str | None = None, window: int = 21, halflife: int = 252) -> tuple[pd.DataFrame, dict]:
    """Two-state covariance: days are labelled calm/stress by rolling market volatility versus its median; each state
    gets an EWMA covariance; the blend weight is the logistic probability of stress given today's vol z-score."""
    F = F_ret.dropna(how="all").fillna(0.0)
    mcol = market_col or ("market" if "market" in F.columns else F.columns[0])
    roll = F[mcol].rolling(window).std()
    z = (roll - roll.mean()) / (roll.std() + 1e-12)
    stress = (roll > roll.median()).fillna(False).values
    out = {}
    for name, mask in (("calm", ~stress), ("stress", stress)):
        sub = F.values[mask]
        if len(sub) < 40:
            sub = F.values
        w = obs_weights(len(sub), "exponential", halflife=min(halflife, len(sub)))
        out[name] = weighted_cov(sub, w) * TRADING_DAYS
    z_now = float(z.dropna().iloc[-1]) if z.notna().any() else 0.0
    p_stress = float(1.0 / (1.0 + math.exp(-1.5 * z_now)))
    cov = p_stress * out["stress"] + (1 - p_stress) * out["calm"]
    info = {"p_stress": p_stress, "market_vol_z": z_now, "n_calm": int((~stress).sum()), "n_stress": int(stress.sum()),
            "stress_vol_multiple": float(np.sqrt(np.trace(out["stress"]) / max(np.trace(out["calm"]), 1e-18)))}
    return pd.DataFrame(cov, index=F.columns, columns=F.columns), info
