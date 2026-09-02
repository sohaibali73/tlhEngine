"""Risk analytics on a fitted model: decomposition, marginal contributions, factor stress tests, historical scenario
replay, parametric VaR/ES, and out-of-sample bias tests of the risk forecasts.

This module is AI-editable (ai/registry.py). Everything here is pure given a FittedRiskModel and weights.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from .model import FittedRiskModel

log = logging.getLogger(__name__)


def _group(f: str) -> str:
    if f == "market":
        return "market"
    if f.startswith(("sec:", "ind:")):
        return "industry"
    if f.startswith("macro:"):
        return "macro"
    return "style"


def _aligned(model: FittedRiskModel, weights: pd.Series, benchmark: pd.Series | None) -> tuple[pd.Series, list[str]]:
    syms = [s for s in weights.index.union(benchmark.index if benchmark is not None else weights.index) if s in model.symbols]
    w = weights.reindex(syms).fillna(0.0)
    w = w / w.sum() if w.sum() else w
    if benchmark is not None:
        b = benchmark.reindex(syms).fillna(0.0)
        b = b / b.sum() if b.sum() else b
        w = w - b
    return w, syms


# ====================================================================================== decomposition
def risk_decomposition(model: FittedRiskModel, weights: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """Total (or active, if benchmark given) risk split by factor group, factor and holding."""
    a, syms = _aligned(model, weights, benchmark)
    X = model.exposures.loc[syms]
    F = model.factor_cov.values
    D = model.specific_var.loc[syms].values
    x = X.T @ a                                   # portfolio factor exposures
    fvar = float(x.values @ F @ x.values)
    svar = float((a.values ** 2 * D).sum())
    total = fvar + svar
    sigma = np.sqrt(max(total, 0.0))
    contrib = pd.Series(x.values * (F @ x.values), index=model.factors)        # sums to fvar
    groups = contrib.groupby(contrib.index.map(_group)).sum()
    groups["specific"] = svar
    # marginal contribution to risk per holding (dσ/dw_i) and % contribution
    Sigma_w = X.values @ (F @ x.values) + D * a.values
    mctr = Sigma_w / sigma if sigma > 0 else np.zeros(len(syms))
    ctr = a.values * mctr
    hold = pd.DataFrame({"weight": a.values, "mctr": mctr, "ctr": ctr, "pct_of_risk": ctr / sigma if sigma > 0 else 0.0}, index=syms)
    hold = hold[hold["weight"].abs() > 1e-9].sort_values("ctr", ascending=False)
    return {
        "sigma": sigma, "factor_var": fvar, "specific_var": svar, "pct_factor": fvar / total if total else 0.0,
        "groups_var": groups.to_dict(), "groups_pct": (groups / total).to_dict() if total else {},
        "factor_contrib_var": contrib.sort_values(ascending=False).to_dict(),
        "factor_contrib_sigma": (contrib / sigma).to_dict() if sigma else {},
        "exposures": x.to_dict(), "holdings": hold, "is_active": benchmark is not None,
    }


# ====================================================================================== stress tests
PRESET_SHOCKS: dict[str, dict[str, float]] = {
    "Market -2σ": {"market": -2.0},
    "Market -10% (raw)": {"market:raw": -0.10},
    "Momentum crash": {"momentum": -3.0, "market": -1.0},
    "Value rally / growth sell-off": {"value": 2.0, "growth": -2.0},
    "Flight to quality": {"quality": 2.0, "lowvol": 1.5, "resvol": -2.0, "beta": -2.0, "market": -1.5},
    "Small-cap squeeze": {"size": -2.5},
    "Rates +100bp": {"macro:rate_10y": 2.0, "value": 1.0, "growth": -1.0},
    "Liquidity shock": {"liquidity": -2.0, "market": -1.0, "resvol": 1.5},
}


def stress_test(model: FittedRiskModel, weights: pd.Series, shocks: dict[str, float], benchmark: pd.Series | None = None,
                propagate: bool = True) -> dict:
    """Factor shock P&L. Shock keys are factor names in sigma units (annualised factor vol), or '<factor>:raw' for a raw
    return shock. With `propagate`, unshocked factors move by their conditional expectation given the shocked ones."""
    a, syms = _aligned(model, weights, benchmark)
    X = model.exposures.loc[syms]
    F = model.factor_cov
    vols = model.factor_vols()
    factors = list(F.columns)
    shocked: dict[str, float] = {}
    ignored: list[str] = []
    for k, v in shocks.items():
        raw = k.endswith(":raw")
        f = k[:-4] if raw else k
        if f not in factors:
            ignored.append(k)
            continue
        shocked[f] = float(v) if raw else float(v) * float(vols[f])
    if not shocked:
        return {"portfolio_return": 0.0, "factor_moves": {}, "ignored": ignored, "holdings": pd.DataFrame()}
    df = pd.Series(0.0, index=factors)
    for f, v in shocked.items():
        df[f] = v
    if propagate and len(shocked) < len(factors):
        sidx = [factors.index(f) for f in shocked]
        oidx = [i for i in range(len(factors)) if i not in sidx]
        Fv = F.values
        Sss = Fv[np.ix_(sidx, sidx)] + 1e-12 * np.eye(len(sidx))
        Sos = Fv[np.ix_(oidx, sidx)]
        cond = Sos @ np.linalg.solve(Sss, df.values[sidx])
        for i, o in enumerate(oidx):
            df.iloc[o] = cond[i]
    x = X.T @ a
    port = float(x.values @ df.values)
    contrib = pd.Series(x.values * df.values, index=factors)
    hold_ret = X.values @ df.values
    hold = pd.DataFrame({"weight": a.values, "shock_return": hold_ret, "contribution": a.values * hold_ret}, index=syms)
    hold = hold[hold["weight"].abs() > 1e-9].sort_values("contribution")
    return {"portfolio_return": port, "factor_moves": df.round(6).to_dict(), "factor_contrib": contrib.sort_values().round(6).to_dict(),
            "shocked": shocked, "propagated": propagate, "ignored": ignored, "holdings": hold, "is_active": benchmark is not None}


def historical_scenario(model: FittedRiskModel, weights: pd.Series, start: str | date, end: str | date,
                        benchmark: pd.Series | None = None) -> dict:
    """Replay the model's own factor returns over a window through today's exposures (specific risk ignored)."""
    a, syms = _aligned(model, weights, benchmark)
    fr = model.factor_returns.loc[pd.Timestamp(start): pd.Timestamp(end)]
    if fr.empty:
        return {"error": "window outside the fitted factor-return history", "available": [str(model.factor_returns.index.min().date()), str(model.factor_returns.index.max().date())]}
    cum = (1 + fr).prod() - 1
    x = model.exposures.loc[syms].T @ a
    common = [f for f in cum.index if f in x.index]
    port = float(x[common].values @ cum[common].values)
    return {"portfolio_return": port, "start": str(fr.index.min().date()), "end": str(fr.index.max().date()), "n_days": int(len(fr)),
            "factor_cum_returns": cum[common].round(5).to_dict(), "factor_contrib": (x[common] * cum[common]).sort_values().round(5).to_dict()}


# ====================================================================================== VaR / ES
def parametric_var(model: FittedRiskModel, weights: pd.Series, horizon_days: int = 21, alpha: float = 0.99,
                   benchmark: pd.Series | None = None) -> dict:
    from scipy.stats import norm
    a, _ = _aligned(model, weights, benchmark)
    sigma_ann = model.portfolio_risk(a)
    sigma_h = sigma_ann * np.sqrt(horizon_days / 252)
    z = norm.ppf(alpha)
    return {"sigma_annual": sigma_ann, "sigma_horizon": sigma_h, "var": z * sigma_h, "es": sigma_h * norm.pdf(z) / (1 - alpha),
            "horizon_days": horizon_days, "alpha": alpha}


# ====================================================================================== bias tests
def bias_test(prices: pd.DataFrame, securities: pd.DataFrame, fundamentals: pd.DataFrame, spec, n_periods: int = 6,
              period_days: int = 21, volume: pd.DataFrame | None = None, macro=None, holdings: pd.Series | None = None,
              progress=None) -> pd.DataFrame:
    """Out-of-sample bias statistics: refit the model at the start of each period, predict portfolio volatility for
    test portfolios, compare with realised returns. Bias stat b = std(realised / predicted); ideal 1, band ~ 1 ± sqrt(2/n)."""
    from .model import FactorRiskModel

    say = progress or (lambda m: None)
    prices = prices.sort_index()
    idx = prices.index
    rows = []
    zs: dict[str, list[float]] = {}
    ends = [idx[-1 - i * period_days] for i in range(n_periods)][::-1]
    fast = type(spec)(**{**spec.__dict__})
    for k in ("eigen_adjust",):
        if hasattr(fast, k):
            setattr(fast, k, False)
    for end in ends:
        start_i = idx.get_loc(end) - period_days
        if start_i < spec.lookback_days // 2:
            continue
        fit_to = idx[start_i]
        say(f"bias test: fitting as of {fit_to.date()}…")
        try:
            m = FactorRiskModel(fast).fit(prices.loc[:fit_to], securities, fundamentals, volume=volume, macro_levels=macro)
        except Exception as e:
            log.warning("bias test fit %s failed: %s", fit_to.date(), e)
            continue
        stocks = [s for s in m.symbols if s in securities.index and not str(securities.loc[s].get("subtype1", "")).lower().startswith("exchange")]
        shares = pd.to_numeric(securities.get("shares_outstanding"), errors="coerce").reindex(stocks)
        mc = (shares * prices.loc[fit_to, stocks]).fillna(0.0)
        ports = {"cap-weighted market": mc / mc.sum() if mc.sum() else None,
                 "equal-weight": pd.Series(1.0 / len(stocks), index=stocks)}
        if holdings is not None:
            h = holdings[holdings.index.isin(m.symbols)]
            if h.sum() > 0:
                ports["holdings"] = h / h.sum()
        for st in m.spec.styles:
            if st in m.exposures.columns:
                z = m.exposures.loc[stocks, st].fillna(0.0)
                ports[f"style: {st}"] = (z.clip(lower=0) / z.clip(lower=0).sum() - z.clip(upper=0).abs() / z.clip(upper=0).abs().sum()) if z.abs().sum() else None
        window = prices.loc[fit_to:end].pct_change().iloc[1:]
        for name, w in ports.items():
            if w is None or w.empty:
                continue
            pred = m.portfolio_risk(w) * np.sqrt(len(window) / 252)
            real = float((window.reindex(columns=w.index).fillna(0.0).values @ w.values).sum())
            if pred > 0:
                zs.setdefault(name, []).append(real / pred)
                rows.append({"period_end": str(end.date()), "portfolio": name, "predicted_vol": pred, "realised_return": real, "z": real / pred})
    detail = pd.DataFrame(rows)
    summary = []
    for name, z in zs.items():
        n = len(z)
        b = float(np.std(z, ddof=1)) if n > 1 else float("nan")
        summary.append({"portfolio": name, "n": n, "bias_stat": b, "band_low": 1 - np.sqrt(2 / n) if n else np.nan,
                        "band_high": 1 + np.sqrt(2 / n) if n else np.nan, "mean_z": float(np.mean(z)),
                        "verdict": "ok" if n > 1 and (1 - np.sqrt(2 / n)) <= b <= (1 + np.sqrt(2 / n)) else ("under-forecast" if b > 1 else "over-forecast")})
    out = pd.DataFrame(summary)
    out.attrs["detail"] = detail
    return out
