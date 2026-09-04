"""Potomac tactical strategies as target-beta signals, read off the affiliated funds' NAVs.

Each Potomac strategy holds the same five tactical mutual funds at target weights (80% core, 4 x 5%). The funds are
either risk-on (the NAV moves with the market) or risk-off (in cash: the NAV prints flat, or drifts by a money-market
accrual). So the strategy's *exposure* on any day can be read from the NAVs themselves:

    exposure_f(t) = 0                                if the fund's NAV was flat on day t while the market moved (risk-off)
                  = clip(slow_beta_f, 0, 1.25)       otherwise: the fund's beta over its last 60 risk-on days, re-read monthly
    exposure_S(t) = sum_f  w_{S,f} * exposure_f(t)

The strategy's signal is generated on the prior close and traded on the next close, so the target beta the overlay
uses on day t is the exposure observed at t-1 (`lag_days = 1`). Data come from Yahoo Finance (yfinance) because the
funds are not in Norgate; pulls are cached under var/tactical/navs for `cache_hours`.

Target allocations are those on the fact sheets and are subject to change with market conditions.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FUNDS: dict[str, str] = {
    "CRDBX": "Potomac Defensive Bull",
    "CRTPX": "Potomac Tactically Passive",
    "CRTBX": "Potomac Tactical Rotation",
    "CRMVX": "Potomac Managed Volatility",
    "CRTOX": "Potomac Tactical Opportunities",
}
INDEX_PROXY = "SPY"
NAV_SUFFIX = "__nav"          # column suffix for the published (unadjusted) NAV kept next to the adjusted series
NAV_CENT = 0.005              # a NAV unchanged to the cent is a flat print

STRATEGIES: dict[str, dict[str, float]] = {
    "Bull Bear": {"CRDBX": 0.80, "CRTPX": 0.05, "CRTBX": 0.05, "CRMVX": 0.05, "CRTOX": 0.05},
    "Focused Growth": {"CRTPX": 0.80, "CRDBX": 0.05, "CRTBX": 0.05, "CRMVX": 0.05, "CRTOX": 0.05},
    "Guardian": {"CRTBX": 0.80, "CRTOX": 0.05, "CRMVX": 0.05, "CRDBX": 0.05, "CRTPX": 0.05},
    "Income Plus": {"CRMVX": 0.80, "CRTOX": 0.05, "CRTBX": 0.05, "CRDBX": 0.05, "CRTPX": 0.05},
    "Navigrowth": {"CRTOX": 0.80, "CRDBX": 0.05, "CRMVX": 0.05, "CRTBX": 0.05, "CRTPX": 0.05},
}
STRATEGY_OBJECTIVE = {
    "Bull Bear": "core CRDBX (Defensive Bull): equity exposure with a defensive risk-off switch",
    "Focused Growth": "core CRTPX (Tactically Passive): passive-like growth exposure with tactical de-risking",
    "Guardian": "core CRTBX (Tactical Rotation): rotation across asset classes, capital preservation first",
    "Income Plus": "core CRMVX (Managed Volatility): absolute return / low volatility",
    "Navigrowth": "core CRTOX (Tactical Opportunities): growth exposure through tactical opportunities",
}


def core_fund(strategy: str) -> str:
    w = STRATEGIES[strategy]
    return max(w, key=w.get)


def holdings_table(strategy: str | None = None) -> pd.DataFrame:
    rows = []
    for s, w in STRATEGIES.items():
        if strategy and s != strategy:
            continue
        for f, a in w.items():
            rows.append({"strategy": s, "fund": f, "name": FUNDS.get(f, f), "allocation": a, "role": "core" if a >= 0.5 else "satellite"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- data
def fetch_navs(tickers: list[str] | None = None, start: str = "2019-01-01", cache_dir: Path | None = None, cache_hours: float = 12.0) -> pd.DataFrame:
    """Adjusted closes (NAV adjusted for distributions) for the funds and the index proxy, from Yahoo Finance."""
    tickers = list(tickers or list(FUNDS) + [INDEX_PROXY])
    if INDEX_PROXY not in tickers:
        tickers.append(INDEX_PROXY)
    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"navs_{'_'.join(sorted(tickers))}_{start}.parquet"
        if cache.exists() and (time.time() - cache.stat().st_mtime) < cache_hours * 3600:
            return pd.read_parquet(cache)
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("yfinance is not installed: pip install yfinance") from e
    df = yf.download(tickers, start=start, auto_adjust=False, progress=False, group_by="column", threads=True)
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned no data for the Potomac funds")
    px = (df["Adj Close"] if "Adj Close" in df.columns.get_level_values(0) else df["Close"]).copy()
    raw = df["Close"].copy() if "Close" in df.columns.get_level_values(0) else None
    px.index = pd.to_datetime(px.index).tz_localize(None) if getattr(px.index, "tz", None) is not None else pd.to_datetime(px.index)
    if raw is not None:
        # published NAVs (to the cent) travel alongside the adjusted series: "flat" means the NAV did not change to the cent
        raw.index = px.index
        for c in raw.columns:
            if c != INDEX_PROXY:
                px[f"{c}{NAV_SUFFIX}"] = raw[c]
    px = px.sort_index().dropna(how="all")
    if cache is not None:
        px.to_parquet(cache)
    return px


# ---------------------------------------------------------------------- exposure from NAVs
def fund_exposure(navs: pd.DataFrame, index: pd.Series, window: int = 60, flat_tol: float = 1e-4, max_beta: float = 1.25, confirm_days: int = 1) -> pd.DataFrame:
    """Daily exposure per fund in [0, max_beta] = state x slow beta.

    state: 0 (risk-off) on a day the NAV printed flat (unchanged to the cent) although the fund's exposure times the
    index move would have moved it by more than two cents (the fund is in cash), else 1 (risk-on). A flat print on a
    day too quiet to be informative carries the previous state (this matters for low-beta funds such as Managed
    Volatility, whose small moves often round to an unchanged NAV).
    slow beta: the fund's beta to the index over its last `window` risk-on days (min 20), clipped to [0, max_beta]; this is
    what a risk-on day is worth in market exposure (about 1 for the equity funds, about 0.1 for Managed Volatility) and it
    moves slowly, so the signal switches only when the fund actually goes to or from cash. `confirm_days` > 1 requires that
    many consecutive flat days before calling the fund risk-off (smooths stale-NAV prints at the cost of a later exit)."""
    idx_r = index.pct_change()
    out = {}
    for f in navs.columns:
        if f == index.name or f.endswith(NAV_SUFFIX):
            continue
        r = navs[f].pct_change()
        both = pd.concat([r, idx_r], axis=1, keys=["f", "i"]).dropna()
        if both.empty:
            continue
        raw = navs.get(f + NAV_SUFFIX)
        if raw is not None:
            flat = (raw.diff().abs() < NAV_CENT).reindex(both.index).fillna(False)   # published NAV unchanged to the cent
            resolution = (0.01 / raw.reindex(both.index).ffill().clip(lower=0.01))    # one cent as a return
        else:
            flat = both["f"].abs() < flat_tol
            resolution = pd.Series(flat_tol, index=both.index)
        # slow beta from risk-on days: what a risk-on day is worth in market exposure
        active = both[~flat]
        mp = max(20, window // 3)
        cov = (active["f"] * active["i"]).rolling(window, min_periods=mp).mean() - active["f"].rolling(window, min_periods=mp).mean() * active["i"].rolling(window, min_periods=mp).mean()
        var = active["i"].rolling(window, min_periods=mp).var(ddof=0)
        beta = (cov / var.replace(0, np.nan)).clip(0, max_beta)
        slow = pd.Series(np.nan, index=both.index)
        slow[active.index] = beta.values
        slow = slow.ffill().bfill().fillna(1.0)
        # re-read once a month (the estimate at the first day of the month, no look-ahead)
        slow = slow.groupby(slow.index.to_period("M")).transform("first").round(2)
        # a flat print is evidence of cash only if the fund *would* have moved by more than the NAV resolution that day:
        # expected move = yesterday's exposure x |index return| >= 2 x one cent. Otherwise the day is uninformative.
        informative = (slow.shift(1).fillna(1.0) * both["i"].abs()) >= 2.0 * resolution
        off_day = flat & informative
        if confirm_days > 1:
            run = off_day.astype(int).groupby((~off_day).cumsum()).cumsum()     # consecutive risk-off prints
            off_day = run >= confirm_days
        state = pd.Series(np.nan, index=both.index)
        state[~flat] = 1.0
        state[off_day] = 0.0
        state = state.ffill().fillna(1.0)
        out[f] = state * slow
    return pd.DataFrame(out)


def strategy_exposure(exposures: pd.DataFrame, strategy: str) -> pd.Series:
    w = STRATEGIES[strategy]
    cols = [f for f in w if f in exposures.columns]
    if not cols:
        raise ValueError(f"no fund history for {strategy}")
    ww = pd.Series({f: w[f] for f in cols})
    ww = ww / ww.sum()                                # a fund without history yet (e.g. a new share class) is dropped and weights renormalised
    e = exposures[cols].ffill()
    return (e * ww).sum(axis=1, min_count=1).rename(strategy)


def strategy_signal(strategy: str, beta_min: float = 0.0, beta_max: float = 1.5, lag_days: int = 1, navs: pd.DataFrame | None = None,
                    start: str = "2019-01-01", cache_dir: Path | None = None, window: int = 60, confirm_days: int = 1) -> tuple[pd.Series, dict]:
    """Target beta series for the overlay: exposure in [0, 1] mapped onto [beta_min, beta_max], shifted by `lag_days`
    (the signal is known at the prior close and traded at the next close)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown Potomac strategy {strategy}; choose from {list(STRATEGIES)}")
    navs = fetch_navs(list(STRATEGIES[strategy]) + [INDEX_PROXY], start=start, cache_dir=cache_dir) if navs is None else navs
    if INDEX_PROXY not in navs.columns:
        raise ValueError(f"{INDEX_PROXY} history needed alongside the fund NAVs")
    expo = fund_exposure(navs.drop(columns=[INDEX_PROXY]), navs[INDEX_PROXY].rename(INDEX_PROXY), window=window, confirm_days=confirm_days)
    expo = expo[[c for c in expo.columns if c in STRATEGIES[strategy]]]
    s_exp = strategy_exposure(expo, strategy).clip(0, 1)
    beta = beta_min + s_exp * (beta_max - beta_min)
    beta = beta.shift(lag_days).dropna() if lag_days else beta
    beta.name = strategy
    latest = expo.iloc[-1] if len(expo) else pd.Series(dtype=float)
    info = {"strategy": strategy, "core": core_fund(strategy), "objective": STRATEGY_OBJECTIVE.get(strategy, ""), "lag_days": lag_days, "confirm_days": confirm_days,
            "funds_used": list(expo.columns), "latest_fund_exposure": {k: round(float(v), 3) for k, v in latest.items()},
            "latest_exposure": float(s_exp.iloc[-1]) if len(s_exp) else None, "latest_target_beta": float(beta.iloc[-1]) if len(beta) else None,
            "as_of": str(expo.index[-1].date()) if len(expo) else None, "tradable_on": "next close after the signal date",
            "pct_days_risk_off_core": float((expo[core_fund(strategy)] <= 1e-9).mean()) if core_fund(strategy) in expo else None,
            "source": "Yahoo Finance adjusted NAVs; exposure inferred from flat-NAV days and rolling beta"}
    return beta, info


def fund_state_table(navs: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Latest state per fund for display."""
    expo = fund_exposure(navs.drop(columns=[c for c in navs.columns if c == INDEX_PROXY]), navs[INDEX_PROXY].rename(INDEX_PROXY), window=window)
    rows = []
    for f in expo.columns:
        r = navs[f].pct_change().dropna()
        raw = navs.get(f + NAV_SUFFIX)
        nav_now = float(raw.dropna().iloc[-1]) if raw is not None else float(navs[f].dropna().iloc[-1])
        flat_share = float((raw.diff().abs().iloc[-252:] < NAV_CENT).mean()) if raw is not None else (float((r.iloc[-252:].abs() < 1e-4).mean()) if len(r) else np.nan)
        rows.append({"fund": f, "name": FUNDS.get(f, f), "nav": nav_now, "as_of": str(navs[f].dropna().index[-1].date()),
                     "last_return": float(r.iloc[-1]) if len(r) else np.nan, "exposure": float(expo[f].iloc[-1]),
                     "state": "risk-off" if expo[f].iloc[-1] <= 1e-9 else "risk-on", "pct_days_flat_1y": flat_share})
    return pd.DataFrame(rows)
