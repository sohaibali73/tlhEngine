"""Benchmark weight construction.

Two flavours: cap-weighted index from constituents (shares outstanding x last price) or a single ETF ticker.
Both return a pd.Series of weights summing to 1, indexed by symbol.
"""
from __future__ import annotations

import pandas as pd


def cap_weighted(securities: pd.DataFrame, last_prices: pd.Series, members: list[str]) -> pd.Series:
    sec = securities.set_index("symbol") if "symbol" in securities.columns else securities
    shares = pd.to_numeric(sec["shares_outstanding"], errors="coerce").reindex(members)
    px = last_prices.reindex(members)
    mc = (shares * px).dropna()
    mc = mc[mc > 0]
    if mc.empty:
        raise ValueError("no market caps available for benchmark members")
    return (mc / mc.sum()).sort_values(ascending=False)


def single_etf(symbol: str) -> pd.Series:
    return pd.Series({symbol: 1.0})


def weights_from_values(values: pd.Series) -> pd.Series:
    v = values[values > 0]
    return v / v.sum() if v.sum() > 0 else v
