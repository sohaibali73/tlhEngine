"""Macro factor inputs: rates level, curve slope, credit spread, dollar.

Default source is Norgate's Economic database (already local). If FRED_API_KEY is set, `fred_panel` offers
the same columns from FRED so the two can be cross-checked. Output is a daily DataFrame indexed by date with
columns: rate_10y, rate_3m, slope_10y2y, credit_spread, usd. `macro_shocks` converts levels to the daily
changes the risk model regresses on.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import get_settings
from .norgate import NorgateClient

log = logging.getLogger(__name__)

FRED_SERIES = {
    "rate_10y": "DGS10",
    "rate_3m": "DGS3MO",
    "slope_10y2y": "T10Y2Y",
    "baa": "DBAA",
    "aaa": "DAAA",
    "usd": "DTWEXBGS",
}

MACRO_COLUMNS = ["rate_10y", "slope_10y2y", "credit_spread", "usd"]


def norgate_panel(start: str, client: NorgateClient | None = None) -> pd.DataFrame:
    client = client or NorgateClient()
    return client.macro_panel(start)


def fred_panel(start: str) -> pd.DataFrame:
    key = get_settings().fred_api_key
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    from fredapi import Fred

    fred = Fred(api_key=key)
    cols = {k: fred.get_series(v, observation_start=start) for k, v in FRED_SERIES.items()}
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    df["credit_spread"] = df["baa"] - df["aaa"]
    return df


def macro_shocks(levels: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Daily shocks: first differences for yields/spreads (in percentage points), log returns for the dollar."""
    columns = columns or MACRO_COLUMNS
    lv = levels.sort_index().ffill()
    out = pd.DataFrame(index=lv.index)
    for c in columns:
        if c not in lv:
            continue
        s = pd.to_numeric(lv[c], errors="coerce").astype("float64")
        out[c] = np.log(s / s.shift(1)) if c == "usd" else s.diff()
    return out.dropna(how="all")
