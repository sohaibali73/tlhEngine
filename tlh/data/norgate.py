"""Thin, testable wrapper around `norgatedata`.

All Norgate access goes through `NorgateClient` so the rest of the app never imports norgatedata directly
(tests substitute a fake). Bulk pulls are parallelised with threads; norgatedata is documented thread-safe.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FUNDAMENTAL_FIELDS = [
    "mktcap", "qbvps", "peexclxor", "ttmpr2rev", "ttmnpmgn", "ttmgrosmgn", "qtotd2eq", "ttmepschg",
    "revchngyr", "ttmfcf", "ttmniac", "sharesoutstanding", "beta", "ttmrevps", "projepsq",
]

MACRO_SYMBOLS = {
    "rate_10y": "%TNX",       # US 10-Year T. Note Yield
    "rate_3m": "%IRX",        # US 13-Week T. Bill Yield
    "slope_10y2y": "#US10Y-2Y",
    "baa": "%COBAA",          # Moody's BAA yield
    "aaa": "%COAAA",          # Moody's AAA yield
    "usd": "$USDX",           # US Dollar Index
}


class NorgateUnavailable(RuntimeError):
    pass


@dataclass
class SecurityMeta:
    assetid: int
    symbol: str
    name: str | None
    subtype1: str | None
    subtype2: str | None
    subtype3: str | None
    gics_sector: str | None
    gics_industry_group: str | None
    gics_industry: str | None
    gics_sub_industry: str | None
    gics_code: str | None
    first_quoted: str | None
    last_quoted: str | None
    shares_outstanding: float | None
    shares_float: float | None

    @property
    def is_etp(self) -> bool:
        return (self.subtype1 or "").lower().startswith("exchange traded")


class NorgateClient:
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._nd: Any = None

    # ------------------------------------------------------------------ availability
    @property
    def nd(self):
        if self._nd is None:
            try:
                import norgatedata  # noqa: WPS433 (lazy import by design)
            except ImportError as e:  # pragma: no cover
                raise NorgateUnavailable("norgatedata package not installed") from e
            self._nd = norgatedata
        return self._nd

    def status(self) -> bool:
        try:
            return bool(self.nd.status())
        except Exception as e:  # pragma: no cover
            log.warning("Norgate status check failed: %s", e)
            return False

    def require(self) -> None:
        if not self.status():
            raise NorgateUnavailable(
                "Norgate Data Updater (NDU) is not running. Start NDU and retry; the engine cannot pull "
                "prices or reference data without it."
            )

    # ------------------------------------------------------------------ lists
    def watchlist_symbols(self, name: str) -> list[str]:
        return list(self.nd.watchlist_symbols(name))

    def watchlists(self) -> list[str]:
        return list(self.nd.watchlists())

    def database_symbols(self, name: str) -> list[str]:
        return list(self.nd.database_symbols(name))

    # ------------------------------------------------------------------ identity
    def assetid(self, symbol: str) -> int | None:
        try:
            return int(self.nd.assetid(symbol))
        except Exception:
            return None

    def symbol(self, assetid: int) -> str | None:
        try:
            return str(self.nd.symbol(int(assetid)))
        except Exception:
            return None

    def _safe(self, fn, *args, default=None):
        try:
            v = fn(*args)
            return default if v is None else v
        except Exception:
            return default

    def security_meta(self, symbol: str) -> SecurityMeta | None:
        nd = self.nd
        aid = self.assetid(symbol)
        if aid is None:
            return None
        so = self._safe(nd.sharesoutstanding, symbol, default=(None, None))
        sf = self._safe(nd.sharesfloat, symbol, default=(None, None))
        return SecurityMeta(
            assetid=aid, symbol=symbol,
            name=self._safe(nd.security_name, symbol),
            subtype1=self._safe(nd.subtype1, symbol), subtype2=self._safe(nd.subtype2, symbol),
            subtype3=self._safe(nd.subtype3, symbol),
            gics_sector=self._safe(nd.classification_at_level, symbol, "GICS", "Name", 1),
            gics_industry_group=self._safe(nd.classification_at_level, symbol, "GICS", "Name", 2),
            gics_industry=self._safe(nd.classification_at_level, symbol, "GICS", "Name", 3),
            gics_sub_industry=self._safe(nd.classification_at_level, symbol, "GICS", "Name", 4),
            gics_code=self._safe(nd.classification, symbol, "GICS", "ClassificationId"),
            first_quoted=self._safe(nd.first_quoted_date, symbol, "iso"),
            last_quoted=self._safe(nd.last_quoted_date, symbol, "iso"),
            shares_outstanding=_f(so[0]) if so else None,
            shares_float=_f(sf[0]) if sf else None,
        )

    def securities_meta(self, symbols: list[str]) -> pd.DataFrame:
        with ThreadPoolExecutor(self.max_workers) as ex:
            metas = [m for m in ex.map(self.security_meta, symbols) if m is not None]
        return pd.DataFrame([m.__dict__ for m in metas])

    # ------------------------------------------------------------------ fundamentals
    def fundamentals(self, symbol: str, fields: list[str] | None = None) -> dict[str, Any]:
        fields = fields or FUNDAMENTAL_FIELDS
        out: dict[str, Any] = {"symbol": symbol}
        for f in fields:
            try:
                val, asof = self.nd.fundamental(symbol, f, datetimeformat="iso")
            except Exception:
                val, asof = None, None
            out[f] = _f(val)
            out[f + "_asof"] = asof
        return out

    def fundamentals_table(self, symbols: list[str], fields: list[str] | None = None) -> pd.DataFrame:
        with ThreadPoolExecutor(self.max_workers) as ex:
            rows = list(ex.map(lambda s: self.fundamentals(s, fields), symbols))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ prices
    def price_history(self, symbol: str, start: str | date | None, end: str | date | None = None,
                      adjustment: str = "TOTALRETURN") -> pd.DataFrame:
        """Long-format daily bars for one symbol: date, symbol, assetid, open, high, low, close, volume,
        unadj_close, dividend."""
        nd = self.nd
        adj = getattr(nd.StockPriceAdjustmentType, adjustment)
        try:
            df = nd.price_timeseries(
                symbol, stock_price_adjustment_setting=adj, padding_setting=nd.PaddingType.NONE,
                start_date=str(start) if start else None, end_date=str(end) if end else None,
                timeseriesformat="pandas-dataframe",
            )
        except Exception as e:
            log.warning("price pull failed for %s: %s", symbol, e)
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
            "Unadjusted Close": "unadj_close", "Dividend": "dividend", "Turnover": "turnover",
        })
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["symbol"] = symbol
        df["assetid"] = self.assetid(symbol)
        keep = [c for c in ["date", "symbol", "assetid", "open", "high", "low", "close", "volume",
                            "unadj_close", "dividend"] if c in df.columns]
        out = df[keep].copy()
        for c in ("open", "high", "low", "close", "volume", "unadj_close", "dividend"):
            if c in out:
                out[c] = out[c].astype("float64")
        return out

    def price_panel(self, symbols: list[str], start: str | date | None, end: str | date | None = None,
                    adjustment: str = "TOTALRETURN") -> pd.DataFrame:
        with ThreadPoolExecutor(self.max_workers) as ex:
            frames = list(ex.map(lambda s: self.price_history(s, start, end, adjustment), symbols))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["date", "symbol", "assetid", "close"])
        return pd.concat(frames, ignore_index=True)

    def last_close(self, symbol: str) -> tuple[date, float] | None:
        nd = self.nd
        try:
            df = nd.price_timeseries(symbol, stock_price_adjustment_setting=nd.StockPriceAdjustmentType.NONE,
                                     padding_setting=nd.PaddingType.NONE, limit=1,
                                     timeseriesformat="pandas-dataframe")
        except Exception:
            return None
        if df is None or df.empty:
            return None
        return pd.Timestamp(df.index[-1]).date(), float(df["Close"].iloc[-1])

    # ------------------------------------------------------------------ index membership
    def index_constituents(self, symbols: list[str], indexname: str, start: str | date | None) -> pd.DataFrame:
        nd = self.nd

        def one(sym: str) -> pd.DataFrame:
            try:
                df = nd.index_constituent_timeseries(sym, indexname, start_date=str(start) if start else None,
                                                     padding_setting=nd.PaddingType.NONE,
                                                     timeseriesformat="pandas-dataframe")
            except Exception:
                return pd.DataFrame()
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"Index Constituent": "member"})
            df.index.name = "date"
            df = df.reset_index()
            df["symbol"] = sym
            return df[["date", "symbol", "member"]]

        with ThreadPoolExecutor(self.max_workers) as ex:
            frames = [f for f in ex.map(one, symbols) if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", "member"])

    # ------------------------------------------------------------------ macro / economic
    def economic_series(self, symbol: str, start: str | date | None) -> pd.Series:
        nd = self.nd
        try:
            df = nd.price_timeseries(symbol, padding_setting=nd.PaddingType.NONE,
                                     start_date=str(start) if start else None,
                                     timeseriesformat="pandas-dataframe")
        except Exception as e:
            log.warning("economic pull failed for %s: %s", symbol, e)
            return pd.Series(dtype="float64", name=symbol)
        if df is None or df.empty:
            return pd.Series(dtype="float64", name=symbol)
        s = df["Close"].astype("float64")
        s.index = pd.to_datetime(s.index).normalize()
        s.name = symbol
        return s

    def macro_panel(self, start: str | date | None, symbols: dict[str, str] | None = None) -> pd.DataFrame:
        symbols = symbols or MACRO_SYMBOLS
        cols = {k: self.economic_series(v, start) for k, v in symbols.items()}
        df = pd.DataFrame(cols).sort_index()
        if {"baa", "aaa"} <= set(df.columns):
            df["credit_spread"] = df["baa"] - df["aaa"]
        return df


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
        return None if np.isnan(fv) else fv
    except (TypeError, ValueError):
        return None
