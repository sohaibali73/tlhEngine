"""Factor definitions for the equity risk model.

This module is AI-editable (see ai/registry.py): the co-pilot may add a style factor by adding an entry to
STYLE_DEFINITIONS and a function that returns a raw cross-sectional score. Everything downstream (standardising,
regression, covariance) is generic.

Each raw-score function receives a `FactorInputs` bundle and returns a pd.Series indexed by symbol (NaN where
the factor is undefined for that name, e.g. ETFs). Higher = more of the factor.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FactorInputs:
    """Everything a raw-score function may need, positioned at one date `t`."""
    prices: pd.DataFrame            # date x symbol total-return close, up to and including t
    t: pd.Timestamp
    fundamentals: pd.DataFrame      # indexed by symbol; Norgate fields (current values)
    securities: pd.DataFrame        # indexed by symbol; shares_outstanding, gics_*, subtype1..
    price_at_asof: pd.Series | None = None   # unadjusted close on the fundamentals as-of date (optional)

    @property
    def px_t(self) -> pd.Series:
        return self.prices.loc[: self.t].iloc[-1]

    def col(self, name: str) -> pd.Series:
        if name in self.fundamentals:
            return pd.to_numeric(self.fundamentals[name], errors="coerce").reindex(self.prices.columns)
        return pd.Series(np.nan, index=self.prices.columns)

    def shares(self) -> pd.Series:
        s = pd.to_numeric(self.securities.get("shares_outstanding"), errors="coerce") if "shares_outstanding" in self.securities else None
        if s is None:
            s = self.col("sharesoutstanding")
        return s.reindex(self.prices.columns)

    def mktcap_t(self) -> pd.Series:
        """Market cap at t = shares outstanding (current) x price at t. Falls back to reported mktcap."""
        mc = self.shares() * self.px_t
        rep = self.col("mktcap") * 1e6   # Norgate mktcap is in $ millions
        return mc.where(mc.notna() & (mc > 0), rep)


# ----------------------------------------------------------------------------------- raw scores
def momentum(fi: FactorInputs) -> pd.Series:
    """12-1 month price momentum: return from t-252 to t-21 trading days."""
    px = fi.prices.loc[: fi.t]
    if len(px) < 260:
        return pd.Series(np.nan, index=px.columns)
    return px.iloc[-22] / px.iloc[-253] - 1.0


def lowvol(fi: FactorInputs) -> pd.Series:
    """Negative of trailing 252-day daily-return volatility (higher score = lower vol)."""
    px = fi.prices.loc[: fi.t].iloc[-253:]
    r = px.pct_change().iloc[1:]
    return -r.std()


def size(fi: FactorInputs) -> pd.Series:
    """Log market capitalisation."""
    mc = fi.mktcap_t()
    return np.log(mc.where(mc > 0))


def value(fi: FactorInputs) -> pd.Series:
    """Composite of book/price, earnings/price and sales/price, each scaled by price at t."""
    px = fi.px_t
    bp = fi.col("qbvps") / px
    # Norgate gives P/E at an as-of date; recover E = P_asof / PE using the price on that date if we have it.
    pe = fi.col("peexclxor")
    p_asof = fi.price_at_asof if fi.price_at_asof is not None else px
    eps = p_asof.reindex(px.index) / pe.where(pe > 0)
    ep = eps / px
    sp = fi.col("ttmrevps") / px
    return _composite([bp, ep, sp])


def quality(fi: FactorInputs) -> pd.Series:
    """Composite of net margin, gross margin, ROE proxy and (negative) leverage."""
    npm = fi.col("ttmnpmgn")
    gpm = fi.col("ttmgrosmgn")
    lev = -fi.col("qtotd2eq")
    book = fi.col("qbvps") * fi.shares()
    roe = (fi.col("ttmniac") * 1e6) / book.where(book > 0)   # ttmniac in $ millions
    return _composite([npm, gpm, roe, lev])


def growth(fi: FactorInputs) -> pd.Series:
    """Composite of trailing EPS growth and revenue growth."""
    return _composite([fi.col("ttmepschg"), fi.col("revchngyr")])


def _composite(parts: list[pd.Series]) -> pd.Series:
    zs = []
    for p in parts:
        p = p.replace([np.inf, -np.inf], np.nan)
        if p.notna().sum() < 10:
            continue
        z = (p - p.mean()) / p.std(ddof=0)
        zs.append(z.clip(-3, 3))
    if not zs:
        return pd.Series(np.nan, index=parts[0].index)
    return pd.concat(zs, axis=1).mean(axis=1, skipna=True)


@dataclass
class StyleDefinition:
    name: str
    fn: Callable[[FactorInputs], pd.Series]
    description: str
    needs_fundamentals: bool = False


STYLE_DEFINITIONS: dict[str, StyleDefinition] = {
    "value": StyleDefinition("value", value, "B/P, E/P, S/P composite", True),
    "momentum": StyleDefinition("momentum", momentum, "12-1 month return"),
    "quality": StyleDefinition("quality", quality, "margins, ROE, low leverage", True),
    "size": StyleDefinition("size", size, "log market cap"),
    "lowvol": StyleDefinition("lowvol", lowvol, "negative 1y daily vol"),
    "growth": StyleDefinition("growth", growth, "EPS and revenue growth", True),
}


# ----------------------------------------------------------------------------------- standardisation
def standardize(raw: pd.Series, cap_weights: pd.Series | None = None, winsor: float = 3.0) -> pd.Series:
    """Barra-style: cap-weighted mean zero, equal-weighted unit std, winsorised. NaN preserved."""
    x = raw.replace([np.inf, -np.inf], np.nan)
    ok = x.notna()
    if ok.sum() < 5:
        return pd.Series(np.nan, index=raw.index)
    if cap_weights is not None:
        w = cap_weights.reindex(x.index).where(ok).fillna(0.0)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / ok.sum(), index=x.index).where(ok, 0.0)
        mu = float((x.fillna(0.0) * w).sum())
    else:
        mu = float(x[ok].mean())
    sd = float(x[ok].std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=raw.index)
    z = (x - mu) / sd
    return z.clip(-winsor, winsor)


@dataclass
class ExposureBuild:
    exposures: pd.DataFrame           # symbol x factor (market, styles, sectors)
    style_cols: list[str]
    sector_cols: list[str]
    cap_weights: pd.Series
    missing_style: pd.Index = field(default_factory=lambda: pd.Index([]))


def build_exposures(fi: FactorInputs, styles: list[str], use_sectors: bool = True,
                    sector_field: str = "gics_sector") -> ExposureBuild:
    """Exposure matrix at date fi.t for every symbol in fi.prices.columns.

    Rows lacking fundamentals/sector (ETFs) are left NaN; the model later fills them by time-series regression.
    """
    symbols = fi.prices.columns
    mc = fi.mktcap_t().reindex(symbols)
    capw = mc.where(mc > 0).fillna(0.0)
    # Exchange-traded products carry placeholder share counts in Norgate; they are not part of the
    # cross-section anyway (no fundamentals / GICS) so give them zero weight in standardisation.
    if "subtype1" in fi.securities:
        is_etp = fi.securities["subtype1"].reindex(symbols).fillna("").str.lower().str.startswith("exchange traded")
        capw = capw.where(~is_etp, 0.0)
    X = pd.DataFrame(index=symbols)
    X["market"] = 1.0
    style_cols = []
    etp_mask = (capw <= 0) & fi.securities.get("subtype1", pd.Series(index=symbols, dtype=object)).reindex(symbols).fillna("").str.lower().str.startswith("exchange traded")
    for s in styles:
        d = STYLE_DEFINITIONS[s]
        raw = d.fn(fi).reindex(symbols)
        raw = raw.where(~etp_mask)          # ETPs are not part of the stock cross-section
        X[s] = standardize(raw, capw)
        style_cols.append(s)
    sector_cols: list[str] = []
    if use_sectors and sector_field in fi.securities:
        sec = fi.securities[sector_field].reindex(symbols)
        for name in sorted(sec.dropna().unique()):
            col = f"sec:{name}"
            X[col] = (sec == name).astype(float).where(sec.notna())
            sector_cols.append(col)
    missing = X[style_cols].isna().all(axis=1)
    return ExposureBuild(exposures=X, style_cols=style_cols, sector_cols=sector_cols, cap_weights=capw,
                         missing_style=X.index[missing])
