"""Descriptor library for the equity risk model (ERM).

Each descriptor is a raw cross-sectional score at date t built from prices, volume, shares and current fundamentals.
Styles are fixed-weight composites of standardized descriptors (Barra USE4-style). Everything degrades gracefully:
a descriptor whose inputs are missing returns NaN and is dropped from its composite.

This module is AI-editable (ai/registry.py): add a descriptor function, register it in DESCRIPTORS, and reference it
from STYLE_COMPOSITES (or a custom module under tlh/risk/custom/ that edits these dicts).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DescriptorInputs:
    prices: pd.DataFrame                      # date x symbol total-return close (up to and including t)
    t: pd.Timestamp
    fundamentals: pd.DataFrame                # indexed by symbol
    securities: pd.DataFrame                  # indexed by symbol (shares_outstanding, gics_*)
    volume: pd.DataFrame | None = None        # date x symbol shares traded
    market_ret: pd.Series | None = None       # cap-weighted market daily return

    @property
    def px(self) -> pd.DataFrame:
        return self.prices.loc[: self.t]

    @property
    def px_t(self) -> pd.Series:
        return self.px.iloc[-1]

    def col(self, name: str) -> pd.Series:
        if name in self.fundamentals:
            return pd.to_numeric(self.fundamentals[name], errors="coerce").reindex(self.prices.columns)
        return pd.Series(np.nan, index=self.prices.columns)

    def shares(self) -> pd.Series:
        s = pd.to_numeric(self.securities["shares_outstanding"], errors="coerce") if "shares_outstanding" in self.securities else pd.Series(dtype=float)
        s = s.reindex(self.prices.columns)
        return s.where(s > 0, self.col("sharesoutstanding"))

    def mktcap(self) -> pd.Series:
        mc = self.shares() * self.px_t
        rep = self.col("mktcap") * 1e6
        return mc.where(mc.notna() & (mc > 0), rep)

    def rets(self, n: int) -> pd.DataFrame:
        return self.px.iloc[-(n + 1):].pct_change().iloc[1:]


def _ew(n: int, halflife: int) -> np.ndarray:
    w = 0.5 ** (np.arange(n)[::-1] / max(halflife, 1))
    return w / w.sum()


# ----------------------------------------------------------------------------------- size
def lncap(d: DescriptorInputs) -> pd.Series:
    mc = d.mktcap()
    return np.log(mc.where(mc > 0))


# ----------------------------------------------------------------------------------- beta / residual vol (shared regression)
def _beta_regression(d: DescriptorInputs, n: int = 252, halflife: int = 63) -> tuple[pd.Series, pd.Series]:
    r = d.rets(n)
    if d.market_ret is None:
        mc = d.mktcap().reindex(r.columns).fillna(0.0)
        w = (mc / mc.sum()).values if mc.sum() > 0 else np.ones(len(r.columns)) / len(r.columns)
        m = pd.Series(np.nan_to_num(r.values) @ w, index=r.index)
    else:
        m = d.market_ret.reindex(r.index).fillna(0.0)
    wts = _ew(len(r), halflife)
    mx = m.values - np.average(m.values, weights=wts)
    var_m = float((wts * mx ** 2).sum())
    betas, hsig = {}, {}
    R = r.values
    for j, s in enumerate(r.columns):
        y = R[:, j]
        ok = ~np.isnan(y)
        if ok.sum() < int(0.6 * n):
            betas[s], hsig[s] = np.nan, np.nan
            continue
        ww = wts[ok] / wts[ok].sum()
        yc = y[ok] - np.average(y[ok], weights=ww)
        mxo = mx[ok]
        vm = float((ww * mxo ** 2).sum()) or var_m
        b = float((ww * yc * mxo).sum() / vm)
        resid = yc - b * mxo
        betas[s], hsig[s] = b, float(np.sqrt((ww * resid ** 2).sum()))
    return pd.Series(betas), pd.Series(hsig)


def beta(d: DescriptorInputs) -> pd.Series:
    return _beta_regression(d)[0]


def hsigma(d: DescriptorInputs) -> pd.Series:
    return _beta_regression(d)[1]


def dastd(d: DescriptorInputs) -> pd.Series:
    r = d.rets(252)
    w = _ew(len(r), 42)
    mu = np.nansum(r.values * w[:, None], axis=0)
    var = np.nansum(((r.values - mu) ** 2) * w[:, None], axis=0)
    ok = r.notna().sum() >= 120
    return pd.Series(np.sqrt(var), index=r.columns).where(ok)


def cmra(d: DescriptorInputs) -> pd.Series:
    """Cumulative range: max minus min of the cumulative 12-month log return path."""
    r = d.rets(252)
    lr = np.log1p(r.fillna(0.0)).cumsum()
    ok = r.notna().sum() >= 120
    return (lr.max() - lr.min()).where(ok)


# ----------------------------------------------------------------------------------- momentum
def rstr(d: DescriptorInputs) -> pd.Series:
    """Relative strength: EWMA-weighted (hl 126) sum of log returns from t-273 to t-21."""
    px = d.px
    if len(px) < 280:
        return pd.Series(np.nan, index=px.columns)
    r = np.log(px.iloc[-274:-21]).diff().iloc[1:]
    w = _ew(len(r), 126) * len(r)
    ok = r.notna().sum() >= int(0.7 * len(r))
    return (r.fillna(0.0) * w[:, None]).sum().where(ok)


def mom6(d: DescriptorInputs) -> pd.Series:
    px = d.px
    if len(px) < 130:
        return pd.Series(np.nan, index=px.columns)
    return px.iloc[-22] / px.iloc[-127] - 1.0


# ----------------------------------------------------------------------------------- value
def btop(d: DescriptorInputs) -> pd.Series:
    return d.col("qbvps") / d.px_t


def etop(d: DescriptorInputs) -> pd.Series:
    pe = d.col("peexclxor")
    return (1.0 / pe.where(pe > 0)).where(pe.notna())


def stop(d: DescriptorInputs) -> pd.Series:
    return d.col("ttmrevps") / d.px_t


def cftop(d: DescriptorInputs) -> pd.Series:
    mc = d.mktcap()
    return (d.col("ttmfcf") * 1e6) / mc.where(mc > 0)


# ----------------------------------------------------------------------------------- quality / growth / leverage
def roe(d: DescriptorInputs) -> pd.Series:
    book = d.col("qbvps") * d.shares()
    return (d.col("ttmniac") * 1e6) / book.where(book > 0)


def npm(d: DescriptorInputs) -> pd.Series:
    return d.col("ttmnpmgn")


def gpm(d: DescriptorInputs) -> pd.Series:
    return d.col("ttmgrosmgn")


def opm(d: DescriptorInputs) -> pd.Series:
    return d.col("ttmopmgn")


def lowlev(d: DescriptorInputs) -> pd.Series:
    return -d.col("qtotd2eq")


def dtoe(d: DescriptorInputs) -> pd.Series:
    return d.col("qtotd2eq")


def epsg(d: DescriptorInputs) -> pd.Series:
    return d.col("ttmepschg")


def revg(d: DescriptorInputs) -> pd.Series:
    return d.col("revchngyr")


# ----------------------------------------------------------------------------------- liquidity
def _turnover(d: DescriptorInputs, n: int) -> pd.Series | None:
    if d.volume is None:
        return None
    v = d.volume.loc[: d.t].iloc[-n:]
    sh = d.shares().reindex(v.columns)
    to = v.sum() / sh.where(sh > 0)
    return to.where(v.notna().sum() >= int(0.7 * n))


def stom(d: DescriptorInputs) -> pd.Series:
    to = _turnover(d, 21)
    return np.log(to.where(to > 0)) if to is not None else pd.Series(np.nan, index=d.prices.columns)


def stoq(d: DescriptorInputs) -> pd.Series:
    to = _turnover(d, 63)
    return np.log((to / 3).where(to > 0)) if to is not None else pd.Series(np.nan, index=d.prices.columns)


def stoa(d: DescriptorInputs) -> pd.Series:
    to = _turnover(d, 252)
    return np.log((to / 12).where(to > 0)) if to is not None else pd.Series(np.nan, index=d.prices.columns)


DESCRIPTORS: dict[str, Callable[[DescriptorInputs], pd.Series]] = {
    "lncap": lncap, "beta": beta, "hsigma": hsigma, "dastd": dastd, "cmra": cmra, "rstr": rstr, "mom6": mom6,
    "btop": btop, "etop": etop, "stop": stop, "cftop": cftop, "roe": roe, "npm": npm, "gpm": gpm, "opm": opm,
    "lowlev": lowlev, "dtoe": dtoe, "epsg": epsg, "revg": revg, "stom": stom, "stoq": stoq, "stoa": stoa,
}

# style -> {descriptor: weight}. `midcap` is derived (cube of size, orthogonalised) and handled in erm.py.
STYLE_COMPOSITES: dict[str, dict[str, float]] = {
    "size": {"lncap": 1.0},
    "beta": {"beta": 1.0},
    "momentum": {"rstr": 0.7, "mom6": 0.3},
    "resvol": {"dastd": 0.6, "cmra": 0.2, "hsigma": 0.2},
    "value": {"btop": 0.4, "etop": 0.3, "stop": 0.15, "cftop": 0.15},
    "quality": {"roe": 0.35, "npm": 0.25, "gpm": 0.2, "lowlev": 0.2},
    "growth": {"epsg": 0.5, "revg": 0.5},
    "liquidity": {"stom": 0.35, "stoq": 0.35, "stoa": 0.3},
    "leverage": {"dtoe": 1.0},
}
DERIVED_STYLES = {"midcap": "cube of standardized size, orthogonalised to size"}
ORTHOGONALISE = {"resvol": ["beta", "size"], "liquidity": ["size"], "midcap": ["size"]}

ERM_DEFAULT_STYLES = ["size", "midcap", "beta", "momentum", "resvol", "value", "quality", "growth", "liquidity", "leverage"]

STYLE_DESCRIPTIONS = {
    "size": "log market cap", "midcap": "non-linear size (cube of size, orthogonal)", "beta": "EWMA beta to cap-weighted market",
    "momentum": "12-1m relative strength + 6-1m", "resvol": "residual volatility: daily std, cumulative range, idiosyncratic sigma",
    "value": "B/P, E/P, S/P, FCF/P", "quality": "ROE, net & gross margin, low leverage", "growth": "EPS and revenue growth",
    "liquidity": "share turnover 1m/3m/12m", "leverage": "debt to equity",
}


def standardize(raw: pd.Series, capw: pd.Series, winsor: float = 3.0) -> pd.Series:
    x = raw.replace([np.inf, -np.inf], np.nan)
    ok = x.notna() & (capw.reindex(x.index).fillna(0) >= 0)
    if ok.sum() < 5:
        return pd.Series(np.nan, index=raw.index)
    w = capw.reindex(x.index).where(ok).fillna(0.0)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=x.index).where(ok, 0.0)
    w = w / w.sum()
    mu = float((x.fillna(0.0) * w).sum())
    sd = float(x[ok].std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=raw.index)
    z = ((x - mu) / sd).clip(-winsor, winsor)
    # re-centre after winsorising so the cap-weighted market has zero exposure
    z = z - float((z.fillna(0.0) * w).sum())
    return z


def orthogonalise(y: pd.Series, xs: list[pd.Series], capw: pd.Series) -> pd.Series:
    """Cap-weighted regression residual of y on xs (with intercept)."""
    df = pd.concat([y] + xs, axis=1).dropna()
    if len(df) < 10:
        return y
    w = capw.reindex(df.index).fillna(0.0).values
    w = w / w.sum() if w.sum() > 0 else np.ones(len(df)) / len(df)
    X = np.column_stack([np.ones(len(df))] + [df.iloc[:, i + 1].values for i in range(len(xs))])
    Wx = X * np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(Wx, df.iloc[:, 0].values * np.sqrt(w), rcond=None)
    resid = pd.Series(df.iloc[:, 0].values - X @ b, index=df.index)
    return resid.reindex(y.index)
