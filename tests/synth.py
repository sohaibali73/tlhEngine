"""Synthetic factor market for risk-model and optimizer tests (importable helper, not a test module)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["Tech", "Health", "Financials", "Energy"]


def make_market(n_stocks: int = 120, n_days: int = 700, seed: int = 0, n_etfs: int = 2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    syms = [f"S{i:03d}" for i in range(n_stocks)]
    sectors = rng.choice(SECTORS, size=n_stocks)
    beta = rng.normal(1.0, 0.25, n_stocks)
    mom_exp = rng.normal(0, 1, n_stocks)
    vol_exp = rng.normal(0, 1, n_stocks)
    # factor returns
    f_mkt = rng.normal(0.0004, 0.010, n_days)
    f_mom = rng.normal(0.0, 0.004, n_days)
    f_vol = rng.normal(0.0, 0.003, n_days)
    f_sec = {s: rng.normal(0.0, 0.005, n_days) for s in SECTORS}
    spec = rng.normal(0.0, 0.012, (n_days, n_stocks))
    R = np.zeros((n_days, n_stocks))
    for i in range(n_stocks):
        R[:, i] = beta[i] * f_mkt + 0.3 * mom_exp[i] * f_mom + 0.3 * vol_exp[i] * f_vol + f_sec[sectors[i]] + spec[:, i]
    px = 100 * np.cumprod(1 + R, axis=0)
    prices = pd.DataFrame(px, index=dates, columns=syms)
    shares = rng.lognormal(20, 0.8, n_stocks)
    securities = pd.DataFrame({
        "symbol": syms, "assetid": np.arange(1000, 1000 + n_stocks), "shares_outstanding": shares,
        "gics_sector": sectors, "gics_industry": [f"{s}-ind{i % 3}" for i, s in enumerate(sectors)],
        "gics_sub_industry": [f"{s}-sub{i % 5}" for i, s in enumerate(sectors)], "subtype1": "Equity",
    }).set_index("symbol")
    fundamentals = pd.DataFrame({
        "symbol": syms, "qbvps": rng.uniform(5, 60, n_stocks), "peexclxor": rng.uniform(8, 40, n_stocks),
        "ttmrevps": rng.uniform(10, 120, n_stocks), "ttmnpmgn": rng.uniform(2, 30, n_stocks),
        "ttmgrosmgn": rng.uniform(20, 70, n_stocks), "qtotd2eq": rng.uniform(0, 150, n_stocks),
        "ttmniac": rng.uniform(100, 5000, n_stocks), "ttmepschg": rng.normal(8, 15, n_stocks),
        "revchngyr": rng.normal(6, 10, n_stocks), "mktcap": shares * px[-1] / 1e6,
    }).set_index("symbol")
    # ETFs: cap-weighted basket of all stocks, and a tech-only basket
    w = shares * px[0]
    w = w / w.sum()
    etf_all = (R * w).sum(axis=1)
    tech = sectors == "Tech"
    wt = w * tech
    wt = wt / wt.sum()
    etf_tech = (R * wt).sum(axis=1)
    etfs = {"ETFALL": etf_all, "ETFTEC": etf_tech}
    for k, r in list(etfs.items())[:n_etfs]:
        prices[k] = 100 * np.cumprod(1 + r)
        securities.loc[k] = {"assetid": 9000 + len(securities), "shares_outstanding": 1.4e7, "gics_sector": None,
                             "gics_industry": None, "gics_sub_industry": None, "subtype1": "Exchange Traded Product"}
    return prices, securities, fundamentals
