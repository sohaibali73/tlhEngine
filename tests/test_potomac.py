"""Potomac strategy signals from fund NAVs: flat NAV = risk-off, slow beta for risk-on days, 80/5/5/5/5 blend, one-day trading lag."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tlh.optim import potomac as pm
from tlh.optim.tactical import SignalSpec, build_signal, signal_stats


def _synthetic_navs(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    ri = rng.normal(0.0004, 0.01, n)
    spy = 400 * np.cumprod(1 + ri)
    # CRDBX: fully invested (beta 1) except days 100-160 in cash (flat NAV)
    r_a = ri.copy()
    r_a[100:160] = 0.0
    # CRMVX: low beta 0.1 always invested
    r_b = 0.1 * ri + rng.normal(0, 0.002, n)
    # CRTOX: beta 1.2, in cash days 300-330
    r_c = 1.2 * ri + rng.normal(0, 0.002, n)
    r_c[300:330] = 0.0
    # CRTBX and CRTPX: beta ~1 always on
    r_d = ri + rng.normal(0, 0.002, n)
    r_e = ri + rng.normal(0, 0.001, n)
    navs = pd.DataFrame({"CRDBX": 100 * np.cumprod(1 + r_a), "CRMVX": 100 * np.cumprod(1 + r_b), "CRTOX": 100 * np.cumprod(1 + r_c),
                         "CRTBX": 100 * np.cumprod(1 + r_d), "CRTPX": 100 * np.cumprod(1 + r_e), "SPY": spy}, index=dates)
    for f in list(pm.FUNDS):
        navs[f + pm.NAV_SUFFIX] = navs[f].round(2)            # the published NAV to the cent, as fetch_navs provides
    return navs


def test_holdings_and_core():
    for s, w in pm.STRATEGIES.items():
        assert abs(sum(w.values()) - 1.0) < 1e-12 and set(w) == set(pm.FUNDS) and max(w.values()) == 0.80
        assert pm.core_fund(s) in w and w[pm.core_fund(s)] == 0.80
    assert pm.core_fund("Navigrowth") == "CRTOX" and pm.core_fund("Income Plus") == "CRMVX"
    h = pm.holdings_table("Guardian")
    assert len(h) == 5 and h.loc[h["role"] == "core", "fund"].iloc[0] == "CRTBX"


def test_flat_nav_means_risk_off_and_slow_beta_sizes_risk_on():
    navs = _synthetic_navs()
    expo = pm.fund_exposure(navs.drop(columns=["SPY"]), navs["SPY"], window=60)
    a = expo["CRDBX"]
    cash = navs.index[100:160]                                # the NAV printed flat on these dates (returns are indexed by date)
    assert (a.loc[cash] == 0).all()                           # cash period reads as risk-off
    assert a.loc[navs.index[200]:].between(0.8, 1.25).all()   # invested again at about beta 1
    mv = expo["CRMVX"].loc[navs.index[200]:]                  # low-beta fund is "on" but worth ~0.1 of market exposure;
    assert 0.03 < mv.median() < 0.3 and (mv == 0).mean() < 0.05  # a rare quiet-day flat print may still read as a cash day
    c = expo["CRTOX"]
    assert (c.loc[navs.index[300:330]] == 0).all() and c.loc[navs.index[250]:navs.index[290]].between(0.9, 1.25).all()
    # monthly re-read: within a calendar month the risk-on value is constant
    on = a.loc[navs.index[200]:]
    per_month = on.groupby(on.index.to_period("M")).nunique()
    assert (per_month <= 1).all()


def test_confirm_days_delays_the_risk_off_call():
    navs = _synthetic_navs()
    e1 = pm.fund_exposure(navs.drop(columns=["SPY"]), navs["SPY"], confirm_days=1)["CRDBX"]
    e3 = pm.fund_exposure(navs.drop(columns=["SPY"]), navs["SPY"], confirm_days=3)["CRDBX"]
    d = navs.index
    assert e1.loc[d[100]] == 0 and e3.loc[d[100]] > 0 and e3.loc[d[101]] > 0 and e3.loc[d[102]] == 0


def test_strategy_signal_blends_and_lags_one_day():
    navs = _synthetic_navs()
    beta, info = pm.strategy_signal("Bull Bear", beta_min=0.0, beta_max=1.5, lag_days=1, navs=navs)
    beta0, _ = pm.strategy_signal("Bull Bear", beta_min=0.0, beta_max=1.5, lag_days=0, navs=navs)
    # lagged series equals the unlagged one shifted by a day: the signal is traded on the next close
    common = beta.index.intersection(beta0.index)[5:]
    assert np.allclose(beta.loc[common].values, beta0.shift(1).loc[common].values)
    # in CRDBX's cash period, Bull Bear keeps only the four 5% satellites -> exposure about 0.05 * (0.1 + 1.2 + 1 + 1) ~ 0.17
    mid = beta0.loc[navs.index[110]:navs.index[150]]
    assert mid.between(0.05, 0.45).all()
    late = beta0.loc[navs.index[350]:]
    assert late.between(1.3, 1.5).all()
    assert info["core"] == "CRDBX" and info["lag_days"] == 1 and set(info["funds_used"]) == set(pm.FUNDS)
    assert 0 < info["pct_days_risk_off_core"] < 0.3


def test_income_plus_is_low_exposure_by_construction():
    navs = _synthetic_navs()
    beta, info = pm.strategy_signal("Income Plus", beta_max=1.5, lag_days=0, navs=navs)
    assert beta.loc[navs.index[200]:].max() < 0.7 and info["core"] == "CRMVX"


def test_build_signal_potomac_kind_and_missing_fund_renormalises():
    navs = _synthetic_navs().drop(columns=["CRTPX", "CRTPX" + pm.NAV_SUFFIX])   # a share class without history yet
    spec = SignalSpec(name="bb", kind="potomac", strategy="Bull Bear", beta_max=1.5)
    s = build_signal(spec, navs=navs)
    assert s.name == "bb" and s.between(0, 1.5).all() and len(s) > 300
    st = signal_stats(s)
    assert st["changes"] >= 2 and st["latest"] > 1.0
    # rules and csv are lagged too (trading at the next close): a manual signal is not
    idx = navs["SPY"]
    r1 = build_signal(SignalSpec(name="t", kind="rule:trend", lag_days=1), index_prices=idx)
    r0 = build_signal(SignalSpec(name="t", kind="rule:trend", lag_days=0), index_prices=idx)
    common = r1.index.intersection(r0.index)[2:]
    assert np.allclose(r1.loc[common].values, r0.shift(1).loc[common].values)
