"""Levered-beta construction, margin policy, tactical overlay sizing, signals and the daily simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.synth import make_market
from tlh.optim import leverage as lv
from tlh.optim import tactical as tc
from tlh.optim.strategies import StrategyInputs, StrategySpec, run_strategy
from tlh.risk.model import FactorRiskModel, RiskModelSpec


def _market_with_levered_etfs():
    prices, sec, fund = make_market(n_stocks=80, n_days=700, seed=21, n_etfs=1)
    # synthetic 2x and 3x funds on the cap-weighted ETF (daily rebalanced, small fee)
    r = prices["ETFALL"].pct_change().fillna(0.0)
    for sym, k in (("SSO", 2.0), ("UPRO", 3.0), ("SDS", -2.0)):
        prices[sym] = 100 * np.cumprod(1 + k * r - lv.INSTRUMENTS[sym].expense_ratio / 252)
        sec.loc[sym] = {"assetid": 9500 + int(abs(k) * 10) + (1 if k < 0 else 0), "shares_outstanding": 1.4e7, "gics_sector": None, "gics_industry": None,
                        "gics_sub_industry": None, "subtype1": "Exchange Traded Product"}
    return prices, sec, fund


@pytest.fixture(scope="module")
def world():
    prices, sec, fund = _market_with_levered_etfs()
    model = FactorRiskModel(RiskModelSpec(model_kind="barra_lite", lookback_days=400)).fit(prices, sec, fund)
    stocks = [s for s in model.symbols if s.startswith("S")]
    cov = model.covariance(model.symbols)
    shares = sec.loc[stocks, "shares_outstanding"].astype(float)
    mc = shares * prices[stocks].iloc[-1]
    bench = mc / mc.sum()
    inp = StrategyInputs(symbols=stocks, cov=cov, benchmark=bench, returns=prices.pct_change().iloc[-300:], sectors=sec["gics_sector"].reindex(stocks), mktcap=mc)
    return model, inp, stocks, prices


def test_margin_policy_and_report():
    pol = lv.MarginPolicy()
    assert [round(pol.maintenance(s), 6) for s in ("AAPL", "SSO", "UPRO", "SPXU")] == [0.3, 0.6, 0.9, 0.9]
    w = pd.Series({"AAPL": 0.8, "SSO": 0.4})           # 120% long on 20% loan
    rep = lv.margin_report(w, 0.2, pol)
    assert rep["initial_margin_ok"] and abs(rep["maintenance_requirement"] - (0.24 + 0.24)) < 1e-9
    assert 0 < rep["market_drop_to_margin_call"] < 1
    assert lv.drag_cost(3.0, 0.16, 0.009) > lv.drag_cost(2.0, 0.16, 0.009) > 0.009


def test_levered_beta_hits_target_with_margin_constraints(world):
    _, inp, _, _ = world
    r = run_strategy(StrategySpec(kind="levered_beta", target_beta=1.5, replicate=False, n_max=40, max_weight=0.08, sector_band=None,
                                  lev_instruments=("SSO", "UPRO"), margin_max=0.5), inp)
    d = r.diagnostics
    assert d["levered"] and abs(d["beta"] - 1.5) < 0.02
    assert abs(r.weights.sum() - (1 + d["loan"])) < 1e-6 and d["loan"] <= 0.5 + 1e-9
    assert d["margin"]["initial_margin_ok"] and d["margin"]["buffer_ok"]
    assert d["etf_weights"] and all(v <= 0.35 + 1e-6 for v in d["etf_weights"].values())
    assert d["n_stocks"] <= 40 and d["te_vs_levered_benchmark"] < 0.08
    # cash-only variant must reach 1.5 with ETFs alone
    r2 = run_strategy(StrategySpec(kind="levered_beta", target_beta=1.5, replicate=False, n_max=40, max_weight=0.08, sector_band=None, margin_max=0.0), inp)
    assert abs(r2.diagnostics["loan"]) < 1e-6 and abs(r2.diagnostics["beta"] - 1.5) < 0.02 and abs(r2.weights.sum() - 1) < 1e-6


def test_levered_beta_full_replication_has_near_zero_tracking_error(world):
    """Default build: every index name at 1.5 x its weight on margin -> model TE essentially zero, beta exactly on target,
    and the cash-only book (2x/3x funds carry the leverage) stays inside the funds' measured tracking noise."""
    _, inp, stocks, _ = world
    r = run_strategy(StrategySpec(kind="levered_beta", target_beta=1.5), inp)
    d = r.diagnostics
    assert d["replication"] == "full" and d["n_stocks"] == len([s for s in stocks if s not in lv.INSTRUMENTS])
    assert abs(d["beta"] - 1.5) < 1e-3
    assert d["te_vs_levered_benchmark"] < 0.002          # < 0.2% a year against 1.5 x the index
    assert d["stock_cap_used"] >= float(inp.benchmark.max()) * 1.5 and d["margin"]["buffer_ok"] and d["margin"]["initial_margin_ok"]
    assert d["beta_source"] == "nominal leverage x index basket"
    assert d["realised"]["available"] and d["realised"]["te_daily"] < 0.01 and abs(d["realised"]["beta_daily"] - 1.5) < 0.05
    cash = run_strategy(StrategySpec(kind="levered_beta", target_beta=1.5, margin_max=0.0), inp).diagnostics
    assert abs(cash["beta"] - 1.5) < 1e-3 and cash["loan"] < 1e-9 and cash["te_vs_levered_benchmark"] < 0.01
    assert cash["etf_weights"] and sum(cash["etf_weights"].values()) > 0.2
    # pure tracking beats the cost-weighted book on TE, never on beta
    pure = run_strategy(StrategySpec(kind="levered_beta", target_beta=1.5, cost_weight=0.0), inp).diagnostics
    assert pure["te_vs_levered_benchmark"] <= d["te_vs_levered_benchmark"] + 1e-6 and abs(pure["beta"] - 1.5) < 1e-3


def test_levered_beta_ignores_leveraged_funds_in_the_benchmark(world):
    """A saved levered basket set as the house benchmark must not make the model track itself."""
    _, inp, stocks, _ = world
    b0 = inp.benchmark[[s for s in inp.benchmark.index if s not in lv.INSTRUMENTS]]
    bad = pd.concat([b0 / b0.sum() * 0.7, pd.Series({"SSO": 0.1, "UPRO": 0.2})])
    inp2 = StrategyInputs(**{**inp.__dict__, "benchmark": bad})
    r = run_strategy(StrategySpec(kind="levered_beta", target_beta=1.5), inp2)
    clean = run_strategy(StrategySpec(kind="levered_beta", target_beta=1.5), inp)
    assert abs(r.diagnostics["beta"] - 1.5) < 1e-3
    real = [s for s in stocks if s not in lv.INSTRUMENTS]
    assert np.allclose(r.weights.reindex(real).fillna(0).values, clean.weights.reindex(real).fillna(0).values, atol=2e-3)


def test_leveraged_covariance_and_costs():
    idx = ["A", "B", "SSO", "UPRO"]
    base = np.array([[0.04, 0.01, 0, 0], [0.01, 0.09, 0, 0], [0, 0, 0.5, 0.1], [0, 0, 0.1, 0.9]])
    cov = pd.DataFrame(base, index=idx, columns=idx)
    wb = pd.Series({"A": 0.5, "B": 0.5})
    C = lv.leveraged_covariance(cov, wb, ["SSO", "UPRO"], tracking_var={"SSO": 1e-4})
    var_b = float(wb.values @ cov.loc[wb.index, wb.index].values @ wb.values)       # 0.0375
    assert abs(C.loc["SSO", "A"] - 2 * (0.5 * 0.04 + 0.5 * 0.01)) < 1e-12           # k x cov(A, index)
    assert abs(C.loc["SSO", "SSO"] - (4 * var_b + 1e-4)) < 1e-12
    assert abs(C.loc["UPRO", "UPRO"] - (9 * var_b + 0.005 ** 2)) < 1e-12
    assert abs(C.loc["SSO", "UPRO"] - 6 * var_b) < 1e-12 and abs(C.loc["UPRO", "SSO"] - 6 * var_b) < 1e-12
    b = lv.nominal_betas(pd.Series({"A": 1.1, "SSO": 1.7, "UPRO": 2.4}), None, ["SSO", "UPRO"])
    assert b["SSO"] == 2.0 and b["UPRO"] == 3.0 and b["A"] == 1.1
    # cost per $ = fee + embedded financing (k-1)(rf + spread) + drag (k^2-k)/2 sigma^2
    c2 = lv.drag_cost(2.0, 0.16, 0.0089, rf=0.04)
    assert abs(c2 - (0.0089 + 1 * (0.04 + lv.SWAP_SPREAD) + 0.5 * 2 * 0.16 ** 2)) < 1e-12
    c3 = lv.drag_cost(3.0, 0.16, 0.0091, rf=0.04)
    assert abs(c3 - (0.0091 + 2 * (0.04 + lv.SWAP_SPREAD) + 0.5 * 6 * 0.16 ** 2)) < 1e-12
    assert c3 / 3 > c2 / 2                                                       # 3x is dearer per unit of beta
    assert lv.drag_cost(-2.0, 0.16, 0.009) > lv.drag_cost(-1.0, 0.16, 0.009) > 0.009


def test_realised_tracking_structure(world):
    """Stock sleeve = index, 2x fund at exactly the weight that makes 1.5 beta on a synthetic fund with zero fee -> the
    daily-rebalanced structure tracks 1.5 x the index almost perfectly; monthly drift adds a little."""
    _, inp, _, prices = world
    R = prices.pct_change().iloc[-300:]
    r = R["ETFALL"]
    R = R.assign(PURE2X=2 * r)                                                   # ideal 2x fund
    lv.INSTRUMENTS["PURE2X"] = lv.Instrument("PURE2X", 2.0, "SPX", 0.0)
    try:
        w = pd.Series({"ETFALL": 0.5, "PURE2X": 0.5})
        out = lv.realised_tracking(R, w, pd.Series({"ETFALL": 1.0}), 1.5, loan=0.0, proxy="ETFALL")
    finally:
        del lv.INSTRUMENTS["PURE2X"]
    assert out["available"] and out["index_leg"] == "ETFALL"
    assert out["te_structure_daily"] < 1e-9 and abs(out["beta_structure_daily"] - 1.5) < 1e-9
    assert out["te_structure_periodic"] >= 0 and abs(out["beta_structure_periodic"] - 1.5) < 0.05
    assert out["te_daily"] < 1e-9                                                # full book identical here


def test_tactical_overlay_sizing_up_and_down():
    pol = lv.MarginPolicy()
    up = lv.tactical_overlay(1_000_000, 1.0, 1.5, pol, cash=0.0)
    t = up["tickets"][0]
    assert t["side"] == "BUY" and t["symbol"] in ("SSO", "UPRO") and abs(up["beta_after"] - 1.5) < 1e-6
    assert up["chosen"]["loan"] <= pol.max_loan + 1e-9 and up["chosen"]["feasible"]
    down = lv.tactical_overlay(1_000_000, 1.0, 0.0, pol, cash=0.0, core_gain_frac=0.4, lt_rate=0.238)
    t2 = down["tickets"][0]
    assert t2["symbol"] in ("SH", "SDS", "SPXU") and down["tax_avoided_vs_selling_core"] > 0
    # beta 0 from beta 1 with -3x needs 33% notional -> loan 0.33 <= 0.5: feasible
    assert down["chosen"]["feasible"] and abs(down["beta_after"]) < 1e-6
    tight = lv.tactical_overlay(1_000_000, 1.0, 0.0, lv.MarginPolicy(max_loan=0.1), cash=0.0)
    assert "margin cap" in tight["note"] and tight["beta_after"] > 0.0
    same = lv.tactical_overlay(1_000_000, 1.0, 1.0, pol)
    assert same["tickets"] == []


def test_signals_and_simulator(world, tmp_path):
    _, _, _, prices = world
    idx = prices["ETFALL"]
    trend = tc.build_signal(tc.SignalSpec(name="trend", kind="rule:trend", beta_max=1.5), index_prices=idx)
    assert set(np.round(trend.unique(), 6)) <= {0.0, 1.5}
    vol = tc.build_signal(tc.SignalSpec(name="vol", kind="rule:vol_regime", beta_max=1.5), index_prices=idx)
    assert vol.between(0, 1.5).all()
    csv = tmp_path / "potomac.csv"
    csv.write_text("Date,State\n2023-01-03,risk_on\n2023-06-01,risk_off\n2024-01-02,risk_on\n", encoding="utf-8")
    s = tc.build_signal(tc.SignalSpec(name="p1", kind="csv", path=str(csv), beta_max=1.5), dates=idx.index)
    assert s.max() == 1.5 and s.min() == 0.0 and s.notna().sum() > 100
    lib = {"trend": trend, "p1": s}
    blend = tc.build_signal(tc.SignalSpec(name="b", kind="blend", components=[{"name": "trend", "weight": 2}, {"name": "p1", "weight": 1}], beta_max=1.5), library=lib)
    assert blend.between(0, 1.5).all()
    st = tc.signal_stats(trend)
    assert st["n_days"] > 0 and st["latest"] in (0.0, 1.5)
    res = lv.simulate_tactical(idx.pct_change().dropna(), trend, policy=lv.MarginPolicy())
    m = res["metrics"]
    assert np.isfinite(m["cagr"]) and m["n_rebalances"] >= 1 and 0 <= m["realised_beta"] <= 2.0
    assert len(res["equity"]) == len(idx) - 1
    store = tc.SignalStore(tmp_path / "sig")
    store.save("trend", trend)
    back = store.load("trend")
    assert back is not None and len(back) == len(trend)
