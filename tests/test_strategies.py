import numpy as np
import pandas as pd
import pytest

from tlh.optim.backtest import BacktestSpec, run_backtest
from tlh.optim.strategies import STRATEGIES, StrategyInputs, StrategySpec, run_strategy, shrunk_sample_cov
from tlh.risk.factors import FactorInputs, build_exposures

from .synth import make_market


@pytest.fixture(scope="module")
def world():
    prices, sec, fund = make_market(n_stocks=70, n_days=650, seed=11)
    stocks = [s for s in prices.columns if s.startswith("S")]
    R = prices[stocks].pct_change().iloc[-300:]
    cov = shrunk_sample_cov(R)
    shares = sec.loc[stocks, "shares_outstanding"].astype(float)
    mktcap = shares * prices[stocks].iloc[-1]
    bench = mktcap / mktcap.sum()
    fi = FactorInputs(prices=prices[stocks], t=prices.index[-1], fundamentals=fund.reindex(stocks), securities=sec.reindex(stocks))
    b = build_exposures(fi, ["momentum", "lowvol", "size", "value", "quality", "growth"])
    rng = np.random.default_rng(3)
    cur = pd.Series(rng.dirichlet(np.ones(15)), index=stocks[:15])
    gain = pd.Series(rng.normal(0.1, 0.3, 15), index=stocks[:15])          # unrealised gain per $ held
    inp = StrategyInputs(symbols=stocks, cov=cov, benchmark=bench, returns=R, signals=b.exposures[b.style_cols],
                         exposures=b.exposures, sectors=sec.loc[stocks, "gics_sector"], mktcap=mktcap,
                         current_weights=cur, gain_frac=gain)
    return prices, sec, fund, inp, stocks


def _valid(res, n_max, max_w):
    w = res.weights
    if res.diagnostics.get("levered"):
        d = res.diagnostics
        assert (w >= 0).all() and abs(w.sum() - (1 + d["loan"])) < 1e-6
        assert d["margin"]["initial_margin_ok"] and d["margin"]["buffer_ok"]
        assert abs(d["beta"] - d["target_beta"]) < 0.05
    elif res.diagnostics.get("long_short"):
        d = res.diagnostics
        assert abs(w.sum() - 1) < 1e-3                                   # net = 1, signed weights
        assert abs(d["long_exposure"] + d["short_exposure"] - 1) < 1e-3
        assert d["short_exposure"] < -0.05
        assert not (set(d["long_weights"]) & set(d["short_weights"]))
    else:
        assert abs(w.sum() - 1) < 1e-6
        assert (w >= 0).all()
        assert w.max() <= max_w + 1e-3
        if n_max:
            assert len(w) <= n_max
    assert np.isfinite(res.diagnostics["tracking_error"]) and np.isfinite(res.diagnostics["volatility"])


@pytest.mark.parametrize("kind", [k for k in STRATEGIES if k not in ("tax_aware_transition", "black_litterman")])
def test_every_strategy_produces_valid_weights(world, kind):
    _, _, _, inp, _ = world
    spec = StrategySpec(kind=kind, n_max=20, max_weight=0.15, sector_band=0.05, tilts={"lowvol": 0.3},
                        signal_weights={"momentum": 1.0, "quality": 0.5})
    res = run_strategy(spec, inp)
    _valid(res, 20, 0.15)
    assert res.kind == kind


def test_min_variance_has_lower_vol_than_equal_weight(world):
    _, _, _, inp, _ = world
    mv = run_strategy(StrategySpec(kind="min_variance", n_max=None, max_weight=0.2), inp)
    ew = run_strategy(StrategySpec(kind="equal_weight", n_max=None, max_weight=0.2), inp)
    assert mv.diagnostics["volatility"] < ew.diagnostics["volatility"]


def test_risk_parity_equalises_contributions(world):
    _, _, _, inp, _ = world
    res = run_strategy(StrategySpec(kind="risk_parity", n_max=12, max_weight=0.5), inp)
    assert res.diagnostics["risk_contrib_max"] / res.diagnostics["risk_contrib_min"] < 1.6


def test_mean_variance_tilts_toward_signal(world):
    _, _, _, inp, stocks = world
    spec = StrategySpec(kind="mean_variance", n_max=None, max_weight=0.1, signal_weights={"momentum": 1.0}, ic=0.1, risk_aversion=2.0)
    res = run_strategy(spec, inp)
    z = inp.signals["momentum"].reindex(res.weights.index).fillna(0.0)
    active = res.weights - inp.benchmark.reindex(res.weights.index).fillna(0.0)
    assert float((active * z).sum()) > 0.0                      # positive active exposure to the signal
    assert res.diagnostics["expected_alpha"] > 0


def test_black_litterman_view_moves_weight(world):
    _, _, _, inp, stocks = world
    base = run_strategy(StrategySpec(kind="black_litterman", n_max=None, max_weight=0.2, views=[]), inp)
    fav = stocks[3]
    viewed = run_strategy(StrategySpec(kind="black_litterman", n_max=None, max_weight=0.2,
                                       views=[{"assets": {fav: 1.0}, "return": 0.15, "confidence": 0.9}]), inp)
    assert viewed.weights.get(fav, 0.0) > base.weights.get(fav, 0.0)
    assert viewed.diagnostics["n_views"] == 1


def test_min_cvar_reports_tail_metrics(world):
    _, _, _, inp, _ = world
    res = run_strategy(StrategySpec(kind="min_cvar", n_max=25, max_weight=0.1, cvar_alpha=0.95), inp)
    assert res.diagnostics["daily_cvar"] >= res.diagnostics["daily_var"] > 0


def test_tax_aware_transition_respects_gain_budget(world):
    _, _, _, inp, stocks = world
    target = {s: 1.0 / 20 for s in stocks[10:30]}
    spec = StrategySpec(kind="tax_aware_transition", target_weights=target, gain_budget=0.0, turnover_max=1.0, max_weight=0.2)
    res = run_strategy(spec, inp)
    d = res.diagnostics
    assert d["realised_gain_frac"] <= 1e-4                        # net gains within budget (zero)
    assert d["te_to_target"] <= d["te_to_target_before"] + 1e-9
    loose = run_strategy(StrategySpec(kind="tax_aware_transition", target_weights=target, gain_budget=0.5, turnover_max=1.0, max_weight=0.2), inp)
    assert loose.diagnostics["te_to_target"] <= d["te_to_target"] + 1e-9   # more budget -> closer to target


def test_stratified_index_covers_sectors(world):
    _, sec, _, inp, stocks = world
    res = run_strategy(StrategySpec(kind="stratified_index", n_max=24, max_weight=0.1, size_buckets=2), inp)
    sectors_picked = set(sec.loc[res.weights.index, "gics_sector"])
    assert sectors_picked == set(sec.loc[stocks, "gics_sector"].unique())


def test_backtest_runs_and_reports(world):
    prices, sec, fund, inp, stocks = world
    spec = BacktestSpec(rebalance="M", lookback_days=200, cost_bps=5, benchmark_symbol="ETFALL", styles=["momentum", "lowvol", "size"])
    strat = StrategySpec(kind="min_variance", n_max=15, max_weight=0.15)
    res = run_backtest(prices, sec, fund, strat, spec)
    m = res.metrics
    assert len(res.equity) > 200 and len(res.weights) >= 10
    assert set(["cagr", "sharpe", "max_drawdown", "tracking_error", "information_ratio", "annual_turnover"]) <= set(m)
    assert abs(res.equity.iloc[-1] - (1 + m["total_return"])) < 1e-9
    assert (res.turnover.iloc[1:] <= 1.0 + 1e-9).all()
    assert any("survivorship" in w for w in res.warnings)
    mv = run_backtest(prices, sec, fund, StrategySpec(kind="mean_variance", n_max=15, signal_weights={"momentum": 1}), spec)
    assert np.isfinite(mv.metrics["information_ratio"])
