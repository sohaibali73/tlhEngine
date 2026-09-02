import numpy as np
import pytest

from tlh.optim.basket import BasketSpec, analyze_basket, build_basket
from tlh.risk.model import FactorRiskModel, RiskModelSpec

from .synth import make_market


@pytest.fixture(scope="module")
def world():
    prices, sec, fund = make_market(n_stocks=90, n_days=600, seed=5)
    model = FactorRiskModel(RiskModelSpec(lookback_days=350)).fit(prices, sec, fund)
    stocks = [s for s in model.symbols if s.startswith("S")]
    shares = sec.loc[stocks, "shares_outstanding"].astype(float)
    mc = shares * prices[stocks].iloc[-1]
    bench = mc / mc.sum()
    return model, sec, bench, stocks


def test_min_te_basket_respects_constraints(world):
    model, sec, bench, stocks = world
    spec = BasketSpec(n_max=20, max_weight=0.12, sector_band=0.03)
    res = build_basket(model, bench, stocks, spec, securities=sec)
    assert res.n_names <= 20
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert res.weights.max() <= 0.12 + 1e-4
    assert res.sectors["active"].abs().max() <= 0.03 + 5e-3     # pruning may loosen the band slightly
    assert np.isfinite(res.tracking_error) and res.tracking_error < 0.08
    assert set(res.weights.index) <= set(stocks)


def test_tilt_moves_active_exposure(world):
    model, sec, bench, stocks = world
    flat = build_basket(model, bench, stocks, BasketSpec(n_max=40, sector_band=None), securities=sec)
    tilted = build_basket(model, bench, stocks, BasketSpec(n_max=40, sector_band=None, tilts={"lowvol": 0.5}, tilt_weight=20), securities=sec)
    assert tilted.exposures.loc["lowvol", "active"] > flat.exposures.loc["lowvol", "active"] + 0.15


def test_exclusions_and_etp_filter(world):
    model, sec, bench, stocks = world
    excl = stocks[:10]
    res = build_basket(model, bench, None, BasketSpec(n_max=30, exclude=excl), securities=sec)
    assert not (set(res.weights.index) & set(excl))
    assert "ETFALL" not in res.weights.index and "ETFTEC" not in res.weights.index


def test_analyze_roundtrip(world):
    model, sec, bench, stocks = world
    res = build_basket(model, bench, stocks, BasketSpec(n_max=15), securities=sec)
    again = analyze_basket(model, res.weights, bench)
    assert again.tracking_error == pytest.approx(res.tracking_error, rel=1e-6)
    assert again.n_names == res.n_names
    m = res.metrics()
    assert m["n_names"] == res.n_names and "active_style" in m


def test_benchmark_itself_has_zero_te(world):
    model, sec, bench, stocks = world
    res = analyze_basket(model, bench, bench)
    assert res.tracking_error == pytest.approx(0.0, abs=1e-9)
    assert (res.exposures["active"].abs() < 1e-9).all()
