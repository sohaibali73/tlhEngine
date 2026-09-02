import numpy as np
import pandas as pd
import pytest

from tlh.risk.factors import FactorInputs, build_exposures, standardize
from tlh.risk.model import FactorRiskModel, FittedRiskModel, RiskModelSpec

from .synth import make_market


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.fixture(scope="module")
def fitted(market):
    prices, sec, fund = market
    spec = RiskModelSpec(lookback_days=400, exposure_refresh_days=21)
    return FactorRiskModel(spec).fit(prices, sec, fund, universe_name="synthetic")


def test_standardize_cap_weighted_mean_zero():
    raw = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, np.nan], index=list("abcdef"))
    capw = pd.Series([1, 1, 1, 1, 6, 1.0], index=list("abcdef"))
    z = standardize(raw, capw)
    w = capw[:5] / capw[:5].sum()
    assert abs(float((z[:5] * w).sum())) < 1e-9
    assert np.isnan(z["f"])
    assert z.abs().max() <= 3.0


def test_build_exposures_shapes(market):
    prices, sec, fund = market
    fi = FactorInputs(prices=prices, t=prices.index[-1], fundamentals=fund, securities=sec)
    b = build_exposures(fi, ["value", "momentum", "quality", "size", "lowvol", "growth"])
    assert set(b.style_cols) == {"value", "momentum", "quality", "size", "lowvol", "growth"}
    assert len(b.sector_cols) == 4
    assert "ETFALL" in b.missing_style      # ETFs have no fundamentals -> missing styles
    assert b.cap_weights["ETFALL"] == 0.0


def test_fit_basic_properties(fitted: FittedRiskModel):
    assert fitted.exposures.shape[0] >= 120                       # stocks + regression-filled ETFs
    assert "ETFALL" in fitted.symbols and "ETFTEC" in fitted.symbols
    F = fitted.factor_cov.values
    assert np.allclose(F, F.T)
    assert np.linalg.eigvalsh(F).min() > -1e-10                  # PSD
    assert (fitted.specific_var > 0).all()
    assert fitted.diagnostics["avg_r2"] > 0.2
    mvol = fitted.factor_vols()["market"]
    assert 0.10 < mvol < 0.25                                     # true daily 1% -> ~16% annual


def test_etf_exposures_from_regression(fitted: FittedRiskModel):
    e = fitted.exposures.loc["ETFALL"]
    assert 0.8 < e["market"] < 1.2
    # tech ETF loads on the Tech sector dummy far more than the broad ETF
    assert fitted.exposures.loc["ETFTEC", "sec:Tech"] > fitted.exposures.loc["ETFALL", "sec:Tech"] + 0.3


def test_te_consistency_with_covariance(fitted: FittedRiskModel):
    syms = fitted.symbols[:30]
    rng = np.random.default_rng(1)
    w = pd.Series(rng.dirichlet(np.ones(30)), index=syms)
    b = pd.Series(rng.dirichlet(np.ones(30)), index=syms)
    te = fitted.tracking_error(w, b)
    cov = fitted.covariance(syms).values
    a = (w - b).values
    assert te == pytest.approx(float(np.sqrt(a @ cov @ a)), rel=1e-8)
    assert fitted.tracking_error(w, w) == pytest.approx(0.0, abs=1e-12)
    dec = fitted.te_decomposition(w, b)
    assert dec["variance"].sum() == pytest.approx(te ** 2, rel=1e-8)
    assert dec.attrs["tracking_error"] == pytest.approx(te)


def test_benchmark_etf_vs_basket_low_te(fitted: FittedRiskModel, market):
    prices, sec, _ = market
    stocks = [s for s in fitted.symbols if s.startswith("S")]
    shares = sec.loc[stocks, "shares_outstanding"].astype(float)
    mc = shares * prices[stocks].iloc[0]
    wb = mc / mc.sum()
    te = fitted.tracking_error(pd.Series({"ETFALL": 1.0}), wb)
    assert te < 0.04                                              # ETF replicates its own basket


def test_save_load_roundtrip(fitted: FittedRiskModel, tmp_path):
    fitted.save(tmp_path / "m")
    m2 = FittedRiskModel.load(tmp_path / "m")
    assert m2.factors == fitted.factors
    pd.testing.assert_frame_equal(m2.exposures, fitted.exposures)
    assert m2.spec.lookback_days == 400


def test_macro_block(market):
    prices, sec, fund = market
    rng = np.random.default_rng(3)
    macro = pd.DataFrame({
        "rate_10y": 4 + np.cumsum(rng.normal(0, 0.03, len(prices))),
        "slope_10y2y": 0.2 + np.cumsum(rng.normal(0, 0.02, len(prices))),
        "credit_spread": 1.0 + np.cumsum(rng.normal(0, 0.01, len(prices))),
        "usd": 100 * np.cumprod(1 + rng.normal(0, 0.003, len(prices))),
    }, index=prices.index)
    spec = RiskModelSpec(lookback_days=300, use_macro=True)
    m = FactorRiskModel(spec).fit(prices, sec, fund, macro_levels=macro)
    assert any(f.startswith("macro:") for f in m.factors)
    assert m.factor_cov.shape[0] == len(m.factors)
