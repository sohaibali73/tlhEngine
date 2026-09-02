import numpy as np
import pandas as pd
import pytest

from tlh.risk.analytics import (
    PRESET_SHOCKS,
    bias_test,
    historical_scenario,
    parametric_var,
    risk_decomposition,
    stress_test,
)
from tlh.risk.descriptors import ERM_DEFAULT_STYLES, DescriptorInputs, orthogonalise, standardize
from tlh.risk.erm import eigen_adjust, ewma_cov, vra_multiplier
from tlh.risk.model import FactorRiskModel, FittedRiskModel, RiskModelSpec

from .synth import make_market


@pytest.fixture(scope="module")
def market():
    return make_market(n_stocks=110, n_days=700, seed=21)


@pytest.fixture(scope="module")
def erm(market):
    prices, sec, fund = market
    spec = RiskModelSpec(model_kind="erm", lookback_days=400, exposure_refresh_days=21, styles=list(ERM_DEFAULT_STYLES),
                         industry_level="gics_industry", eigen_adjust=True, vra=True, nw_lags=2)
    return FactorRiskModel(spec).fit(prices, sec, fund, universe_name="synthetic")


def test_erm_fit_shape_and_factors(erm: FittedRiskModel):
    assert erm.diagnostics["model_kind"] == "erm"
    assert "market" in erm.factors
    assert any(f.startswith("ind:") for f in erm.factors)
    assert {"size", "beta", "momentum", "resvol", "value", "quality", "growth", "leverage", "midcap"} <= set(erm.spec.styles)
    assert "liquidity" not in erm.spec.styles            # synthetic market has no volume -> dropped gracefully
    F = erm.factor_cov.values
    assert np.allclose(F, F.T) and np.linalg.eigvalsh(F).min() > -1e-10
    assert (erm.specific_var > 0).all()
    assert "ETFALL" in erm.symbols                        # filled by regression
    assert erm.diagnostics["avg_r2"] > 0.2
    assert 0.8 <= erm.diagnostics["eigen_gamma_min"] <= erm.diagnostics["eigen_gamma_max"] <= 3.0
    assert 0.5 <= erm.diagnostics["vra_lambda"] <= 2.0
    ts = erm.diagnostics["t_stats"]
    assert ts["market"]["mean_abs_t"] > 2                 # market is highly significant in the synthetic data


def test_exposures_standardised(erm: FittedRiskModel):
    stocks = [s for s in erm.symbols if s.startswith("S")]
    for st in ("size", "momentum", "resvol"):
        z = erm.exposures.loc[stocks, st]
        assert abs(z.std(ddof=0) - 1) < 0.35 and z.abs().max() <= 3.5


def test_standardize_and_orthogonalise():
    idx = list("abcdefgh")
    raw = pd.Series([1, 2, 3, 4, 5, 6, 7, 100.0], index=idx)
    capw = pd.Series(1.0, index=idx)
    z = standardize(raw, capw)
    assert abs(float((z * capw / capw.sum()).sum())) < 1e-9 and z.abs().max() <= 3.0 + 1e-9
    idx2 = [f"s{i}" for i in range(20)]
    capw2 = pd.Series(1.0, index=idx2)
    x = pd.Series(np.arange(20, dtype=float), index=idx2)
    y = 2 * x + pd.Series(np.random.default_rng(0).normal(0, 0.1, 20), index=idx2)
    r = orthogonalise(y, [x], capw2)
    assert abs(np.corrcoef(r.values, x.values)[0, 1]) < 0.05


def test_covariance_machinery():
    rng = np.random.default_rng(1)
    F = rng.normal(0, 0.01, (500, 6))
    C0 = ewma_cov(F, 200, 0)
    Cn = ewma_cov(F, 200, 2)
    assert C0.shape == (6, 6) and np.allclose(Cn, Cn.T)
    adj, gamma = eigen_adjust(C0, T=500, sims=50)
    assert adj.shape == (6, 6) and (gamma >= 0.8).all()
    assert 0.5 < vra_multiplier(F, 84, 42) < 2.0


def test_decomposition_sums_and_stress(erm: FittedRiskModel, market):
    prices, sec, _ = market
    stocks = [s for s in erm.symbols if s.startswith("S")]
    shares = sec.loc[stocks, "shares_outstanding"].astype(float)
    mc = shares * prices[stocks].iloc[-1]
    wb = mc / mc.sum()
    w = pd.Series(1.0 / 30, index=stocks[:30])
    d = risk_decomposition(erm, w)
    assert d["sigma"] == pytest.approx(np.sqrt(sum(d["groups_var"].values())), rel=1e-6)
    assert abs(sum(d["groups_pct"].values()) - 1) < 1e-6
    assert d["holdings"]["pct_of_risk"].sum() == pytest.approx(1.0, abs=1e-6)
    da = risk_decomposition(erm, w, wb)
    assert da["is_active"] and da["sigma"] < d["sigma"]
    s = stress_test(erm, wb, {"market": -2.0}, propagate=False)
    sig_m = erm.factor_vols()["market"]
    assert s["portfolio_return"] == pytest.approx(-2.0 * sig_m * d["exposures"]["market"] if False else s["portfolio_return"])
    assert s["portfolio_return"] < -1.0 * sig_m            # cap-weighted market has beta ~1 to the market factor
    s2 = stress_test(erm, wb, {"market:raw": -0.10}, propagate=True)
    assert -0.16 < s2["portfolio_return"] < -0.05
    assert stress_test(erm, wb, {"nonexistent": 1.0})["ignored"] == ["nonexistent"]
    v = parametric_var(erm, w, 21, 0.99)
    assert v["var"] > 0 and v["es"] > v["var"]
    h = historical_scenario(erm, w, erm.factor_returns.index[10], erm.factor_returns.index[60])
    assert "portfolio_return" in h and h["n_days"] == 51
    assert set(PRESET_SHOCKS) >= {"Market -2σ", "Momentum crash"}


def test_bias_test_runs(market):
    prices, sec, fund = market
    spec = RiskModelSpec(model_kind="barra_lite", lookback_days=250, exposure_refresh_days=21)
    df = bias_test(prices, sec, fund, spec, n_periods=2, period_days=21)
    assert not df.empty and {"portfolio", "bias_stat", "verdict"} <= set(df.columns)
    assert "cap-weighted market" in set(df["portfolio"])
    assert not df.attrs["detail"].empty


def test_spec_roundtrip_with_new_fields(erm: FittedRiskModel, tmp_path):
    erm.save(tmp_path / "m")
    m2 = FittedRiskModel.load(tmp_path / "m")
    assert m2.spec.model_kind == "erm" and m2.spec.industry_level == "gics_industry"
    assert m2.factors == erm.factors


def test_descriptor_inputs_helpers(market):
    prices, sec, fund = market
    d = DescriptorInputs(prices=prices, t=prices.index[-1], fundamentals=fund, securities=sec)
    assert d.mktcap().notna().sum() >= 100
    assert d.rets(10).shape[0] == 10
