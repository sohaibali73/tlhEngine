"""Statistical / dynamic risk models, the model library and the calibration study (synthetic market)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.synth import make_market
from tlh.risk import statistical as st
from tlh.risk.calibration import CalibrationGrid, pair_study, run_calibration
from tlh.risk.model import RISK_MODEL_PRESETS, FactorRiskModel, RiskModelSpec, preset_spec


@pytest.fixture(scope="module")
def market():
    return make_market(n_stocks=80, n_days=900, seed=3)


def test_ledoit_wolf_preserves_diagonal_and_shrinks_correlations():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (120, 30))
    w = st.obs_weights(120, "equal")
    S = st.weighted_cov(R, w)
    LW, delta = st.ledoit_wolf_cc(R, w)
    assert 0.0 <= delta <= 1.0
    assert np.allclose(np.diag(LW), np.diag(S))
    off_s = np.abs(S - np.diag(np.diag(S))).mean()
    off_lw = np.abs(LW - np.diag(np.diag(LW))).mean()
    assert off_lw <= off_s + 1e-12
    # a tight pair keeps most of its correlation only with the sample matrix
    x = rng.normal(0, 0.01, 200)
    P = np.column_stack([x, x + rng.normal(0, 0.0005, 200)] + [rng.normal(0, 0.01, 200) for _ in range(20)])
    Ss = st.weighted_cov(P, st.obs_weights(200))
    Ls, _ = st.ledoit_wolf_cc(P, st.obs_weights(200))
    rho_s = Ss[0, 1] / np.sqrt(Ss[0, 0] * Ss[1, 1])
    rho_lw = Ls[0, 1] / np.sqrt(Ls[0, 0] * Ls[1, 1])
    assert rho_s > 0.99 and rho_lw < rho_s


def test_exponential_weights_effective_n_ratio():
    w = st.obs_weights(252, "exponential", halflife_ratio=0.35)
    assert abs(st.effective_n(w) / 252 - 0.765) < 0.02      # the calibration paper's 76.5%


def test_eigen_factorise_reproduces_covariance():
    rng = np.random.default_rng(1)
    B = rng.normal(0, 1, (40, 3))
    Sigma = pd.DataFrame(B @ B.T + np.diag(rng.uniform(0.5, 1.5, 40)), index=[f"S{i}" for i in range(40)], columns=[f"S{i}" for i in range(40)])
    X, F, D, info = st.eigen_factorise(Sigma, n_factors=3, floor_frac=0.0)
    approx = X.values @ F.values @ X.values.T + np.diag(D.values)
    assert np.allclose(np.diag(approx), np.diag(Sigma.values), rtol=1e-6)
    # low-rank part captured: off-diagonal error small relative to scale
    err = np.abs(approx - Sigma.values)
    np.fill_diagonal(err, 0)
    assert err.max() < 0.35 * np.abs(Sigma.values).max()


def test_calibrated_and_pca_fit_give_working_models(market):
    prices, sec, fund = market
    for kind in ("statistical", "pca"):
        spec = RiskModelSpec(model_kind=kind, stat_lookback=126, stat_weighting="exponential")
        m = FactorRiskModel(spec).fit(prices, sec, fund)
        assert len(m.symbols) >= 70 and all(f.startswith("stat:") for f in m.factors)
        w = pd.Series(1 / 10, index=m.symbols[:10])
        b = pd.Series(1 / len(m.symbols), index=m.symbols)
        te = m.tracking_error(w, b)
        assert 0.005 < te < 0.6
        dense = m.covariance(m.symbols[:15]).values
        assert np.allclose(dense, dense.T) and np.all(np.linalg.eigvalsh(dense) > -1e-10)
        assert m.diagnostics["model_kind"] in ("statistical", "pca")


def test_hybrid_adds_stat_factors_and_lowers_specific_risk(market):
    prices, sec, fund = market
    base = FactorRiskModel(RiskModelSpec(model_kind="barra_lite", lookback_days=400)).fit(prices, sec, fund)
    hyb_spec = RiskModelSpec(model_kind="hybrid", lookback_days=400, hybrid_stat_factors=3)
    # hybrid runs on the ERM path; the synthetic market lacks volume so the ERM drops liquidity but still fits
    hyb = FactorRiskModel(hyb_spec).fit(prices, sec, fund)
    stat = [f for f in hyb.factors if f.startswith("stat:")]
    assert len(stat) == 3 and hyb.diagnostics.get("stat_factors") == 3
    assert hyb.specific_var.reindex(base.symbols).dropna().median() <= base.specific_var.median() * 1.05


def test_garch_and_regime_post_processors(market):
    prices, sec, fund = market
    for method in ("garch", "regime"):
        spec = RiskModelSpec(model_kind="barra_lite", lookback_days=400, cov_method=method, horizon_days=21)
        m = FactorRiskModel(spec).fit(prices, sec, fund)
        assert m.diagnostics["cov_method"] == method
        vals = np.linalg.eigvalsh(m.factor_cov.values)
        assert vals.min() > -1e-8
        assert 0.3 < m.diagnostics["dynamic_vs_base_vol_ratio"] < 3.0
    x = np.random.default_rng(2).normal(0, 0.01, 600) * np.repeat([1, 2, 1, 0.5, 1, 1.5], 100)
    v, p = st.garch_forecast_var(x, 21)
    assert v > 0 and 0 < p["persistence"] < 1


def test_every_preset_constructs_and_fits_on_synthetic(market):
    prices, sec, fund = market
    for name in RISK_MODEL_PRESETS:
        spec = preset_spec(name)
        assert spec.preset == name
    # fit the cheap ones end to end
    for name in ("Potomac Calibrated · 126d equal Ledoit-Wolf", "Statistical · PCA (auto factors)", "barra_lite · Fast six-style"):
        m = FactorRiskModel(preset_spec(name, lookback_days=400)).fit(prices, sec, fund)
        assert m.diagnostics.get("preset") == name


def test_calibration_study_scoreboard_and_recommendation(market):
    prices, _, _ = market
    grid = CalibrationGrid(lookbacks=(63, 126), weightings=("equal", "exponential"), estimators=("sample", "ledoit_wolf"),
                           horizons=(21, 63), n_baskets=8, n_pairs=300, max_symbols=40, max_dates_per_horizon=6)
    out = run_calibration(prices, grid)
    board = out["scoreboard"]
    assert len(board) == 2 * 2 * 2 * 2
    for col in ("BiasRatio", "TEBiasRatio", "TESpearman", "CorrRMSE", "Score"):
        assert board[col].notna().all()
    assert (board.groupby("Horizon")["RankInHorizon"].min() == 1).all()
    rec = out["recommendation"]
    assert rec["lookback"] in (63, 126) and rec["estimator"] in ("sample", "ledoit_wolf")
    # shrinkage never touches the diagonal: identical vol bias for both estimators within a lookback/weighting/horizon
    g = board.groupby(["Lookback", "Weighting", "Horizon"])["BiasRatio"].agg(["min", "max"])
    assert np.allclose(g["min"], g["max"], rtol=1e-9)


def test_pair_study_prefers_sample_for_tight_pairs(market):
    prices, _, _ = market
    twin = prices["S000"] * (1 + np.random.default_rng(5).normal(0, 0.0004, len(prices))).cumprod() ** 0.0 * 1.0
    twin = prices["S000"].shift(0) * 1.0
    px = prices.copy()
    px["TWIN"] = twin * (1 + np.random.default_rng(6).normal(0, 0.0003, len(px)))
    df = pair_study(px, [("TWIN", "S000")], lookbacks=(126,), horizon=42, universe_for_shrinkage=list(prices.columns[:40]))
    assert not df.empty
    s = df[df["Estimator"] == "sample"]["abs_bias_dev"].min()
    lw = df[df["Estimator"] == "Ledoit-Wolf"]["abs_bias_dev"].min()
    assert s <= lw + 1e-9
