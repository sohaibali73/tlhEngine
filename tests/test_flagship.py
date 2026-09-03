"""New strategies (multi-factor, defensive, long/short extension, overlay), overlay planner, long/short economics,
state tax table, holdings import parsing, explanations and the wealth projection."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.synth import make_market
from tlh.explain import explain_harvest, explain_state
from tlh.optim import longshort as ls
from tlh.optim import overlay as ov
from tlh.optim.basket_library import LIBRARY, recipe_table
from tlh.optim.strategies import STRATEGIES, StrategyInputs, StrategySpec, run_strategy
from tlh.risk.model import FactorRiskModel, RiskModelSpec
from tlh.services.import_service import guess_mapping, plan_import, template_csv
from tlh.tax import state_rates as sr


@pytest.fixture(scope="module")
def fitted():
    prices, sec, fund = make_market(n_stocks=90, n_days=700, seed=11)
    model = FactorRiskModel(RiskModelSpec(model_kind="barra_lite", lookback_days=400)).fit(prices, sec, fund)
    stocks = [s for s in model.symbols if s.startswith("S")]
    cov = model.covariance(stocks)
    bench = pd.Series(1.0 / len(stocks), index=stocks)
    styles = [c for c in model.factors if c in model.spec.styles]
    rets = prices[stocks].pct_change().iloc[-300:]
    cur = pd.Series(0.0, index=stocks)
    cur[stocks[:20]] = 1.0 / 20
    inp = StrategyInputs(symbols=stocks, cov=cov, benchmark=bench, returns=rets, signals=model.exposures.loc[stocks, styles],
                         exposures=model.exposures.loc[stocks], sectors=sec["gics_sector"].reindex(stocks), current_weights=cur)
    return model, inp, stocks


def test_multi_factor_integrated_and_mixed(fitted):
    _, inp, _ = fitted
    a = run_strategy(StrategySpec(kind="multi_factor", n_max=30, max_weight=0.08, sector_band=0.05), inp)
    assert a.n_names <= 30 and abs(a.weights.sum() - 1) < 1e-6 and a.diagnostics["approach"] == "integrated"
    assert a.diagnostics["composite_exposure"] > 0
    b = run_strategy(StrategySpec(kind="multi_factor", n_max=30, max_weight=0.08, integrated=False), inp)
    assert b.diagnostics["approach"] == "mixed" and abs(b.weights.sum() - 1) < 1e-6


def test_defensive_equity_respects_beta_cap(fitted):
    _, inp, _ = fitted
    r = run_strategy(StrategySpec(kind="defensive_equity", n_max=30, max_weight=0.08, beta_cap=0.85, sector_band=None), inp)
    assert r.diagnostics["beta"] <= 0.85 + 1e-4
    assert abs(r.weights.sum() - 1) < 1e-6


def test_long_short_extension_books(fitted):
    _, inp, _ = fitted
    r = run_strategy(StrategySpec(kind="long_short_extension", extension=0.30, max_weight=0.06, short_max_weight=0.03, sector_band=0.03,
                                  beta_target=1.0, beta_tolerance=0.05, risk_aversion=10.0), inp)
    d = r.diagnostics
    assert d["long_short"] and abs(d["long_exposure"] - 1.30) < 1e-3 and abs(d["short_exposure"] + 0.30) < 1e-3
    assert abs(d["net"] - 1.0) < 1e-3 and d["n_short"] >= 5
    assert abs(d["beta"] - 1.0) <= 0.05 + 1e-3
    assert not (set(d["long_weights"]) & set(d["short_weights"]))          # never long and short the same name
    assert (r.weights < 0).any() and r.weights.sum() == pytest.approx(1.0, abs=1e-3)


def test_overlay_neutral_keeps_core_and_never_shorts_held(fitted):
    _, inp, stocks = fitted
    r = run_strategy(StrategySpec(kind="overlay_neutral", extension=0.30, max_weight=0.10, short_max_weight=0.03, sector_band=0.03, risk_aversion=10.0), inp)
    d = r.diagnostics
    held = set(stocks[:20])
    assert d["core_untouched"] and not (set(d["short_weights"]) & held)
    assert abs(d["long_exposure"] - 1.30) < 1e-3 and abs(d["short_exposure"] + 0.30) < 1e-3
    assert abs(d["extension_beta"]) < 0.06
    for s in held:                                    # core weights are preserved (only added to, never sold)
        assert r.weights.get(s, 0.0) >= 1.0 / 20 - 1e-6


def test_catalogue_and_library_consistency():
    for r in LIBRARY:
        assert r.kind in STRATEGIES
    t = recipe_table()
    assert len(t) == len(LIBRARY) and t["name"].is_unique


def test_overlay_plan_sizes_micro_contracts_and_flags_1256():
    plan = ov.plan_overlay(ov.OverlayInputs(portfolio_value=2_000_000, portfolio_beta=1.0, target_beta=1.0, cash=150_000, index_level=5600.0,
                                            contract="MES", harvested_losses=50_000))
    # 150k of cash lowers beta to 0.925 -> need +150k notional = 150000 / (5 * 5600) ≈ 5.4 -> 5 contracts
    assert plan.contracts == 5 and abs(plan.notional - 5 * 5 * 5600) < 1e-6
    assert plan.beta_after > 0.99 and plan.margin_required == pytest.approx(plan.notional * 0.055)
    assert any("§1256" in f for f in plan.flags)
    short = ov.plan_overlay(ov.OverlayInputs(portfolio_value=1_000_000, portfolio_beta=1.0, target_beta=0.5, index_level=5600.0, contract="ES", embedded_gain_hedged=400_000))
    assert short.contracts < 0 and any("§1092" in f for f in short.flags)
    assert short.tax["rate_advantage_vs_st"] > 0
    tbl = ov.micro_vs_mini(300_000, 5600.0)
    assert set(tbl["contract"]) == {"ES", "MES"} and abs(tbl.set_index("contract").loc["MES", "rounding_error"]) <= abs(tbl.set_index("contract").loc["ES", "rounding_error"])


def test_long_short_simulation_and_exchange_glide():
    res = ls.simulate_loss_generation(ls.LongShortSpec(years=5, n_paths=30, n_long=60, n_short=40))
    by = res.by_year
    assert len(by) == 5
    assert by["long_short_net_loss_pct"].iloc[0] > by["long_only_net_loss_pct"].iloc[0]          # extension adds losses in year 1
    assert by["long_only_net_loss_pct"].iloc[-1] < by["long_only_net_loss_pct"].iloc[0]          # long-only decays
    assert res.summary["uplift_vs_long_only"] > 0
    fin = ls.financing_cost(0.30)
    assert fin["net_pre_tax"] < 0 and fin["net_post_tax"] > fin["net_pre_tax"]
    g = ls.exchange_glide(10_000_000, 1_000_000, extension=0.30, years=12)
    assert (g["net_tax"].abs().iloc[:-1] < 1e-6).all()             # tax-neutral by construction (final year sells the remainder)
    assert g["concentrated_remaining"].is_monotonic_decreasing
    assert g.attrs["years_to_full_divestiture"] is not None
    assert len(ls.years_to_diversify_table()) == 4


def test_state_rates_cover_every_state_and_key_cases():
    t = sr.table(300_000)
    assert len(t) == 51 and t["abbrev"].is_unique
    assert set(t["treatment"]) <= {"ordinary", "none", "exclusion", "flat_cg", "excise"}
    assert sr.state_rates("TX")["lt_rate"] == 0 and sr.state_rates("FL")["st_rate"] == 0
    ca = sr.combined_marginal("CA", "mfj", 400_000)
    assert 0.35 < ca["total_st"] < 0.40 and ca["state_lt"] == ca["state_st"]
    wa = sr.combined_marginal("WA", "single", 300_000, gain=500_000)
    assert wa["state_st"] == 0 and abs(wa["state_lt"] - 0.07) < 1e-9
    assert abs(sr.state_rates("SC", 100_000)["lt_rate"] - 0.062 * 0.56) < 1e-9
    assert sr.state_rates("MT", 100_000)["lt_rate"] == 0.041 and sr.state_rates("MT", 100_000)["st_rate"] == 0.059
    ma = sr.state_rates("MA", 2_000_000)
    assert abs(ma["st_rate"] - 0.125) < 1e-9 and abs(ma["lt_rate"] - 0.09) < 1e-9
    md = sr.state_rates("MD", 400_000)
    assert abs(md["lt_rate"] - (0.0575 + 0.02)) < 1e-9
    rank = sr.rank_for_harvesting()
    assert rank.iloc[0]["abbrev"] == "CA"
    assert "approximate" in explain_state(ca)


def test_import_plan_parses_broker_export(tmp_path):
    p = tmp_path / "schwab.csv"
    p.write_text("Positions for account Individual ...XYZ as of 09/01/2026\n\n"
                 "\"Symbol\",\"Description\",\"Quantity\",\"Price\",\"Cost Basis\",\"Date Acquired\"\n"
                 "\"AAPL\",\"APPLE INC\",\"100\",\"$228.10\",\"$14,825.00\",\"03/15/2023\"\n"
                 "\"MSFT\",\"MICROSOFT\",\"40\",\"$410.00\",\"--\",\"\"\n"
                 "\"Cash & Cash Investments\",\"--\",\"--\",\"--\",\"$5,000\",\"--\"\n"
                 "\"Account Total\",\"--\",\"--\",\"--\",\"$50,000\",\"--\"\n", encoding="utf-8")
    plan = plan_import(p)
    f = plan.frame
    assert list(f["symbol"]) == ["AAPL", "MSFT"]
    assert f.iloc[0]["cost_per_share"] == pytest.approx(148.25) and f.iloc[0]["acquired"] == date(2023, 3, 15)
    assert f.iloc[1]["cost_per_share"] == pytest.approx(410.0) and "assumed" in f.iloc[1]["flags"]
    m = guess_mapping(["Ticker", "Shares", "Avg Cost", "Purchase Date", "Account Name"])
    assert m["symbol"] == "Ticker" and m["quantity"] == "Shares" and m["cost_per_share"] == "Avg Cost" and m["acquired"] == "Purchase Date"
    t = tmp_path / "t.csv"
    t.write_text(template_csv(), encoding="utf-8")
    assert plan_import(t).n_rows == 3


def test_explanations_and_projection():
    s = {"harvested_loss": 12_000.0, "harvested_loss_st": 8_000.0, "harvested_loss_lt": 4_000.0, "tax_benefit": 4_100.0, "tax_alpha": 2_500.0,
         "n_sells": 4, "n_buys": 3, "te_before": 0.012, "te_after": 0.013, "n_blocked_lots": 1, "portfolio_value": 1_000_000.0}
    lines = explain_harvest(s, 0.408, 0.238, "the S&P 500")
    assert any("$12,000" in x for x in lines) and any("wash-sale" in x for x in lines) and any("Nothing here trades" in x for x in lines)
    assert explain_harvest({"harvested_loss": 0.0, "n_blocked_lots": 2}, 0.4, 0.2)[0].startswith("No wash-safe losses")
    from tlh.services.home_service import HomeService
    pr = HomeService.wealth_projection(1_000_000, 0.408, 0.238, years=10)
    assert len(pr["years"]) == 11 and pr["tlh_after_deferred_tax"][-1] > pr["hold"][-1]
    assert np.isfinite(pr["gain_vs_hold"])
