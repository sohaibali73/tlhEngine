import numpy as np
import pytest

from tlh.optim.glidepath import (
    GlidePathSpec,
    MonteCarloSpec,
    PositionFacts,
    compare_policies,
    evaluate_schedule,
    monte_carlo,
    solve_glidepath,
    tax_curve,
)
from tlh.tax.concentration import (
    BracketSchedule,
    bs_delta,
    bs_price,
    charitable_comparison,
    collar_analysis,
    concentration_stats,
    convex_pieces,
    exchange_fund_breakeven,
    gift_to_lower_bracket,
    ltcg_tax,
    marginal_ltcg_rate,
    ordinary_tax,
    stepup_value,
    tax_from_pieces,
    zero_cost_collar,
)


# ------------------------------------------------------------------ brackets
def test_ltcg_bracket_stacking_hand_computed():
    s = BracketSchedule.default("mfj")           # 0% to 98,900; 15% to 613,700; 20% above; NIIT above 250k
    # 100k gain on 50k income: 48,900 at 0%, 51,100 at 15% = 7,665; NIIT headroom 200k -> 0
    r = ltcg_tax(100_000, 50_000, s)
    assert r["federal"] == pytest.approx(48_900 * 0 + 51_100 * 0.15)
    assert r["niit"] == 0 and r["total"] == pytest.approx(7_665)
    # 1,000,000 gain on 300,000 income: 313,700 at 15% + 686,300 at 20%; NIIT on full gain (already above threshold)
    r = ltcg_tax(1_000_000, 300_000, s)
    assert r["federal"] == pytest.approx(313_700 * 0.15 + 686_300 * 0.20)
    assert r["niit"] == pytest.approx(0.038 * 1_000_000)
    assert marginal_ltcg_rate(1_000_000, 300_000, s) == pytest.approx(0.238)
    assert marginal_ltcg_rate(10_000, 50_000, s) == 0.0
    # state adds linearly
    s2 = BracketSchedule.default("single", state_rate=0.05)
    r2 = ltcg_tax(100_000, 60_000, s2)
    assert r2["state"] == pytest.approx(5_000)
    assert ltcg_tax(0, 100_000, s)["total"] == 0


def test_convex_pieces_match_bracket_tax():
    s = BracketSchedule.default("single", state_rate=0.05)
    for inc in (0, 40_000, 220_000, 700_000):
        pieces = convex_pieces(s, inc, "ltcg")
        for g in (1_000, 30_000, 120_000, 400_000, 2_000_000):
            assert tax_from_pieces(g, pieces) == pytest.approx(ltcg_tax(g, inc, s)["total"], abs=1e-6)
        pieces_o = convex_pieces(s, inc, "ordinary")
        for g in (5_000, 80_000, 500_000):
            assert tax_from_pieces(g, pieces_o) == pytest.approx(ordinary_tax(g, inc, s)["total"], abs=1e-6)


def test_bracket_roundtrip():
    s = BracketSchedule.default("hoh", state_rate=0.03)
    s2 = BracketSchedule.from_dict(s.to_dict())
    assert s2.ltcg == s.ltcg and s2.ordinary == s.ordinary and s2.state_rate == 0.03


# ------------------------------------------------------------------ options
def test_black_scholes_parity_and_collar():
    S, K, T, r, sig, q = 100, 105, 1.0, 0.04, 0.3, 0.01
    c, p = bs_price(S, K, T, r, sig, q, "call"), bs_price(S, K, T, r, sig, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-8)
    assert 0 < bs_delta(S, K, T, r, sig, q, "call") < 1 and -1 < bs_delta(S, K, T, r, sig, q, "put") < 0
    zc = zero_cost_collar(S, 90, T, r, sig, q)
    assert zc["put_premium"] == pytest.approx(zc["call_premium"], rel=1e-4)
    assert zc["call_strike"] > S
    an = collar_analysis(S, 1000, 40, T, sig, r, q, put_strike_pct=0.95, call_strike_pct=1.05, is_long_term=False)
    assert any("CONSTRUCTIVE" in f for f in an["flags"]) and any("STRADDLE" in f for f in an["flags"])
    an2 = collar_analysis(S, 1000, 40, T, sig, r, q, put_strike_pct=0.85, is_long_term=True)
    assert not any("CONSTRUCTIVE" in f for f in an2["flags"])
    assert an2["embedded_gain"] == pytest.approx(60_000) and an2["floor_value"] == pytest.approx(85_000)


# ------------------------------------------------------------------ charitable / alternatives
def test_charitable_and_alternatives():
    c = charitable_comparison(100_000, 20_000, ltcg_rate=0.238, ordinary_marginal_rate=0.37, agi=200_000)
    assert c["donate_shares"]["cap_gains_tax_paid"] == 0 and c["sell_then_donate"]["cap_gains_tax_paid"] == pytest.approx(80_000 * 0.238)
    assert c["advantage_of_donating_shares"] > 0 and c["extra_to_charity"] == pytest.approx(80_000 * 0.238)
    assert c["flags"]                                                     # 100k > 30% of 200k AGI
    g = gift_to_lower_bracket(50_000, 0.238, 0.0)
    assert g["tax_saved"] == pytest.approx(11_900)
    ef = exchange_fund_breakeven(1_000_000, 200_000, 0.238)
    assert ef["tax_deferred"] == pytest.approx(190_400) and ef["pv_of_fees"] > 0
    su = stepup_value(800_000, 0.238, 10, 0.04, 0.5)
    assert su["expected_pv_with_stepup"] == pytest.approx(0.5 * 190_400 / 1.04 ** 10)
    st = concentration_stats(np.array([0.5, 0.2, 0.1, 0.1, 0.1]))
    assert st["hhi"] == pytest.approx(0.32) and st["effective_n"] == pytest.approx(3.125) and st["top1"] == pytest.approx(0.5)


# ------------------------------------------------------------------ glide path
@pytest.fixture
def pos():
    return PositionFacts("XYZ", value=2_000_000, basis=400_000, st_value=0.0, specific_vol=0.30, beta=1.1, total_vol=0.35, total_wealth=5_000_000)


def test_glidepath_respects_constraints_and_beats_naive(pos):
    s = BracketSchedule.default("mfj", state_rate=0.05)
    spec = GlidePathSpec(horizon_years=5, other_taxable_income=300_000, risk_aversion=4.0, annual_gain_budget=500_000, losses_by_year={1: 100_000},
                         carryforward=50_000)
    res = solve_glidepath(pos, spec, s)
    sch = res.schedule
    assert res.status.startswith("optimal")
    assert (sch["realised_gain"] <= 500_000 + 1).all()                 # annual gain budget
    assert sch["cumulative_fraction"].iloc[-1] <= 1.0 + 1e-6
    assert sch["loss_offset_used"].sum() <= 150_000 + 1e-6 and sch["loss_offset_used"].sum() > 0
    assert (sch["tax"] >= 0).all()
    comp = res.comparison.set_index("policy")
    feas = comp[comp["feasible"]]
    assert comp.loc["optimised", "feasible"] and not comp.loc["sell all now", "feasible"]     # sell-all breaks the gain budget
    assert comp.loc["optimised", "total_objective"] <= feas["total_objective"].min() + 1e-6 * abs(feas["total_objective"].min()) + 1.0


def test_glidepath_extremes(pos):
    s = BracketSchedule.default("mfj")
    high_ra = solve_glidepath(pos, GlidePathSpec(horizon_years=3, risk_aversion=200.0), s)
    assert high_ra.schedule["cumulative_fraction"].iloc[0] > 0.8          # very risk averse -> sell almost everything now
    lazy = solve_glidepath(pos, GlidePathSpec(horizon_years=3, risk_aversion=0.0, alpha_view=0.05, p_stepup=1.0), s)
    assert lazy.schedule["sold"].sum() < 1.0                              # no risk penalty, alpha, step-up -> hold


def test_schedule_accounting_and_tax_curve(pos):
    s = BracketSchedule.default("single")
    spec = GlidePathSpec(horizon_years=4, other_taxable_income=100_000)
    d = np.array([500_000, 500_000, 500_000, 500_000.0])
    sch = evaluate_schedule(d, pos, spec, s)
    assert len(sch) == 4 and sch["sold"].iloc[0] == pytest.approx(500_000)
    assert sch["realised_gain"].iloc[0] == pytest.approx(500_000 * (1 - 400_000 / (2_000_000 * 1.07)))
    comp = compare_policies(pos, spec, s, d)
    assert set(comp["policy"]) == {"optimised", "sell all now", "equal instalments", "hold to horizon"}
    tc = tax_curve(np.array([0, 50_000, 500_000]), 100_000, s)
    assert (tc["tax"] - tc["pieces_tax"]).abs().max() < 1e-6


def test_monte_carlo_shapes(pos):
    s = BracketSchedule.default("mfj")
    gp = GlidePathSpec(horizon_years=3)
    out = monte_carlo(pos, gp, MonteCarloSpec(n_paths=500, horizon_years=3, seed=1), s, optimised=np.array([600_000, 600_000, 600_000.0]))
    assert set(out["policies"]) == {"hold", "sell all now", "equal instalments", "optimised"}
    h, sn = out["policies"]["hold"], out["policies"]["sell all now"]
    assert h["fan"].shape == (4, 5) and h["std"] > sn["std"]              # concentrated is riskier than diversified
    assert 0 <= h["p_beats_sell_now"] <= 1
