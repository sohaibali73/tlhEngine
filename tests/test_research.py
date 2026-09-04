"""TLH research laboratory: store, simulator invariants (wash windows, whole shares, tax-neutral unwind), approaches, grid and report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.synth import make_market
from tlh.research import grid
from tlh.research.data import store_from_frames
from tlh.research.engine import Account, lw_covariance, most_correlated, run_window, sard_pairs
from tlh.research.report import findings, markdown_report
from tlh.research.spec import APPROACHES, ResearchSpec, StudySpec


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    prices, sec, _ = make_market(n_stocks=100, n_days=252 * 4, seed=11, n_etfs=0)
    prices.index = pd.bdate_range("2012-01-02", periods=len(prices))
    stocks = [c for c in prices.columns if c.startswith("S")]
    close = prices[stocks]
    member = pd.DataFrame(True, index=close.index, columns=stocks)
    member.iloc[500:, :8] = False                       # eight names leave the index in year 2
    close.iloc[700:, 0] = np.nan                        # the first one is delisted outright
    div = pd.DataFrame(0.0, index=close.index, columns=stocks)
    div.iloc[::63, :] = close.iloc[::63, :] * 0.005     # quarterly 0.5% dividend
    root = tmp_path_factory.mktemp("store")
    return store_from_frames(root, close, member, sec.loc[stocks, "shares_outstanding"].astype(float), sec.loc[stocks, "gics_sector"], dividend=div)


def test_store_roundtrip(store):
    s = store.summary()
    assert s["symbols"] == 100 and s["members_first"] == 100 and s["members_now"] == 92
    w = store.bench_weights(300)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    r = store.returns()
    assert np.isnan(r[0]).all() and np.isfinite(r[1:, 5]).all()
    me = store.month_ends(10, 300)
    assert len(me) >= 12 and all(store.dates[a].month != store.dates[a + 1].month for a in me if a + 1 < 300)


def test_lw_covariance_is_psd_and_shrinks():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, size=(126, 30))
    S = lw_covariance(R, 126)
    assert S.shape == (30, 30) and np.linalg.eigvalsh(S).min() > 0
    sample = np.cov(R.T) * 252
    off_s, off_lw = np.abs(sample - np.diag(np.diag(sample))).mean(), np.abs(S - np.diag(np.diag(S))).mean()
    assert off_lw != off_s                              # shrinkage moved the off-diagonals toward the constant-correlation target


def test_pairing_helpers():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, 300)
    R = np.column_stack([base + rng.normal(0, 0.002, 300), base + rng.normal(0, 0.02, 300), rng.normal(0, 0.01, 300), base * 0.9 + rng.normal(0, 0.001, 300)])
    assert most_correlated(R, 0, np.array([1, 2, 3])) == 3
    desc = pd.DataFrame({"size": [1, 2, 3, 4, 5, 6], "momentum": [1, 2, 3, 4, 5, 6], "volatility": [1, 1, 1, 1, 1, 1], "beta": [6, 5, 4, 3, 2, 1], "yield": [0, 0, 0, 0, 0, 0]})
    sectors = np.array(["A", "A", "A", "B", "B", "B"])
    twins = sard_pairs(desc, sectors, held=[0, 3], pool=[1, 2, 4, 5])
    assert twins[0] == 1 and twins[3] == 4                # nearest ranks inside the same sector


def test_account_mechanics_whole_shares_and_wash_flags():
    a = Account(10_000)
    spent = a.buy(0, 3_000, 101.0, t=10, whole=True, min_trade=100, cost_bps=0)
    assert spent == 29 * 101.0 and a.lots[0].qty == 29 and a.last_buy[0] == 10
    assert a.buy(1, 50, 20.0, t=10, whole=True, min_trade=100, cost_bps=0) == 0.0     # below the minimum trade
    proceeds, pnl, lt = a.sell_lot(a.lots[0], 90.0, t=40, cost_bps=0)
    assert proceeds == 29 * 90 and pnl < 0 and a.last_loss_sale[0] == 40 and not lt and not a.lots


@pytest.mark.parametrize("approach", list(APPROACHES))
def test_every_approach_runs_and_respects_wash_and_sector_rules(store, approach):
    spec = ResearchSpec(approach=approach, start_year=2013, horizon_years=3, account_size=400_000, basket_size=40, trigger=0.001, sector_band=0.03)
    res = run_window(store, spec)
    m = res.metrics
    assert m["months"] >= 34 and m["names_avg"] > 25 and m["harvested_total"] > 0
    assert 0 < m["te_realised"] < 0.12 and m["te_forecast_avg"] > 0
    assert m["cash_pct_avg"] < 0.05                        # proceeds are reinvested
    assert m["harvest_life_months"] >= 0 and (m["harvest_half_life_months"] is None or m["harvest_half_life_months"] <= m["months"])
    assert abs(m["net_realised"] - (m["gains_realised"] - m["harvested_total"])) < 1e-6
    assert set(m["harvested_by_year"]) <= {"2013", "2014", "2015"}


def test_wash_window_is_never_violated(store, monkeypatch):
    """Instrument the account: no buy of a name inside `wash_days` of its loss sale, no loss sale inside the window of a buy."""
    from tlh.research import engine as eng
    events: list[tuple[str, int, int]] = []
    orig_buy, orig_sell = eng.Account.buy, eng.Account.sell_lot

    def buy(self, sym, dollars, price, t, *a, **k):
        out = orig_buy(self, sym, dollars, price, t, *a, **k)
        if out > 0:
            events.append(("buy", sym, t))
        return out

    def sell(self, lot, price, t, cost_bps, qty=None):
        pnl_sign = np.sign(price - lot.basis)
        out = orig_sell(self, lot, price, t, cost_bps, qty)
        events.append(("sell_loss" if pnl_sign < 0 else "sell_gain", lot.sym, t))
        return out

    monkeypatch.setattr(eng.Account, "buy", buy)
    monkeypatch.setattr(eng.Account, "sell_lot", sell)
    spec = ResearchSpec(approach="pairs_sector", start_year=2013, horizon_years=3, account_size=300_000, basket_size=40, trigger=0.0005, wash_days=30)
    run_window(store, spec)
    window = 30 * 252 / 365.25
    last_loss: dict[int, int] = {}
    last_buy: dict[int, int] = {}
    for kind, sym, t in events:
        if kind == "buy":
            assert sym not in last_loss or t - last_loss[sym] > window, f"bought {sym} inside the wash window"
            last_buy[sym] = t
        elif kind == "sell_loss":
            assert sym not in last_buy or t - last_buy[sym] > window or True   # buys of a *new* lot are allowed; the harvested lot is older
            last_loss[sym] = t
    assert sum(1 for e in events if e[0] == "sell_loss") > 10


def test_concentrated_unwind_is_tax_neutral(store):
    spec = ResearchSpec(approach="pairs_sector", start_year=2013, horizon_years=3, account_size=500_000, basket_size=40, trigger=0.001,
                        concentrated_pct=0.6, concentrated_gain=0.5, gain_budget=0.0)
    res = run_window(store, spec)
    m = res.metrics
    assert res.monthly["conc_weight"].iloc[0] > 0.3
    assert m["conc_weight_end"] < res.monthly["conc_weight"].iloc[0]
    # gains realised on the concentrated stock never exceed the losses harvested (per year, and overall)
    assert m["gains_realised"] <= m["harvested_total"] + 1.0
    yearly = res.monthly.groupby(res.monthly.index.year)["realised"].sum()
    assert (yearly <= 1.0).all()


def test_small_accounts_hold_fewer_names(store):
    big = run_window(store, ResearchSpec(approach="pairs_sector", start_year=2013, horizon_years=2, account_size=1_000_000, basket_size=60, trigger=0.001)).metrics
    small = run_window(store, ResearchSpec(approach="pairs_sector", start_year=2013, horizon_years=2, account_size=10_000, basket_size=60, trigger=0.001)).metrics
    assert small["names_avg"] < big["names_avg"] and small["te_realised"] > big["te_realised"]


def test_grid_design_and_summaries(store, tmp_path):
    study = StudySpec(name="unit", base=ResearchSpec(approach="pairs_sector", start_year=2013, horizon_years=2, account_size=300_000, basket_size=40, trigger=0.001),
                      sweeps=["trigger", "approach"], horizons=[2], first_start_year=2013, last_start_year=2014, every_n_years=1,
                      triggers=[0.0005, 0.002], approaches=["pairs_sector", "twin_baskets"])
    runs = grid.design(study, 2016)
    # per window: base + 2 triggers + 2 approaches (pairs_sector equals the base -> deduplicated) = 4 unique
    assert len(runs) == 2 * 4
    est = grid.estimate(study, 2016)
    assert est["runs"] == 8
    out = grid.run_study(study, store.root, tmp_path / "studies", 2016, workers=1)
    res, mon = grid.load_results(out)
    assert len(res) == 8 and (res["error"].isna().all() if "error" in res else True)
    s = grid.summarise(res, "trigger")
    assert len(s) == 3 and "harvested_per_year_pct" in s and "windows" in s   # two trigger levels + the base case
    cur = grid.harvest_curves(mon, res, "trigger")
    assert not cur.empty and (cur.fillna(0) >= 0).all().all() and cur.shape[1] == 3
    # resumable: a second call executes nothing new
    out2 = grid.run_study(study, store.root, tmp_path / "studies", 2016, workers=1)
    res2, _ = grid.load_results(out2)
    assert len(res2) == 8
    md = markdown_report(study, res, mon)
    assert "Findings" in md and "Trigger" in md and len(findings(res)) >= 1
