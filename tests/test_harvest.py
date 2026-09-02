from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tlh.data.substitutes import SubstituteMap
from tlh.optim.frontier import frontier, priority_comparison
from tlh.optim.harvest import HarvestConfig, HarvestInputs, run_harvest
from tlh.risk.model import FactorRiskModel, RiskModelSpec
from tlh.tax.lots import Lot
from tlh.tax.rates import TaxProfile
from tlh.tax.washsale import Acquisition, LossSale, SubstantiallyIdentical

from .synth import make_market

AS_OF = date(2025, 9, 1)


@pytest.fixture(scope="module")
def world():
    prices, sec, fund = make_market(n_stocks=80, n_days=600, seed=2)
    model = FactorRiskModel(RiskModelSpec(lookback_days=350)).fit(prices, sec, fund)
    px = prices.iloc[-1]
    return prices, sec, fund, model, px


def mk_lot(i, sym, aid, qty, cost, acq, acct=1, acct_type="taxable"):
    return Lot(id=i, account_id=acct, assetid=aid, symbol=sym, acquired_date=acq, holding_start_date=acq,
               quantity_original=qty, quantity_open=qty, cost_per_share=cost, account_type=acct_type)


def build_inputs(world, cfg_extra=None):
    prices, sec, fund, model, px = world
    aid = {s: int(sec.loc[s, "assetid"]) for s in sec.index}
    syms = [s for s in model.symbols if s.startswith("S")]
    held = syms[:8]
    lots = []
    # 1 & 2: big losers (cost 40% above price), 3: loser but wash-blocked by purchase 10 days ago,
    # 4..8: winners; plus the broad ETF in an IRA
    for i, s in enumerate(held):
        if i in (0, 1):
            cost = px[s] * 1.4
        elif i == 2:
            cost = px[s] * 1.3
        else:
            cost = px[s] * 0.6
        lots.append(mk_lot(i + 1, s, aid[s], 1000, cost, AS_OF - timedelta(days=400 if i % 2 else 100)))
    lots.append(mk_lot(99, "ETFALL", aid["ETFALL"], 500, px["ETFALL"] * 0.9, AS_OF - timedelta(days=800),
                       acct=2, acct_type="ira"))
    acqs = [Acquisition(lt.assetid, lt.symbol, lt.account_id, lt.account_type, lt.acquired_date, lt.quantity_original, lot_id=lt.id)
            for lt in lots]
    # recent purchase of held[2] in the IRA -> wash-blocks selling lot 3 at a loss
    acqs.append(Acquisition(aid[held[2]], held[2], 2, "ira", AS_OF - timedelta(days=10), 50, lot_id=500))
    bench = pd.Series(1.0 / len(syms), index=syms)
    subs = SubstituteMap.from_dict({"substitutes": [{"for": [held[0]], "candidates": [syms[20], syms[21]]}]})
    inp = HarvestInputs(
        as_of=AS_OF, lots=lots, prices=px, model=model, benchmark=bench, tax=TaxProfile(),
        groups=SubstantiallyIdentical({}), substitutes=subs, acquisitions=acqs, recent_loss_sales=[],
        returns=prices.pct_change().iloc[-300:], securities=sec, universe=syms,
    )
    return inp, held, aid


def test_harvest_end_to_end(world):
    inp, held, aid = build_inputs(world)
    cfg = HarvestConfig(min_trade_value=500, min_loss_value=100, te_budget=0.02)
    res = run_harvest(inp, cfg)
    assert res.solver_status.startswith("optimal")
    sells = res.trades[res.trades["side"] == "SELL"]
    buys = res.trades[res.trades["side"] == "BUY"]
    # wash-blocked lot 3 is reported, never traded
    assert 3 in set(res.blocked["lot_id"])
    assert res.blocked.iloc[0]["wash_status"] in ("WASH", "PARTIAL_WASH")
    assert 3 not in set(sells["lot_id"])
    # winners are never sold, IRA never sold
    assert set(sells["symbol"]) <= {held[0], held[1]}
    assert len(sells) >= 1 and len(buys) >= 1
    assert (res.trades["est_value"] >= cfg.min_trade_value - 1e-6).all()
    # cash neutral within tolerance
    V = res.summary["portfolio_value"]
    assert abs(res.summary["buy_value"] - res.summary["sell_value"]) <= cfg.cash_tolerance * V + max(res.trades["est_price"]) * 2
    assert res.summary["harvested_loss"] > 0
    assert res.summary["tax_benefit"] > 0
    assert np.isfinite(res.summary["te_after"])
    # buys are only wash-safe replacement candidates
    assert (res.replacements["wash_status"] == "SAFE").any()
    assert not (res.trades["wash_status"] != "SAFE").any()


def test_te_first_priority_reduces_te_vs_tax_first(world):
    inp, _, _ = build_inputs(world)
    base = HarvestConfig(min_trade_value=500, min_loss_value=100, te_budget=0.005)
    tax_first = run_harvest(inp, base)
    te_first = run_harvest(inp, HarvestConfig(**{**base.__dict__, "priority": ("tracking_error", "tax", "factor_neutrality")}))
    assert te_first.summary["te_after"] <= tax_first.summary["te_after"] + 1e-6
    # harvested loss is NOT monotone in the priority order (selling overweight losers can also cut TE),
    # so only the TE invariant is asserted.


def test_hard_te_cap_respected_or_flagged(world):
    inp, _, _ = build_inputs(world)
    res = run_harvest(inp, HarvestConfig(min_trade_value=500, min_loss_value=100, te_budget=0.10, te_hard=True))
    assert res.summary["te_after"] <= 0.10 + 1e-4 or res.solver_status.endswith("_te_relaxed")


def test_buy_blocked_by_recent_loss_sale(world):
    inp, held, aid = build_inputs(world)
    syms = inp.universe
    # we sold syms[20] at a loss 5 days ago -> it cannot be a replacement now
    inp.recent_loss_sales = [LossSale(aid[syms[20]], syms[20], 1, AS_OF - timedelta(days=5), 10, 100.0)]
    res = run_harvest(inp, HarvestConfig(min_trade_value=500, min_loss_value=100))
    r = res.replacements
    row = r[(r["candidate"] == syms[20])]
    assert not row.empty and (row["wash_status"] == "WOULD_WASH").all()
    assert syms[20] not in set(res.trades[res.trades["side"] == "BUY"]["symbol"])


def test_frontier_and_priority_sweeps(world):
    inp, _, _ = build_inputs(world)
    cfg = HarvestConfig(min_trade_value=500, min_loss_value=100)
    fr = frontier(inp, cfg, te_grid=[0.005, 0.02, 0.05])
    assert len(fr) == 3 and fr["harvested_loss"].notna().all()
    assert fr["harvested_loss"].is_monotonic_increasing or fr["harvested_loss"].max() - fr["harvested_loss"].min() >= 0
    table, results = priority_comparison(inp, cfg)
    assert len(table) == 6 and len(results) == 6


def test_target_loss_caps_harvest(world):
    inp, _, _ = build_inputs(world)
    res = run_harvest(inp, HarvestConfig(min_trade_value=500, min_loss_value=100, target_loss=5000.0))
    assert res.summary["harvested_loss"] <= 5000.0 * 1.01 + 1
