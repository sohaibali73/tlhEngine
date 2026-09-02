from datetime import date

import pytest

from tlh.tax.lots import Lot, LotMethod, aggregate_position, select_lots
from tlh.tax.rates import TaxProfile


def mk(id_, acq, qty, cost, adj=0.0):
    return Lot(id=id_, account_id=1, assetid=100, symbol="XYZ", acquired_date=acq, holding_start_date=acq,
               quantity_original=qty, quantity_open=qty, cost_per_share=cost, basis_adjustment=adj)


@pytest.fixture
def lots():
    return [
        mk(1, date(2024, 1, 10), 100, 50.0),
        mk(2, date(2025, 3, 1), 100, 80.0),
        mk(3, date(2025, 8, 15), 50, 65.0),
    ]


def test_fifo(lots):
    s = select_lots(lots, 150, LotMethod.FIFO)
    assert [(x.lot.id, x.quantity) for x in s] == [(1, 100), (2, 50)]


def test_lifo(lots):
    s = select_lots(lots, 120, LotMethod.LIFO)
    assert [(x.lot.id, x.quantity) for x in s] == [(3, 50), (2, 70)]


def test_hifo(lots):
    s = select_lots(lots, 120, LotMethod.HIFO)
    assert [(x.lot.id, x.quantity) for x in s] == [(2, 100), (3, 20)]


def test_specific(lots):
    s = select_lots(lots, 120, LotMethod.SPECIFIC, specific_ids=[3, 1])
    assert [(x.lot.id, x.quantity) for x in s] == [(3, 50), (1, 70)]
    with pytest.raises(ValueError):
        select_lots(lots, 60, LotMethod.SPECIFIC, specific_ids=[3])


def test_max_loss_prefers_short_term_losses(lots):
    # price 60: lot1 gain, lot2 loss (ST at 2025-09-01), lot3 loss (ST)
    s = select_lots(lots, 150, LotMethod.MAX_LOSS, price=60.0, as_of=date(2025, 9, 1))
    assert [(x.lot.id, x.quantity) for x in s] == [(2, 100), (3, 50)]
    # at 2026-04-01 lot2 is LT, lot3 still ST -> lot3 first
    s = select_lots(lots, 60, LotMethod.MAX_LOSS, price=60.0, as_of=date(2026, 4, 1))
    assert [(x.lot.id, x.quantity) for x in s] == [(3, 50), (2, 10)]


def test_insufficient_raises(lots):
    with pytest.raises(ValueError):
        select_lots(lots, 251, LotMethod.FIFO)


def test_basis_adjustment_spreads_per_original_share():
    lot = mk(9, date(2025, 1, 1), 100, 10.0, adj=200.0)
    assert lot.basis_per_share == 12.0
    lot.quantity_open = 40
    assert lot.open_basis == pytest.approx(480.0)
    assert lot.basis_for(10) == pytest.approx(120.0)


def test_aggregate_position(lots):
    agg = aggregate_position(lots, price=60.0, as_of=date(2025, 9, 1))
    assert agg["quantity"] == 250
    assert agg["cost_basis"] == pytest.approx(100 * 50 + 100 * 80 + 50 * 65)
    assert agg["unrealized_gain"] == pytest.approx(250 * 60 - agg["cost_basis"])
    assert agg["harvestable_loss"] == pytest.approx(100 * 20 + 50 * 5)
    assert agg["unrealized_lt"] == pytest.approx(100 * 10)
    assert agg["unrealized_st"] == pytest.approx(-2000 - 250)


def test_tax_profile_rates():
    p = TaxProfile(fed_st_rate=0.37, fed_lt_rate=0.20, state_rate=0.05, niit_rate=0.038)
    assert p.st_rate == pytest.approx(0.458)
    assert p.lt_rate == pytest.approx(0.288)
    assert p.benefit_of_loss(1000, "ST") == pytest.approx(458.0)
    assert p.benefit_of_loss(1000, "ST", offsets_term="LT") == pytest.approx(288.0)
    # tax alpha with step-up at death: full benefit
    assert p.tax_alpha(1000, "ST", horizon_years=float("inf")) == pytest.approx(458.0)
    # with 10y horizon the deferred LT tax is discounted
    ta = p.tax_alpha(1000, "ST", horizon_years=10, discount_rate=0.04)
    assert ta == pytest.approx(458.0 - 288.0 * 1.04 ** -10)
    assert TaxProfile(filing_status="mfs").effective_ordinary_offset == 1500
