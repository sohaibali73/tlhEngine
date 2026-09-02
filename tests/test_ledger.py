from datetime import date

import pytest

from tlh.tax.ledger import LotBook, net_capital_position
from tlh.tax.lots import LotMethod
from tlh.tax.washsale import Acquisition, SubstantiallyIdentical

D = date


def test_simple_gain_and_loss_terms():
    book = LotBook()
    book.record_purchase(1, 100, "AAA", D(2024, 1, 10), 100, 50.0)
    book.record_purchase(1, 100, "AAA", D(2025, 6, 1), 100, 80.0)
    cs = book.record_sale(1, 100, D(2025, 9, 1), 150, 60.0, method=LotMethod.HIFO)
    # HIFO: 100 @80 (ST loss -2000), then 50 @50 (LT gain +500); closes oldest first
    by_lot = {c.lot.cost_per_share: c for c in cs}
    assert by_lot[80.0].realized_gain == pytest.approx(-2000) and by_lot[80.0].term == "ST"
    assert by_lot[50.0].realized_gain == pytest.approx(500) and by_lot[50.0].term == "LT"
    s = book.realized_summary(2025)
    assert s["net_st"] == pytest.approx(-2000) and s["net_lt"] == pytest.approx(500)


def test_fees_raise_basis_and_lower_proceeds():
    book = LotBook()
    lot, _ = book.record_purchase(1, 100, "AAA", D(2025, 1, 1), 10, 100.0, fees=5.0)
    assert lot.cost_per_share == pytest.approx(100.5)
    cs = book.record_sale(1, 100, D(2025, 2, 1), 10, 110.0, fees=5.0)
    assert cs[0].proceeds == pytest.approx(1095.0)
    assert cs[0].realized_gain == pytest.approx(1095 - 1005)


def test_wash_on_sale_with_prior_purchase_adjusts_basis_and_tacks_holding():
    book = LotBook()
    old, _ = book.record_purchase(1, 100, "AAA", D(2024, 1, 10), 100, 100.0)
    new, _ = book.record_purchase(1, 100, "AAA", D(2025, 3, 1), 100, 70.0)
    cs = book.record_sale(1, 100, D(2025, 3, 15), 100, 60.0, method=LotMethod.SPECIFIC, specific_ids=[old.id])
    c = cs[0]
    assert c.realized_gain == pytest.approx(-4000)
    assert c.wash_disallowed == pytest.approx(4000)
    assert c.allowed_gain == pytest.approx(0)
    assert new.basis_adjustment == pytest.approx(4000)
    assert new.basis_per_share == pytest.approx(110.0)
    assert new.holding_start_date == D(2024, 1, 10)      # tacked
    assert new.term_at(D(2025, 3, 16)) == "LT"


def test_retroactive_wash_when_buying_after_loss_sale():
    book = LotBook()
    book.record_purchase(1, 100, "AAA", D(2024, 1, 10), 100, 100.0)
    cs = book.record_sale(1, 100, D(2025, 3, 15), 100, 60.0)
    assert cs[0].wash_disallowed == 0
    new, touched = book.record_purchase(1, 100, "AAA", D(2025, 4, 10), 40, 65.0)
    assert touched == [cs[0]]
    assert cs[0].wash_disallowed == pytest.approx(1600)     # 40 sh x $40 loss/sh
    assert new.basis_adjustment == pytest.approx(1600)
    assert new.holding_start_date == D(2024, 1, 10)
    # a further purchase 31 days later is clean
    _, touched2 = book.record_purchase(1, 100, "AAA", D(2025, 4, 15), 100, 65.0)
    assert touched2 == []
    assert cs[0].wash_disallowed == pytest.approx(1600)


def test_ira_purchase_permanently_disallows_without_basis_step_up():
    book = LotBook()
    book.record_purchase(1, 100, "AAA", D(2024, 1, 10), 100, 100.0)
    cs = book.record_sale(1, 100, D(2025, 3, 15), 100, 60.0)
    ira_lot, touched = book.record_purchase(2, 100, "AAA", D(2025, 3, 20), 100, 61.0, account_type="ira")
    assert cs[0].wash_disallowed == pytest.approx(4000)
    assert ira_lot.basis_adjustment == 0
    assert "no basis step-up" in cs[0].wash_explanation


def test_substantially_identical_etf_pair_via_groups():
    groups = SubstantiallyIdentical({1: "spx", 2: "spx"})
    book = LotBook(groups=groups)
    book.record_purchase(1, 1, "SPY", D(2024, 1, 10), 100, 500.0)
    cs = book.record_sale(1, 1, D(2025, 3, 15), 100, 450.0)
    ivv, touched = book.record_purchase(1, 2, "IVV", D(2025, 3, 16), 100, 452.0)
    assert touched and cs[0].wash_disallowed == pytest.approx(5000)
    assert ivv.basis_adjustment == pytest.approx(5000)


def test_same_day_buy_and_sell_all_preserves_total_loss():
    """Buy B (young), then sell A (old) and B together at a loss: A's loss washes into B, and B's closure
    realises it, so the total recognised loss equals the economic loss."""
    book = LotBook()
    book.record_purchase(1, 100, "AAA", D(2024, 1, 10), 100, 100.0)
    book.record_purchase(1, 100, "AAA", D(2025, 3, 1), 100, 70.0)
    cs = book.record_sale(1, 100, D(2025, 3, 15), 200, 60.0, method=LotMethod.FIFO)
    econ = 200 * 60 - (100 * 100 + 100 * 70)
    assert sum(c.allowed_gain for c in cs) == pytest.approx(econ)
    assert sum(c.wash_disallowed for c in cs) == pytest.approx(4000)


def test_scheduled_events_visible_to_acquisitions():
    book = LotBook()
    book.scheduled.append(Acquisition(100, "AAA", 1, "taxable", D(2025, 4, 1), 5, kind="scheduled_drip"))
    assert any(a.is_scheduled for a in book.acquisitions())
    assert not any(a.is_scheduled for a in book.acquisitions(include_scheduled=False))


# ------------------------------------------------------------------ Schedule D netting
def test_netting_both_gains():
    r = net_capital_position(1000, 2000)
    assert (r.taxable_st_gain, r.taxable_lt_gain, r.ordinary_deduction) == (1000, 2000, 0)
    assert r.carryforward_st == 0 and r.carryforward_lt == 0


def test_netting_both_losses_st_used_first_for_ordinary_offset():
    r = net_capital_position(-2000, -5000)
    assert r.ordinary_deduction == 3000
    assert r.carryforward_st == 0            # 2000 ST fully used
    assert r.carryforward_lt == 4000         # 1000 of LT used, 4000 carried


def test_netting_cross_character():
    r = net_capital_position(4000, -1000)     # ST gain absorbs LT loss
    assert r.taxable_st_gain == 3000 and r.taxable_lt_gain == 0 and r.ordinary_deduction == 0
    r = net_capital_position(-8000, 2000)     # ST loss > LT gain -> 6000 ST loss remains
    assert r.taxable_lt_gain == 0 and r.ordinary_deduction == 3000
    assert r.carryforward_st == 3000 and r.carryforward_lt == 0
    r = net_capital_position(2000, -8000)     # LT loss remains, keeps LT character
    assert r.ordinary_deduction == 3000 and r.carryforward_lt == 3000 and r.carryforward_st == 0


def test_netting_with_prior_carryforward():
    r = net_capital_position(500, 0, prior_cf_st=4000)
    assert r.net_st == -3500
    assert r.ordinary_deduction == 3000 and r.carryforward_st == 500


def test_netting_mfs_offset():
    r = net_capital_position(-5000, 0, ordinary_offset=1500)
    assert r.ordinary_deduction == 1500 and r.carryforward_st == 3500
