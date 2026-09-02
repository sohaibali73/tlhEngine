"""Wash-sale engine tests. Expected values are hand-computed from IRS Pub 550 examples and §1.1091-1."""
from datetime import date, timedelta

import pytest

from tlh.tax.washsale import (
    Acquisition,
    LossSale,
    SubstantiallyIdentical,
    evaluate_loss_sale,
    in_window,
    repurchase_allowed_from,
    screen_proposed_buy,
    screen_proposed_sale,
    window_for,
)

D = date


def acq(assetid, d, qty, account=1, acct_type="taxable", lot_id=None, kind="buy", symbol=None):
    return Acquisition(assetid=assetid, symbol=symbol or f"S{assetid}", account_id=account, account_type=acct_type,
                       acquired_date=d, quantity=qty, lot_id=lot_id, kind=kind)


def sale(assetid, d, qty, loss, account=1, lot_id=None):
    return LossSale(assetid=assetid, symbol=f"S{assetid}", account_id=account, sale_date=d, quantity=qty,
                    loss_amount=loss, lot_id=lot_id)


# ------------------------------------------------------------------ window arithmetic
def test_window_is_61_days_inclusive():
    s, e = window_for(D(2025, 3, 15))
    assert s == D(2025, 2, 13) and e == D(2025, 4, 14)
    assert (e - s).days + 1 == 61
    assert in_window(D(2025, 2, 13), D(2025, 3, 15))
    assert not in_window(D(2025, 2, 12), D(2025, 3, 15))
    assert in_window(D(2025, 4, 14), D(2025, 3, 15))
    assert not in_window(D(2025, 4, 15), D(2025, 3, 15))
    assert repurchase_allowed_from(D(2025, 3, 15)) == D(2025, 4, 15)


# ------------------------------------------------------------------ core determinations
def test_no_acquisitions_is_safe():
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 1000), [])
    assert det.status == "SAFE" and det.disallowed_loss == 0 and "SAFE" in det.explanation


def test_pub550_partial_example():
    """Pub 550: 100 sh bought Sep 24 for 5,000; 50 bought Dec 19; 25 bought Dec 26; sold the 100 Jan 8 for
    4,000 -> $1,000 loss, 75 replacement shares -> $750 disallowed, $250 allowed."""
    acqs = [acq(1, D(2023, 9, 24), 100, lot_id=1), acq(1, D(2023, 12, 19), 50, lot_id=2),
            acq(1, D(2023, 12, 26), 25, lot_id=3)]
    det = evaluate_loss_sale(sale(1, D(2024, 1, 8), 100, 1000, lot_id=1), acqs)
    assert det.status == "PARTIAL_WASH"
    assert det.disallowed_quantity == 75
    assert det.disallowed_loss == pytest.approx(750)
    assert det.allowed_loss == pytest.approx(250)
    assert [m.acquisition.lot_id for m in det.matches] == [2, 3]  # chronological


def test_full_wash_when_replacement_exceeds_sold():
    acqs = [acq(1, D(2025, 1, 1), 100, lot_id=1), acq(1, D(2025, 3, 20), 150, lot_id=2)]
    det = evaluate_loss_sale(sale(1, D(2025, 3, 10), 100, 500, lot_id=1), acqs)
    assert det.status == "WASH"
    assert det.disallowed_loss == pytest.approx(500)
    assert acqs[1].used_as_replacement == 100  # only 100 of the 150 consumed


def test_sold_lot_is_never_its_own_replacement():
    # bought 10 days ago, sold today at a loss, nothing else -> not a wash sale
    acqs = [acq(1, D(2025, 3, 5), 100, lot_id=7)]
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 300, lot_id=7), acqs)
    assert det.status == "SAFE"


def test_purchase_outside_window_ignored():
    acqs = [acq(1, D(2025, 2, 12), 100, lot_id=2)]  # 31 days before
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 300, lot_id=1), acqs)
    assert det.status == "SAFE"
    acqs = [acq(1, D(2025, 4, 15), 100, lot_id=2)]  # 31 days after
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 300, lot_id=1), acqs)
    assert det.status == "SAFE"
    acqs = [acq(1, D(2025, 4, 14), 100, lot_id=2)]  # 30 days after -> wash
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 300, lot_id=1), acqs)
    assert det.status == "WASH"


def test_cross_account_and_ira_permanent():
    acqs = [acq(1, D(2025, 3, 20), 100, account=2, acct_type="ira", lot_id=5)]
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 800, account=1, lot_id=1), acqs)
    assert det.status == "WASH"
    assert det.matches[0].permanent is True
    assert "PERMANENTLY" in det.explanation


def test_spouse_account_same_entity_counts():
    acqs = [acq(1, D(2025, 3, 1), 40, account=9, lot_id=5)]
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 1000, account=1, lot_id=1), acqs)
    assert det.status == "PARTIAL_WASH" and det.disallowed_loss == pytest.approx(400)


def test_substantially_identical_group_share_classes():
    groups = SubstantiallyIdentical({1: "alphabet", 2: "alphabet"})
    acqs = [acq(2, D(2025, 3, 20), 100, lot_id=5)]
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 800, lot_id=1), acqs, groups)
    assert det.status == "WASH"
    # different group -> safe
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 800, lot_id=1), acqs, SubstantiallyIdentical({}))
    assert det.status == "SAFE"


def test_replacement_shares_consumed_once_across_sales():
    acqs = [acq(1, D(2025, 3, 20), 100, lot_id=5)]
    d1 = evaluate_loss_sale(sale(1, D(2025, 3, 10), 60, 600, lot_id=1), acqs)
    d2 = evaluate_loss_sale(sale(1, D(2025, 3, 12), 60, 600, lot_id=2), acqs)
    assert d1.disallowed_quantity == 60
    assert d2.disallowed_quantity == 40
    assert d2.status == "PARTIAL_WASH" and d2.allowed_loss == pytest.approx(200)


def test_replacement_disposed_before_loss_sale_not_counted():
    a = acq(1, D(2025, 3, 1), 100, lot_id=2)
    a.disposed_date = D(2025, 3, 10)
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 300, lot_id=1), [a])
    assert det.status == "SAFE"
    a.disposed_date = D(2025, 3, 15)  # same-day disposal still counts
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 300, lot_id=1), [a])
    assert det.status == "WASH"


def test_scheduled_drip_blocks_forward():
    acqs = [acq(1, D(2025, 3, 30), 3, lot_id=None, kind="scheduled_drip")]
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 1000, lot_id=1), acqs)
    assert det.status == "PARTIAL_WASH" and det.has_forward_conflict
    assert det.disallowed_loss == pytest.approx(30)
    assert "FUTURE" in det.explanation
    det2 = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 1000, lot_id=1), acqs, include_scheduled=False)
    assert det2.status == "SAFE"


def test_scheduled_full_cover_is_blocked_forward():
    acqs = [acq(1, D(2025, 4, 1), 200, kind="scheduled_buy")]
    det = evaluate_loss_sale(sale(1, D(2025, 3, 15), 100, 1000, lot_id=1), acqs)
    assert det.status == "BLOCKED_FORWARD"


def test_screen_proposed_sale_does_not_mutate():
    acqs = [acq(1, D(2025, 3, 20), 100, lot_id=5)]
    det = screen_proposed_sale(1, "S1", 1, D(2025, 3, 15), 100, 500, acqs, lot_id=1)
    assert det.status == "WASH"
    assert acqs[0].used_as_replacement == 0.0


# ------------------------------------------------------------------ buy-side screen
def test_screen_proposed_buy():
    sales = [sale(1, D(2025, 3, 15), 100, 500)]
    groups = SubstantiallyIdentical({1: "spx", 2: "spx", 3: "russell"})
    assert screen_proposed_buy(2, "IVV", D(2025, 3, 16), sales, groups).status == "WOULD_WASH"
    assert screen_proposed_buy(3, "IWB", D(2025, 3, 16), sales, groups).status == "SAFE"
    assert screen_proposed_buy(2, "IVV", D(2025, 4, 14), sales, groups).status == "WOULD_WASH"
    ok = screen_proposed_buy(2, "IVV", D(2025, 4, 15), sales, groups)
    assert ok.status == "SAFE"
    bad = screen_proposed_buy(2, "IVV", D(2025, 3, 16), sales, groups)
    assert "2025-04-15" in bad.explanation


@pytest.mark.parametrize("offset", range(-35, 36))
def test_every_day_in_window_and_out(offset):
    d = D(2025, 6, 10)
    acqs = [acq(1, d + timedelta(days=offset), 100, lot_id=2)]
    det = evaluate_loss_sale(sale(1, d, 100, 100, lot_id=1), acqs)
    assert (det.status == "WASH") == (abs(offset) <= 30)
