from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tlh.tax.holding import days_until_long_term, first_long_term_date, is_long_term, term_for


def test_anniversary_is_still_short_term():
    acq = date(2025, 1, 5)
    assert term_for(acq, date(2026, 1, 5)) == "ST"
    assert term_for(acq, date(2026, 1, 6)) == "LT"
    assert first_long_term_date(acq) == date(2026, 1, 6)


def test_day_before_anniversary_short_term():
    assert term_for(date(2025, 6, 30), date(2026, 6, 29)) == "ST"
    assert term_for(date(2025, 6, 30), date(2026, 6, 30)) == "ST"
    assert term_for(date(2025, 6, 30), date(2026, 7, 1)) == "LT"


def test_leap_day_acquisition():
    acq = date(2024, 2, 29)
    # anniversary in a non-leap year is Feb 28 2025; first LT day is Mar 1 2025
    assert first_long_term_date(acq) == date(2025, 3, 1)
    assert term_for(acq, date(2025, 2, 28)) == "ST"
    assert term_for(acq, date(2025, 3, 1)) == "LT"


def test_acquired_feb_28_before_leap_year():
    acq = date(2023, 2, 28)
    assert first_long_term_date(acq) == date(2024, 2, 29)
    assert term_for(acq, date(2024, 2, 28)) == "ST"
    assert term_for(acq, date(2024, 2, 29)) == "LT"


def test_sale_before_acquisition_raises():
    with pytest.raises(ValueError):
        is_long_term(date(2025, 1, 5), date(2025, 1, 4))


def test_days_until_long_term():
    acq = date(2025, 1, 5)
    assert days_until_long_term(acq, date(2025, 1, 5)) == 366
    assert days_until_long_term(acq, date(2026, 1, 5)) == 1
    assert days_until_long_term(acq, date(2026, 1, 6)) == 0
    assert days_until_long_term(acq, date(2027, 1, 1)) == 0


@given(st.dates(min_value=date(1990, 1, 1), max_value=date(2040, 1, 1)),
       st.integers(min_value=0, max_value=800))
def test_property_monotone_and_threshold(acq, offset):
    from datetime import timedelta
    sale = acq + timedelta(days=offset)
    lt = is_long_term(acq, sale)
    # held > 1 year means at least 366 days later (367 across a Feb 29)
    if offset <= 365:
        assert not lt
    if offset >= 367:
        assert lt
