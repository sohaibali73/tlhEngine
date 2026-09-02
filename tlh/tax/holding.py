"""Holding-period rules (IRC §1222).

Convention (DECISIONS.md D8): the holding period begins the day AFTER acquisition and includes the day of
sale. A gain/loss is long-term only if the property was held MORE than one year, i.e. the sale date is
strictly after the first anniversary of acquisition. Sale on the anniversary itself is short-term.

Leap days: `dateutil.relativedelta` maps Feb 29 -> Feb 28 of the following year, so the first long-term
sale date for a Feb 29 acquisition is Mar 1 of the next year. This matches the IRS treatment that the
anniversary of Feb 29 in a non-leap year is Feb 28.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from dateutil.relativedelta import relativedelta

Term = Literal["ST", "LT"]

ST: Term = "ST"
LT: Term = "LT"


def first_long_term_date(holding_start: date) -> date:
    """First calendar date on which a sale of a lot acquired on `holding_start` is long-term."""
    return holding_start + relativedelta(years=1) + timedelta(days=1)


def is_long_term(holding_start: date, sale_date: date) -> bool:
    if sale_date < holding_start:
        raise ValueError(f"sale_date {sale_date} precedes holding_start {holding_start}")
    return sale_date >= first_long_term_date(holding_start)


def term_for(holding_start: date, sale_date: date) -> Term:
    return LT if is_long_term(holding_start, sale_date) else ST


def days_until_long_term(holding_start: date, as_of: date) -> int:
    """Calendar days from `as_of` until the lot becomes long-term (0 if already long-term)."""
    d = (first_long_term_date(holding_start) - as_of).days
    return max(d, 0)


def holding_days(holding_start: date, as_of: date) -> int:
    """Days held under the IRS convention (acquisition day excluded, `as_of` included)."""
    return (as_of - holding_start).days
