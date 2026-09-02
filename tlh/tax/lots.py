"""Tax lots and lot-selection methods.

A `Lot` is an open (or partially open) parcel of shares with its own cost basis and holding period. Wash-sale
basis adjustments are stored as a lot-level dollar amount and spread evenly per original share, so partial
closures carry their pro-rata share of the adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from .holding import Term, days_until_long_term, term_for


class LotMethod(StrEnum):
    FIFO = "FIFO"
    LIFO = "LIFO"
    HIFO = "HIFO"          # highest basis per share first (maximises loss / minimises gain)
    SPECIFIC = "SPECIFIC"  # caller supplies lot ids in order
    MAX_LOSS = "MAX_LOSS"  # HIFO restricted to loss lots, short-term losses first (harvest default)


@dataclass
class Lot:
    id: int | None
    account_id: int
    assetid: int
    symbol: str
    acquired_date: date
    holding_start_date: date
    quantity_original: float
    quantity_open: float
    cost_per_share: float
    basis_adjustment: float = 0.0
    source: str = "buy"
    account_type: str = "taxable"
    entity_id: int | None = None
    is_closed: bool = False
    notes: str | None = None
    extra: dict = field(default_factory=dict)

    # --------------------------------------------------------------------- basis
    @property
    def adjustment_per_share(self) -> float:
        return self.basis_adjustment / self.quantity_original if self.quantity_original else 0.0

    @property
    def basis_per_share(self) -> float:
        return self.cost_per_share + self.adjustment_per_share

    @property
    def open_basis(self) -> float:
        return self.basis_per_share * self.quantity_open

    def basis_for(self, quantity: float) -> float:
        if quantity < 0 or quantity > self.quantity_open + 1e-9:
            raise ValueError(f"quantity {quantity} exceeds open quantity {self.quantity_open}")
        return self.basis_per_share * quantity

    # --------------------------------------------------------------------- valuation
    def market_value(self, price: float) -> float:
        return price * self.quantity_open

    def unrealized_gain(self, price: float) -> float:
        return (price - self.basis_per_share) * self.quantity_open

    def unrealized_gain_pct(self, price: float) -> float:
        return price / self.basis_per_share - 1.0 if self.basis_per_share else 0.0

    def term_at(self, as_of: date) -> Term:
        return term_for(self.holding_start_date, as_of)

    def days_to_long_term(self, as_of: date) -> int:
        return days_until_long_term(self.holding_start_date, as_of)

    def is_taxable(self) -> bool:
        return self.account_type == "taxable"


@dataclass(frozen=True)
class LotSlice:
    lot: Lot
    quantity: float

    @property
    def basis(self) -> float:
        return self.lot.basis_for(self.quantity)


def select_lots(
    lots: list[Lot],
    quantity: float,
    method: LotMethod,
    price: float | None = None,
    as_of: date | None = None,
    specific_ids: list[int] | None = None,
) -> list[LotSlice]:
    """Choose which lots (and how many shares of each) satisfy a sale of `quantity` shares.

    Raises ValueError if open quantity is insufficient. `price` is required for MAX_LOSS.
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    open_lots = [lot for lot in lots if lot.quantity_open > 1e-12 and not lot.is_closed]
    total = sum(lot.quantity_open for lot in open_lots)
    if quantity > total + 1e-9:
        raise ValueError(f"insufficient shares: need {quantity}, have {total}")

    ordered = order_lots(open_lots, method, price=price, as_of=as_of, specific_ids=specific_ids)
    if method == LotMethod.SPECIFIC:
        avail = sum(lot.quantity_open for lot in ordered)
        if quantity > avail + 1e-9:
            raise ValueError(f"specified lots hold {avail} shares; need {quantity}")

    out: list[LotSlice] = []
    remaining = quantity
    for lot in ordered:
        if remaining <= 1e-12:
            break
        take = min(lot.quantity_open, remaining)
        out.append(LotSlice(lot, take))
        remaining -= take
    return out


def order_lots(
    lots: list[Lot],
    method: LotMethod,
    price: float | None = None,
    as_of: date | None = None,
    specific_ids: list[int] | None = None,
) -> list[Lot]:
    if method == LotMethod.FIFO:
        return sorted(lots, key=lambda lot: (lot.holding_start_date, lot.id or 0))
    if method == LotMethod.LIFO:
        return sorted(lots, key=lambda lot: (lot.holding_start_date, lot.id or 0), reverse=True)
    if method == LotMethod.HIFO:
        return sorted(lots, key=lambda lot: (-lot.basis_per_share, lot.holding_start_date))
    if method == LotMethod.SPECIFIC:
        if not specific_ids:
            raise ValueError("SPECIFIC requires specific_ids")
        by_id = {lot.id: lot for lot in lots}
        missing = [i for i in specific_ids if i not in by_id]
        if missing:
            raise ValueError(f"unknown or closed lot ids: {missing}")
        return [by_id[i] for i in specific_ids]
    if method == LotMethod.MAX_LOSS:
        if price is None or as_of is None:
            raise ValueError("MAX_LOSS requires price and as_of")
        losers = [lot for lot in lots if lot.unrealized_gain(price) < 0]
        # short-term losses first (worth more), then biggest per-share loss
        return sorted(losers, key=lambda lot: (0 if lot.term_at(as_of) == "ST" else 1, -lot.basis_per_share))
    raise ValueError(f"unknown method {method}")


def aggregate_position(lots: list[Lot], price: float, as_of: date) -> dict:
    """Position-level summary across lots of one (account, asset)."""
    open_lots = [lot for lot in lots if lot.quantity_open > 1e-12]
    qty = sum(lot.quantity_open for lot in open_lots)
    basis = sum(lot.open_basis for lot in open_lots)
    mv = qty * price
    st_gain = sum(lot.unrealized_gain(price) for lot in open_lots if lot.term_at(as_of) == "ST")
    lt_gain = sum(lot.unrealized_gain(price) for lot in open_lots if lot.term_at(as_of) == "LT")
    return {
        "quantity": qty,
        "cost_basis": basis,
        "market_value": mv,
        "unrealized_gain": mv - basis,
        "unrealized_st": st_gain,
        "unrealized_lt": lt_gain,
        "harvestable_loss": -sum(min(lot.unrealized_gain(price), 0.0) for lot in open_lots),
        "n_lots": len(open_lots),
    }
