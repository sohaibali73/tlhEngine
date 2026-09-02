"""Realized / unrealized gain-loss ledger and capital-loss carryforward netting.

`LotBook` is an in-memory book of lots + closures for one tax entity that knows how to record purchases and
sales while applying wash-sale rules in both directions:

* recording a loss SALE looks back 30 days for replacement purchases (and forward at scheduled events);
* recording a PURCHASE looks back 30 days for loss sales that it retroactively turns into wash sales.

The DB repository (`db/repos.py`) hydrates a LotBook from SQLite and writes the resulting lots/closures back.
Netting follows Schedule D / §1211-1212.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .holding import LT, ST, Term, term_for
from .lots import Lot, LotMethod, LotSlice, select_lots
from .washsale import (
    Acquisition,
    LossSale,
    SubstantiallyIdentical,
    WashSaleDetermination,
    evaluate_loss_sale,
)


@dataclass
class Closure:
    lot: Lot
    sale_date: date
    quantity: float
    proceeds: float
    cost_basis: float
    term: Term
    wash_disallowed: float = 0.0
    wash_replacement_lot: Lot | None = None
    wash_explanation: str = ""
    wash_matched_quantity: float = 0.0
    determination: WashSaleDetermination | None = None
    id: int | None = None
    sell_tx_id: int | None = None

    def wash_matched_qty(self) -> float:
        return self.wash_matched_quantity

    @property
    def realized_gain(self) -> float:
        return self.proceeds - self.cost_basis

    @property
    def allowed_gain(self) -> float:
        """Recognised gain/loss after wash-sale disallowance."""
        return self.realized_gain + self.wash_disallowed


@dataclass
class LotBook:
    groups: SubstantiallyIdentical = field(default_factory=SubstantiallyIdentical)
    lots: list[Lot] = field(default_factory=list)
    closures: list[Closure] = field(default_factory=list)
    scheduled: list[Acquisition] = field(default_factory=list)
    _next_lot_id: int = -1

    # --------------------------------------------------------------------- queries
    def open_lots(self, account_id: int | None = None, assetid: int | None = None) -> list[Lot]:
        return [
            lot for lot in self.lots
            if lot.quantity_open > 1e-12
            and (account_id is None or lot.account_id == account_id)
            and (assetid is None or lot.assetid == assetid)
        ]

    def acquisitions(self, include_scheduled: bool = True) -> list[Acquisition]:
        """All share-adding events in the entity, expressed for the wash-sale engine."""
        acqs = [
            Acquisition(
                assetid=lot.assetid, symbol=lot.symbol, account_id=lot.account_id,
                account_type=lot.account_type, acquired_date=lot.acquired_date,
                quantity=lot.quantity_original, lot_id=lot.id, kind=lot.source,
                used_as_replacement=float(lot.extra.get("used_as_replacement", 0.0)),
                disposed_date=self._disposed_date(lot),
            )
            for lot in self.lots
        ]
        if include_scheduled:
            acqs.extend(self.scheduled)
        return acqs

    def _disposed_date(self, lot: Lot) -> date | None:
        if lot.quantity_open > 1e-12:
            return None
        dates = [c.sale_date for c in self.closures if c.lot is lot]
        return max(dates) if dates else None

    def loss_sales(self, since: date | None = None) -> list[LossSale]:
        out = []
        for c in self.closures:
            if c.realized_gain < 0 and (since is None or c.sale_date >= since):
                out.append(LossSale(c.lot.assetid, c.lot.symbol, c.lot.account_id, c.sale_date,
                                    c.quantity, -c.realized_gain, lot_id=c.lot.id,
                                    lot_holding_start=c.lot.holding_start_date))
        return out

    # --------------------------------------------------------------------- mutations
    def add_lot(self, lot: Lot) -> Lot:
        if lot.id is None:
            lot.id = self._next_lot_id
            self._next_lot_id -= 1
        self.lots.append(lot)
        return lot

    def record_purchase(
        self, account_id: int, assetid: int, symbol: str, trade_date: date, quantity: float,
        price: float, fees: float = 0.0, account_type: str = "taxable", source: str = "buy",
        entity_id: int | None = None,
    ) -> tuple[Lot, list[Closure]]:
        """Open a new lot; then check whether it retroactively washes a loss sale in the prior 30 days.

        Returns the new lot and the closures whose wash status changed.
        """
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        cost_ps = price + (fees / quantity if quantity else 0.0)
        lot = self.add_lot(Lot(
            id=None, account_id=account_id, assetid=assetid, symbol=symbol, acquired_date=trade_date,
            holding_start_date=trade_date, quantity_original=quantity, quantity_open=quantity,
            cost_per_share=cost_ps, source=source, account_type=account_type, entity_id=entity_id,
        ))
        touched = self._apply_retroactive_wash(lot)
        return lot, touched

    def _apply_retroactive_wash(self, new_lot: Lot) -> list[Closure]:
        """A new purchase may turn prior-30-day loss sales in the same group into wash sales."""
        group = self.groups.group_of(new_lot.assetid)
        touched: list[Closure] = []
        prior = sorted(
            [c for c in self.closures
             if c.realized_gain < 0
             and self.groups.group_of(c.lot.assetid) == group
             and 0 <= (new_lot.acquired_date - c.sale_date).days <= 30
             and c.quantity - c.wash_matched_qty() > 1e-12],
            key=lambda c: c.sale_date,
        )
        available = new_lot.quantity_original
        for c in prior:
            if available <= 1e-12:
                break
            unmatched_qty = c.quantity - c.wash_matched_qty()
            take = min(available, unmatched_qty)
            loss_ps = -c.realized_gain / c.quantity
            disallowed = take * loss_ps
            self._attach_wash(c, new_lot, take, disallowed,
                              note=f"Retroactive: {take:g} sh of {new_lot.symbol} bought {new_lot.acquired_date:%Y-%m-%d} "
                                   f"within 30 days after the {c.sale_date:%Y-%m-%d} loss sale absorb ${disallowed:,.2f}.")
            available -= take
            touched.append(c)
        return touched

    def _attach_wash(self, closure: Closure, repl: Lot, qty: float, disallowed: float, note: str) -> None:
        closure.wash_disallowed += disallowed
        closure.wash_replacement_lot = repl
        closure.wash_explanation = (closure.wash_explanation + " " + note).strip()
        closure.wash_matched_quantity += qty
        repl.extra["used_as_replacement"] = float(repl.extra.get("used_as_replacement", 0.0)) + qty
        if repl.account_type in {"ira", "roth", "401k", "other_deferred"}:
            # Rev. Rul. 2008-5: no basis adjustment, loss is gone for good.
            closure.wash_explanation += " Replacement in tax-deferred account: no basis step-up."
            return
        repl.basis_adjustment += disallowed
        if closure.lot.holding_start_date < repl.holding_start_date:
            repl.holding_start_date = closure.lot.holding_start_date

    def record_sale(
        self, account_id: int, assetid: int, sale_date: date, quantity: float, price: float,
        fees: float = 0.0, method: LotMethod = LotMethod.HIFO, specific_ids: list[int] | None = None,
    ) -> list[Closure]:
        lots = self.open_lots(account_id, assetid)
        if not lots:
            raise ValueError(f"no open lots for asset {assetid} in account {account_id}")
        slices = select_lots(lots, quantity, method, price=price, as_of=sale_date, specific_ids=specific_ids)
        # Close oldest lots first so any wash-sale basis adjustment lands on a younger lot that is still
        # open at that moment (and is then realised if that younger lot closes later in this same sale).
        slices = sorted(slices, key=lambda s: (s.lot.holding_start_date, s.lot.id or 0))
        fee_ps = fees / quantity if quantity else 0.0
        out: list[Closure] = []
        acqs = self.acquisitions(include_scheduled=False)
        for sl in slices:
            out.append(self._close_slice(sl, sale_date, price - fee_ps, acqs))
        assert abs(sum(c.quantity for c in out) - quantity) < 1e-9
        return out

    def _close_slice(self, sl: LotSlice, sale_date: date, net_price: float, acqs: list[Acquisition]) -> Closure:
        lot = sl.lot
        proceeds = net_price * sl.quantity
        basis = lot.basis_for(sl.quantity)
        c = Closure(lot=lot, sale_date=sale_date, quantity=sl.quantity, proceeds=proceeds, cost_basis=basis,
                    term=term_for(lot.holding_start_date, sale_date))
        lot.quantity_open -= sl.quantity
        if lot.quantity_open <= 1e-9:
            lot.quantity_open = 0.0
            lot.is_closed = True
        if c.realized_gain < 0:
            sale = LossSale(lot.assetid, lot.symbol, lot.account_id, sale_date, sl.quantity, -c.realized_gain,
                            lot_id=lot.id, lot_holding_start=lot.holding_start_date)
            det = evaluate_loss_sale(sale, acqs, self.groups, include_scheduled=False)
            c.determination = det
            for m in det.matches:
                repl = next((x for x in self.lots if x.id == m.acquisition.lot_id), None)
                if repl is None:
                    continue
                self._attach_wash(c, repl, m.quantity, m.disallowed_loss, note="")
            c.wash_explanation = det.explanation
        self.closures.append(c)
        return c

    # --------------------------------------------------------------------- reporting
    def realized_summary(self, year: int) -> dict:
        cs = [c for c in self.closures if c.sale_date.year == year]
        st = [c for c in cs if c.term == ST]
        lt = [c for c in cs if c.term == LT]
        return {
            "year": year,
            "st_gains": sum(c.allowed_gain for c in st if c.allowed_gain > 0),
            "st_losses": sum(c.allowed_gain for c in st if c.allowed_gain < 0),
            "lt_gains": sum(c.allowed_gain for c in lt if c.allowed_gain > 0),
            "lt_losses": sum(c.allowed_gain for c in lt if c.allowed_gain < 0),
            "wash_disallowed": sum(c.wash_disallowed for c in cs),
            "net_st": sum(c.allowed_gain for c in st),
            "net_lt": sum(c.allowed_gain for c in lt),
            "n_closures": len(cs),
        }


# ------------------------------------------------------------------------------ carryforward netting
@dataclass(frozen=True)
class NettingResult:
    net_st: float                 # after carryover, before ordinary offset (negative = loss)
    net_lt: float
    taxable_st_gain: float        # amounts actually taxed this year
    taxable_lt_gain: float
    ordinary_deduction: float     # loss used against ordinary income (positive number)
    carryforward_st: float        # positive = loss carried to next year
    carryforward_lt: float


def net_capital_position(
    st_gain_loss: float, lt_gain_loss: float, prior_cf_st: float = 0.0, prior_cf_lt: float = 0.0,
    ordinary_offset: float = 3000.0,
) -> NettingResult:
    """Schedule D netting.

    Step 1: net within character, bringing carryovers in (carryovers are losses, passed as positives).
    Step 2: if one is a gain and the other a loss, net across. A remaining loss keeps the character of the
            loss side.
    Step 3: up to `ordinary_offset` of remaining net loss is deducted against ordinary income, short-term
            first (§1211(b), §1212(b)); the rest carries forward retaining character.
    """
    st = st_gain_loss - prior_cf_st
    lt = lt_gain_loss - prior_cf_lt
    taxable_st = taxable_lt = 0.0
    loss_st = loss_lt = 0.0

    if st >= 0 and lt >= 0:
        taxable_st, taxable_lt = st, lt
    elif st < 0 and lt < 0:
        loss_st, loss_lt = -st, -lt
    elif st >= 0 > lt:                       # ST gain, LT loss
        net = st + lt
        if net >= 0:
            taxable_st = net
        else:
            loss_lt = -net
    else:                                    # ST loss, LT gain
        net = st + lt
        if net >= 0:
            taxable_lt = net
        else:
            loss_st = -net

    remaining_offset = max(ordinary_offset, 0.0)
    use_st = min(loss_st, remaining_offset)
    remaining_offset -= use_st
    use_lt = min(loss_lt, remaining_offset)
    ordinary = use_st + use_lt
    return NettingResult(
        net_st=st, net_lt=lt, taxable_st_gain=taxable_st, taxable_lt_gain=taxable_lt,
        ordinary_deduction=ordinary, carryforward_st=loss_st - use_st, carryforward_lt=loss_lt - use_lt,
    )
