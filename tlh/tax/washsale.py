"""Wash-sale compliance engine (IRC §1091, Treas. Reg. §1.1091-1, Rev. Rul. 2008-5).

Pure functions over plain dataclasses so the rules can be unit-tested exhaustively without a database.
The ledger (ledger.py) feeds it real lots/transactions; the optimizer (optim/harvest.py) feeds it proposed
trades. Every determination carries a human-readable explanation.

Rules implemented
-----------------
* Window: 30 calendar days before and 30 after the loss sale (61 days inclusive of the sale date).
* Scope: every acquisition of a security in the same substantially-identical group, in ANY account of the
  same tax entity, including IRAs/Roths/401(k)s.
* Replacement matching: acquisitions are matched to loss shares in chronological order of acquisition
  (§1.1091-1(b)-(c)); each replacement share can absorb at most one loss share; the shares being sold are
  never their own replacement.
* Partial: if replacement shares < loss shares, only the proportional part of the loss is disallowed.
* Consequence: disallowed loss is added to the replacement lot's basis; the sold lot's holding period tacks
  onto the replacement lot. If the replacement was bought in an IRA (Rev. Rul. 2008-5) the loss is
  disallowed permanently with no basis step-up.
* Forward guard: scheduled/known future purchases and DRIPs within the next 30 days are treated as
  acquisitions for pre-trade screening.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

WINDOW_DAYS = 30

TAX_DEFERRED_TYPES = {"ira", "roth", "401k", "other_deferred"}


@dataclass
class Acquisition:
    """Any event that adds shares: a buy, a DRIP, a transfer-in, or a scheduled future purchase."""
    assetid: int
    symbol: str
    account_id: int
    account_type: str
    acquired_date: date
    quantity: float
    lot_id: int | None = None
    kind: str = "buy"                       # buy | drip | transfer | scheduled_buy | scheduled_drip
    used_as_replacement: float = 0.0        # shares already absorbing an earlier wash sale
    disposed_date: date | None = None       # date the lot was fully closed, if it was

    @property
    def available(self) -> float:
        return max(self.quantity - self.used_as_replacement, 0.0)

    @property
    def is_tax_deferred(self) -> bool:
        return self.account_type in TAX_DEFERRED_TYPES

    @property
    def is_scheduled(self) -> bool:
        return self.kind.startswith("scheduled")


@dataclass
class LossSale:
    assetid: int
    symbol: str
    account_id: int
    sale_date: date
    quantity: float
    loss_amount: float                      # positive dollars of loss
    lot_id: int | None = None
    lot_holding_start: date | None = None


@dataclass
class ReplacementMatch:
    acquisition: Acquisition
    quantity: float
    disallowed_loss: float
    permanent: bool                          # True when replacement sits in a tax-deferred account


@dataclass
class WashSaleDetermination:
    sale: LossSale
    window_start: date
    window_end: date
    status: str                              # SAFE | WASH | PARTIAL_WASH | BLOCKED_FORWARD
    matches: list[ReplacementMatch] = field(default_factory=list)
    disallowed_quantity: float = 0.0
    disallowed_loss: float = 0.0
    explanation: str = ""

    @property
    def allowed_loss(self) -> float:
        return self.sale.loss_amount - self.disallowed_loss

    @property
    def is_clean(self) -> bool:
        return self.status == "SAFE"

    @property
    def has_forward_conflict(self) -> bool:
        return any(m.acquisition.is_scheduled for m in self.matches)


def window_for(sale_date: date) -> tuple[date, date]:
    return sale_date - timedelta(days=WINDOW_DAYS), sale_date + timedelta(days=WINDOW_DAYS)


def in_window(acq_date: date, sale_date: date) -> bool:
    start, end = window_for(sale_date)
    return start <= acq_date <= end


def repurchase_allowed_from(sale_date: date) -> date:
    """First date on which the same (or substantially identical) security may be bought again."""
    return sale_date + timedelta(days=WINDOW_DAYS + 1)


class SubstantiallyIdentical:
    """Maps assetids to a group key. Anything sharing a key is substantially identical.

    The default implementation treats each assetid as its own group; `data/substitutes.py` supplies the
    real mapping (share classes, same-index ETFs).
    """

    def __init__(self, mapping: dict[int, str] | None = None):
        self._map = dict(mapping or {})

    def group_of(self, assetid: int) -> str:
        return self._map.get(assetid, f"asset:{assetid}")

    def same_group(self, a: int, b: int) -> bool:
        return self.group_of(a) == self.group_of(b)


def evaluate_loss_sale(
    sale: LossSale,
    acquisitions: list[Acquisition],
    groups: SubstantiallyIdentical | None = None,
    *,
    include_scheduled: bool = True,
) -> WashSaleDetermination:
    """Determine whether `sale` is (partly) a wash sale given all acquisitions in the tax entity.

    Mutates `used_as_replacement` on matched acquisitions so that a sequence of sales evaluated in
    chronological order never double-counts replacement shares. Callers evaluating hypothetical trades
    should pass copies.
    """
    groups = groups or SubstantiallyIdentical()
    start, end = window_for(sale.sale_date)
    group = groups.group_of(sale.assetid)

    candidates = [
        a for a in acquisitions
        if groups.group_of(a.assetid) == group
        and start <= a.acquired_date <= end
        and a.available > 1e-12
        and not (sale.lot_id is not None and a.lot_id == sale.lot_id)      # never its own replacement
        and (a.disposed_date is None or a.disposed_date >= sale.sale_date)  # see convention note below
        and (include_scheduled or not a.is_scheduled)
    ]
    # Convention: shares acquired in the window but fully disposed of BEFORE the loss sale are not treated
    # as replacement shares (the taxpayer no longer holds anything for the disallowed loss to attach to).
    # Same-day dispositions are still counted, so "buy B, sell A and B together" washes A into B's basis
    # and the loss is realised on B; the total deductible loss is preserved.
    candidates.sort(key=lambda a: (a.acquired_date, a.lot_id or 0))

    det = WashSaleDetermination(sale=sale, window_start=start, window_end=end, status="SAFE")
    if not candidates:
        det.explanation = (
            f"SAFE: no purchase of {sale.symbol} or a substantially identical security in any linked "
            f"account between {start:%Y-%m-%d} and {end:%Y-%m-%d}. Repurchase permitted from "
            f"{repurchase_allowed_from(sale.sale_date):%Y-%m-%d}."
        )
        return det

    remaining = sale.quantity
    loss_per_share = sale.loss_amount / sale.quantity if sale.quantity else 0.0
    lines: list[str] = []
    for acq in candidates:
        if remaining <= 1e-12:
            break
        take = min(acq.available, remaining)
        disallowed = take * loss_per_share
        acq.used_as_replacement += take
        permanent = acq.is_tax_deferred
        det.matches.append(ReplacementMatch(acq, take, disallowed, permanent))
        det.disallowed_quantity += take
        det.disallowed_loss += disallowed
        remaining -= take
        when = "scheduled" if acq.is_scheduled else "bought"
        rel = "before" if acq.acquired_date <= sale.sale_date else "after"
        days = abs((acq.acquired_date - sale.sale_date).days)
        tail = (" Replacement is in a tax-deferred account: loss is PERMANENTLY disallowed (Rev. Rul. 2008-5)."
                if permanent else " Disallowed loss is added to the replacement lot's basis and the holding period tacks.")
        lines.append(
            f"{take:g} sh of {acq.symbol} {when} {acq.acquired_date:%Y-%m-%d} in account {acq.account_id} "
            f"({days} days {rel} sale) absorb ${disallowed:,.2f} of the loss.{tail}"
        )

    if det.disallowed_quantity >= sale.quantity - 1e-9:
        det.status = "WASH"
        head = f"WASH SALE: the entire ${sale.loss_amount:,.2f} loss on {sale.quantity:g} sh of {sale.symbol} is disallowed."
    else:
        det.status = "PARTIAL_WASH"
        head = (f"PARTIAL WASH SALE: ${det.disallowed_loss:,.2f} of the ${sale.loss_amount:,.2f} loss on "
                f"{sale.symbol} is disallowed ({det.disallowed_quantity:g} of {sale.quantity:g} sh matched); "
                f"${det.allowed_loss:,.2f} remains deductible.")
    if det.has_forward_conflict:
        det.status = "BLOCKED_FORWARD" if det.status == "WASH" else det.status
        head += " A known FUTURE purchase/DRIP falls inside the window."
    det.explanation = head + " " + " ".join(lines)
    return det


def screen_proposed_sale(
    assetid: int,
    symbol: str,
    account_id: int,
    sale_date: date,
    quantity: float,
    loss_amount: float,
    acquisitions: list[Acquisition],
    groups: SubstantiallyIdentical | None = None,
    lot_id: int | None = None,
) -> WashSaleDetermination:
    """Pre-trade screen used by the optimizer. Never mutates the acquisitions passed in."""
    import copy

    sale = LossSale(assetid, symbol, account_id, sale_date, quantity, loss_amount, lot_id=lot_id)
    return evaluate_loss_sale(sale, copy.deepcopy(acquisitions), groups, include_scheduled=True)


@dataclass
class BuyScreen:
    assetid: int
    symbol: str
    buy_date: date
    status: str                              # SAFE | WOULD_WASH
    conflicting_sales: list[LossSale] = field(default_factory=list)
    explanation: str = ""


def screen_proposed_buy(
    assetid: int,
    symbol: str,
    buy_date: date,
    recent_loss_sales: list[LossSale],
    groups: SubstantiallyIdentical | None = None,
) -> BuyScreen:
    """Would buying `assetid` on `buy_date` turn an existing or proposed loss sale into a wash sale?

    A purchase is a problem if a loss sale of the same group happened (or is proposed) within 30 days
    before OR after the purchase date.
    """
    groups = groups or SubstantiallyIdentical()
    group = groups.group_of(assetid)
    conflicts = [
        s for s in recent_loss_sales
        if groups.group_of(s.assetid) == group and in_window(buy_date, s.sale_date) and s.loss_amount > 0
    ]
    if not conflicts:
        return BuyScreen(assetid, symbol, buy_date, "SAFE",
                         explanation=f"SAFE: no loss sale of {symbol} or a substantially identical security within "
                                     f"30 days of {buy_date:%Y-%m-%d}.")
    parts = [f"{s.quantity:g} sh {s.symbol} sold at a ${s.loss_amount:,.2f} loss on {s.sale_date:%Y-%m-%d} "
             f"(account {s.account_id})" for s in conflicts]
    return BuyScreen(
        assetid, symbol, buy_date, "WOULD_WASH", conflicts,
        explanation=f"WOULD TRIGGER WASH SALE: buying {symbol} on {buy_date:%Y-%m-%d} is within 30 days of "
                    + "; ".join(parts) + ". Earliest clean purchase date: "
                    + f"{max(repurchase_allowed_from(s.sale_date) for s in conflicts):%Y-%m-%d}.",
    )


def blocked_groups_for_purchase(
    recent_loss_sales: list[LossSale], buy_date: date, groups: SubstantiallyIdentical
) -> set[str]:
    """Set of substantially-identical group keys that cannot be bought on `buy_date`."""
    return {groups.group_of(s.assetid) for s in recent_loss_sales
            if s.loss_amount > 0 and in_window(buy_date, s.sale_date)}
