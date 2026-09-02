"""Portfolio views: lots with live valuation, wash-sale status per lot, wash calendar, realized ledger."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..tax.ledger import LotBook, net_capital_position
from ..tax.lots import LotMethod
from ..tax.washsale import repurchase_allowed_from, screen_proposed_sale, window_for
from .context import AppContext
from .data_service import DataService


class PortfolioService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.data = DataService(ctx)

    # ------------------------------------------------------------------ hydration
    def book(self, entity_id: int) -> LotBook:
        return self.ctx.portfolio.load_book(entity_id, self.ctx.groups())

    def prices(self, entity_id: int, snap=None) -> pd.Series:
        snap = snap or self.data.latest_snapshot()
        held = self.ctx.portfolio.held_symbols(entity_id)
        if snap is None:
            px = pd.Series(dtype=float)
            if self.ctx.norgate.status():
                for s in held:
                    lc = self.ctx.norgate.last_close(s)
                    if lc:
                        px[s] = lc[1]
            return px
        return self.data.prices_for(snap, held)

    # ------------------------------------------------------------------ positions
    def lots_view(self, entity_id: int, as_of: date | None = None, snap=None) -> pd.DataFrame:
        as_of = as_of or date.today()
        book = self.book(entity_id)
        px = self.prices(entity_id, snap)
        accts = {a.id: a for a in self.ctx.entities.accounts(entity_id, active_only=False)}
        acqs = book.acquisitions(include_scheduled=True)
        rows = []
        for lot in book.open_lots():
            p = float(px.get(lot.symbol, float("nan")))
            ug = lot.unrealized_gain(p) if p == p else float("nan")
            wash_status, wash_expl = "", ""
            if p == p and ug < 0:
                det = screen_proposed_sale(lot.assetid, lot.symbol, lot.account_id, as_of, lot.quantity_open, -ug,
                                           acqs, book.groups, lot_id=lot.id)
                wash_status, wash_expl = det.status, det.explanation
            rows.append({
                "lot_id": lot.id, "account": accts[lot.account_id].name, "account_id": lot.account_id,
                "account_type": lot.account_type, "symbol": lot.symbol, "assetid": lot.assetid,
                "acquired": lot.acquired_date, "holding_start": lot.holding_start_date,
                "quantity": lot.quantity_open, "cost_per_share": lot.basis_per_share, "cost_basis": lot.open_basis,
                "price": p, "market_value": lot.market_value(p) if p == p else float("nan"),
                "unrealized": ug, "unrealized_pct": lot.unrealized_gain_pct(p) if p == p else float("nan"),
                "term": lot.term_at(as_of), "days_to_lt": lot.days_to_long_term(as_of),
                "basis_adjustment": lot.basis_adjustment, "source": lot.source,
                "wash_status": wash_status, "wash_explanation": wash_expl,
            })
        cols = ["lot_id", "account", "account_id", "account_type", "symbol", "assetid", "acquired", "holding_start",
                "quantity", "cost_per_share", "cost_basis", "price", "market_value", "unrealized", "unrealized_pct",
                "term", "days_to_lt", "basis_adjustment", "source", "wash_status", "wash_explanation"]
        return pd.DataFrame(rows, columns=cols)

    def positions_view(self, lots: pd.DataFrame) -> pd.DataFrame:
        if lots.empty:
            return pd.DataFrame()
        g = lots.groupby(["symbol"], as_index=False)
        out = g.agg(quantity=("quantity", "sum"), cost_basis=("cost_basis", "sum"),
                    market_value=("market_value", "sum"), unrealized=("unrealized", "sum"),
                    n_lots=("lot_id", "count"), price=("price", "first"),
                    accounts=("account", lambda s: ", ".join(sorted(set(s)))))
        st = lots[lots["term"] == "ST"].groupby("symbol")["unrealized"].sum()
        lt = lots[lots["term"] == "LT"].groupby("symbol")["unrealized"].sum()
        losses = lots[lots["unrealized"] < 0].groupby("symbol")["unrealized"].sum()
        out["unrealized_st"] = out["symbol"].map(st).fillna(0.0)
        out["unrealized_lt"] = out["symbol"].map(lt).fillna(0.0)
        out["harvestable_loss"] = -out["symbol"].map(losses).fillna(0.0)
        tot = out["market_value"].sum()
        out["weight"] = out["market_value"] / tot if tot else 0.0
        out["unrealized_pct"] = out["market_value"] / out["cost_basis"] - 1.0
        return out.sort_values("market_value", ascending=False).reset_index(drop=True)

    def summary(self, lots: pd.DataFrame) -> dict:
        if lots.empty:
            return {"market_value": 0.0, "cost_basis": 0.0, "unrealized": 0.0, "harvestable_loss": 0.0,
                    "harvestable_st": 0.0, "harvestable_lt": 0.0, "n_lots": 0, "n_positions": 0, "blocked_loss": 0.0}
        losses = lots[lots["unrealized"] < 0]
        return {
            "market_value": float(lots["market_value"].sum()), "cost_basis": float(lots["cost_basis"].sum()),
            "unrealized": float(lots["unrealized"].sum()),
            "harvestable_loss": float(-losses["unrealized"].sum()),
            "harvestable_st": float(-losses.loc[losses["term"] == "ST", "unrealized"].sum()),
            "harvestable_lt": float(-losses.loc[losses["term"] == "LT", "unrealized"].sum()),
            "blocked_loss": float(-losses.loc[losses["wash_status"].isin(["WASH", "BLOCKED_FORWARD", "PARTIAL_WASH"]), "unrealized"].sum()),
            "n_lots": int(len(lots)), "n_positions": int(lots["symbol"].nunique()),
        }

    # ------------------------------------------------------------------ wash calendar
    def wash_calendar(self, entity_id: int, as_of: date | None = None) -> pd.DataFrame:
        """Every open 61-day window that constrains trading today: recent loss sales (block buys of the group) and
        recent purchases (would wash a loss sale of the group)."""
        as_of = as_of or date.today()
        book = self.book(entity_id)
        accts = {a.id: a.name for a in self.ctx.entities.accounts(entity_id, active_only=False)}
        rows = []
        for c in book.closures:
            if c.realized_gain < 0 and (as_of - c.sale_date).days <= 30:
                s, e = window_for(c.sale_date)
                rows.append({"symbol": c.lot.symbol, "group": book.groups.group_of(c.lot.assetid), "kind": "loss_sale",
                             "event_date": c.sale_date, "window_start": s, "window_end": e,
                             "constraint": f"Do not buy {c.lot.symbol} or substantially identical until {repurchase_allowed_from(c.sale_date):%Y-%m-%d}",
                             "amount": -c.realized_gain, "account": accts.get(c.lot.account_id, "")})
        for lot in book.lots:
            if 0 <= (as_of - lot.acquired_date).days <= 30 and lot.quantity_open > 0:
                s, e = window_for(lot.acquired_date)
                rows.append({"symbol": lot.symbol, "group": book.groups.group_of(lot.assetid), "kind": "purchase",
                             "event_date": lot.acquired_date, "window_start": s, "window_end": e,
                             "constraint": f"Selling {lot.symbol} (or identical) at a loss before {(lot.acquired_date + timedelta(days=31)):%Y-%m-%d} would be a wash sale",
                             "amount": lot.quantity_original * lot.cost_per_share, "account": accts.get(lot.account_id, "")})
        for a in book.scheduled:
            if -30 <= (a.acquired_date - as_of).days <= 30:
                s, e = window_for(a.acquired_date)
                rows.append({"symbol": a.symbol, "group": book.groups.group_of(a.assetid), "kind": a.kind,
                             "event_date": a.acquired_date, "window_start": s, "window_end": e,
                             "constraint": f"Scheduled {a.kind.replace('scheduled_', '')} of {a.symbol} on {a.acquired_date:%Y-%m-%d} blocks loss sales within 30 days",
                             "amount": a.quantity, "account": accts.get(a.account_id, "")})
        return pd.DataFrame(rows).sort_values("event_date") if rows else pd.DataFrame(
            columns=["symbol", "group", "kind", "event_date", "window_start", "window_end", "constraint", "amount", "account"])

    # ------------------------------------------------------------------ realized
    def realized(self, entity_id: int, year: int) -> dict:
        book = self.book(entity_id)
        s = book.realized_summary(year)
        prof = self.ctx.tax.default_profile()
        cf_st, cf_lt = self.ctx.tax.carryforward(entity_id, year - 1)
        net = net_capital_position(s["net_st"], s["net_lt"], cf_st, cf_lt, prof.effective_ordinary_offset)
        s.update({"prior_cf_st": cf_st, "prior_cf_lt": cf_lt, "netting": net.__dict__})
        return s

    def closures_view(self, entity_id: int) -> pd.DataFrame:
        return self.ctx.portfolio.closures_frame(entity_id)

    # ------------------------------------------------------------------ mutations
    def buy(self, account_id: int, symbol: str, trade_date: date, quantity: float, price: float, fees: float = 0.0,
            source: str = "buy", notes: str | None = None):
        aid = self.ctx.resolve_assetid(symbol)
        if aid is None:
            raise ValueError(f"unknown symbol {symbol}")
        return self.ctx.portfolio.record_purchase(account_id, symbol.upper(), aid, trade_date, quantity, price, fees,
                                                  source=source, notes=notes, groups=self.ctx.groups())

    def sell(self, account_id: int, symbol: str, sale_date: date, quantity: float, price: float, fees: float = 0.0,
             method: LotMethod = LotMethod.HIFO, specific_ids: list[int] | None = None, notes: str | None = None,
             source: str = "manual"):
        aid = self.ctx.resolve_assetid(symbol)
        if aid is None:
            raise ValueError(f"unknown symbol {symbol}")
        return self.ctx.portfolio.record_sale(account_id, symbol.upper(), aid, sale_date, quantity, price, fees, method,
                                              specific_ids, notes=notes, groups=self.ctx.groups(), source=source)
