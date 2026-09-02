"""Seed a realistic demo household so every screen has data on first launch.

Uses actual historical unadjusted closes from the data snapshot for cost basis, so unrealised gains/losses,
holding periods and wash-sale windows are real. Idempotent: does nothing if the demo entity already exists.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .context import AppContext
from .data_service import DataService
from .portfolio_service import PortfolioService

log = logging.getLogger(__name__)

DEMO_ENTITY = "Demo Household"

BROKERAGE = [  # (symbol, shares, months_ago)
    ("AAPL", 120, 30), ("AAPL", 40, 8), ("MSFT", 60, 26), ("MSFT", 25, 5), ("NVDA", 80, 20), ("AMZN", 70, 14),
    ("GOOGL", 90, 22), ("META", 30, 9), ("BRK.B", 50, 34), ("JPM", 60, 28), ("UNH", 25, 11), ("XOM", 90, 18),
    ("JNJ", 70, 33), ("PG", 60, 16), ("HD", 30, 7), ("MRK", 80, 13), ("PFE", 200, 24), ("KO", 120, 31), ("PEP", 50, 10),
    ("COST", 20, 12), ("CVX", 60, 15), ("ABBV", 45, 6), ("NKE", 90, 9), ("DIS", 110, 27), ("INTC", 300, 21),
    ("BA", 40, 19), ("TGT", 70, 8), ("MMM", 60, 23), ("SPY", 60, 36), ("SPY", 25, 4), ("QQQ", 40, 17), ("XLK", 80, 6),
    ("LLY", 15, 3), ("ADBE", 30, 10), ("CRM", 35, 5), ("PYPL", 120, 25), ("SBUX", 90, 14), ("CMCSA", 200, 29),
]
SPOUSE = [("IVV", 25, 20), ("VTV", 80, 12), ("HD", 20, 3), ("UNH", 15, 4), ("NKE", 60, 2), ("SCHD", 150, 9)]
IRA = [("VTI", 120, 40), ("AGG", 200, 30), ("QQQ", 30, 2)]


def reset_demo(ctx: AppContext) -> None:
    """Delete the demo entity and everything attached to it (runs are kept but detached)."""
    ents = {e["name"]: e["id"] for e in ctx.entities.list()}
    eid = ents.get(DEMO_ENTITY)
    if eid is None:
        return
    with ctx.db.transaction():
        acct_ids = [a.id for a in ctx.entities.accounts(eid, active_only=False)]
        if acct_ids:
            ph = ",".join("?" * len(acct_ids))
            ctx.db.execute(f"DELETE FROM run_trades WHERE account_id IN ({ph})", acct_ids)
            ctx.db.execute(f"DELETE FROM lot_closures WHERE lot_id IN (SELECT id FROM lots WHERE account_id IN ({ph}))", acct_ids)
            ctx.db.execute(f"DELETE FROM lots WHERE account_id IN ({ph})", acct_ids)
            ctx.db.execute(f"DELETE FROM transactions WHERE account_id IN ({ph})", acct_ids)
            ctx.db.execute(f"DELETE FROM scheduled_events WHERE account_id IN ({ph})", acct_ids)
        ctx.db.execute("UPDATE runs SET entity_id = NULL WHERE entity_id = ?", (eid,))
        ctx.db.execute("DELETE FROM carryforwards WHERE entity_id = ?", (eid,))
        ctx.db.execute("DELETE FROM accounts WHERE entity_id = ?", (eid,))
        ctx.db.execute("DELETE FROM entities WHERE id = ?", (eid,))
        ctx.db.audit("user", "demo.reset", str(eid))
    if ctx.get("current_entity_id") == eid:
        ctx.db.execute("DELETE FROM settings WHERE key = 'current_entity_id'")


def seed_demo(ctx: AppContext, progress=None) -> int:
    say = progress or (lambda m: log.info(m))
    ents = {e["name"]: e["id"] for e in ctx.entities.list()}
    if DEMO_ENTITY in ents:
        say("Demo household already exists.")
        return ents[DEMO_ENTITY]
    eid = ctx.entities.get_or_create(DEMO_ENTITY, filing_status="mfj")
    brok = ctx.entities.get_or_create_account(eid, "Schwab Brokerage", "taxable", broker="Schwab", owner="self")
    spouse = ctx.entities.get_or_create_account(eid, "Spouse Fidelity", "taxable", broker="Fidelity", owner="spouse")
    ira = ctx.entities.get_or_create_account(eid, "Rollover IRA", "ira", broker="Schwab", owner="self")
    ctx.current_entity_id = eid

    from ..tax.rates import TaxProfile
    ctx.tax.save(TaxProfile(name="default", fed_st_rate=0.37, fed_lt_rate=0.20, state_rate=0.05, niit_rate=0.038,
                            filing_status="mfj"), make_default=True)
    ctx.tax.set_carryforward(eid, date.today().year - 1, st=4200.0, lt=0.0, notes="demo carryforward")

    data = DataService(ctx)
    say("Ensuring data snapshot for demo universe...")
    symbols = sorted({s for s, _, _ in BROKERAGE + SPOUSE + IRA})
    snap = data.latest_snapshot()
    have = set(snap.manifest().get("returned_symbols", [])) if snap else set()
    if snap is None or not set(symbols) <= have:
        snap = ctx.store.create(data.universe_name(), sorted(set(data.universe_symbols()) | set(symbols)),
                                ctx.settings.price_history_start, notes="demo seed", progress=say)
        ctx.securities.upsert_frame(snap.securities())
    unadj = snap.close_matrix("unadj_close")
    ps = PortfolioService(ctx)
    rng = np.random.default_rng(7)
    today = date.today()

    def px_on(sym: str, d: date) -> tuple[date, float] | None:
        if sym not in unadj.columns:
            return None
        s = unadj[sym].dropna()
        s = s.loc[: pd.Timestamp(d)]
        if s.empty:
            return None
        return s.index[-1].date(), float(s.iloc[-1])

    def buy_all(acct: int, rows) -> None:
        for sym, qty, months in rows:
            d = today - timedelta(days=int(months * 30.4 + rng.integers(0, 10)))
            hit = px_on(sym, d)
            if hit is None:
                say(f"  skip {sym}: no price history")
                continue
            d_actual, p = hit
            ps.buy(acct, sym, d_actual, qty, round(p, 2), fees=0.0, notes="demo seed")

    say("Seeding brokerage lots...")
    buy_all(brok, BROKERAGE)
    say("Seeding spouse lots...")
    buy_all(spouse, SPOUSE)
    say("Seeding IRA lots...")
    buy_all(ira, IRA)

    # A recent LOSS sale (opens a 30-day buy block on that name) — pick the first brokerage lot that was
    # under water 12 days ago so the wash calendar and the optimizer's replacement screen have something to show.
    book = ps.book(eid)
    sale_day = today - timedelta(days=12)
    for lot in sorted(book.open_lots(brok), key=lambda x: x.symbol):
        hit = px_on(lot.symbol, sale_day)
        if hit and hit[1] < lot.cost_per_share * 0.97 and lot.quantity_open >= 20:
            qty = float(int(lot.quantity_open * 0.3))
            ps.sell(brok, lot.symbol, hit[0], qty, round(hit[1], 2), notes="demo: recent loss sale")
            say(f"  demo loss sale: {qty:g} {lot.symbol} @ {hit[1]:.2f} on {hit[0]}")
            break
    # A recent re-purchase of another loser (10 days ago) so one loss lot shows as wash-blocked today.
    for lot in sorted(book.open_lots(brok), key=lambda x: x.symbol, reverse=True):
        hit_now = px_on(lot.symbol, today)
        hit_then = px_on(lot.symbol, today - timedelta(days=10))
        if hit_now and hit_then and hit_now[1] < lot.cost_per_share * 0.9:
            ps.buy(spouse, lot.symbol, hit_then[0], 10, round(hit_then[1], 2), notes="demo: recent buy in spouse account")
            say(f"  demo recent buy: 10 {lot.symbol} in spouse account on {hit_then[0]} (wash-blocks the brokerage lot)")
            break
    spy_aid = ctx.resolve_assetid("SPY")
    if spy_aid:
        ctx.portfolio.add_scheduled_event(brok, "SPY", spy_aid, today + timedelta(days=12), "DRIP", 0.8,
                                          notes="demo: quarterly dividend reinvestment")
    ctx.db.audit("system", "demo.seed", str(eid))
    say("Demo household seeded.")
    return eid
