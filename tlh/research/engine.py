"""Lot-level monthly simulator for the harvesting research.

One `run_window` call = one account, one parameter set, one rolling window (e.g. 2007-01 to 2017-01) on the point-in-time
S&P 500. Every month-end:

  1. mark to market; force-sell names that were delisted (at their last close);
  2. harvest: sell every lot whose loss clears the trigger (a fraction of account value, or of the lot's cost), unless the
     name was bought inside the wash window (selling it at a loss would be a wash sale);
  3. unwind a concentrated position only as far as realised losses (plus any gain budget) cover the gain;
  4. reinvest the proceeds and any cash by the chosen approach:
        pairs_sector / pairs_index  -> most correlated eligible index member (same sector / anywhere)
        twin_baskets                -> the pre-assigned twin (SARD pairing), falling back to pairs_sector
        optimizer                   -> convex minimum-TE buy list against the index under sector / factor / name-count limits
     never buying a name sold at a loss inside the wash window;
  5. record harvested losses (short / long by holding period), realised gains, forecast TE, names held, turnover.

Between month-ends holdings are fixed; daily portfolio values (prices + cash + dividends) give the realised tracking
error against the S&P 500 total-return index. Positions are whole shares by default, so account size matters.

All arithmetic is numpy over the memory-mapped store; the optimizer uses cvxpy (CLARABEL) on a reduced candidate set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..lazy import lazy_module
from .spec import APPROACHES, ResearchSpec

cp = lazy_module("cvxpy")
log = logging.getLogger(__name__)

LOOKBACK = 252
FACTORS = ("size", "momentum", "volatility", "beta")


@dataclass
class Lot:
    sym: int
    qty: float
    basis: float          # per share
    opened: int           # row position


@dataclass
class RunResult:
    spec: ResearchSpec
    metrics: dict
    monthly: pd.DataFrame
    daily: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


# ====================================================================================== descriptors / risk
def _window_returns(store, t: int, lookback: int = LOOKBACK) -> np.ndarray:
    a = max(t - lookback, 1)
    c = np.asarray(store.close[a - 1: t + 1], dtype=float)
    d = np.asarray(store.dividend[a: t + 1], dtype=float)
    return (c[1:] + d) / c[:-1] - 1.0


def _index_window(store, t: int, lookback: int = LOOKBACK) -> np.ndarray:
    a = max(t - lookback, 1)
    lvl = store.index_tr[a - 1: t + 1]
    return lvl[1:] / lvl[:-1] - 1.0


def descriptors(store, t: int, members: np.ndarray, R: np.ndarray | None = None) -> pd.DataFrame:
    """Cross-sectional z-scores among `members` (bool mask) at row t: size, momentum (12-1), volatility, beta, dividend yield."""
    R = _window_returns(store, t) if R is None else R
    ri = _index_window(store, t)
    cap = store.cap_proxy(t)
    idx = np.where(members)[0]
    Rm = R[:, idx]
    ok = np.isfinite(Rm).sum(axis=0) >= int(0.6 * len(Rm))
    Rm = np.where(np.isfinite(Rm), Rm, 0.0)
    vol = Rm.std(axis=0)
    ri_c = ri - ri.mean()
    beta = (Rm - Rm.mean(axis=0)).T @ ri_c / max(float(ri_c @ ri_c), 1e-12)
    c_now = np.asarray(store.close[t, idx], dtype=float)
    back = max(t - LOOKBACK, 0)
    c_back = np.asarray(store.close[back, idx], dtype=float)
    c_1m = np.asarray(store.close[max(t - 21, 0), idx], dtype=float)
    mom = c_1m / c_back - 1.0
    dy = np.asarray(store.dividend[back: t + 1, idx], dtype=float).sum(axis=0) / np.where(c_now > 0, c_now, np.nan)
    df = pd.DataFrame({"size": np.log(np.where(cap[idx] > 0, cap[idx], np.nan)), "momentum": mom, "volatility": vol, "beta": beta, "yield": dy}, index=idx)
    df.loc[~ok, ["volatility", "beta"]] = np.nan
    for c in df.columns:
        x = df[c]
        med, mad = x.median(), (x - x.median()).abs().median() * 1.4826 + 1e-12
        z = ((x - med) / mad).clip(-3, 3)
        df[c] = z.fillna(0.0)
    return df


def lw_covariance(R: np.ndarray, lookback: int) -> np.ndarray:
    """Ledoit-Wolf constant-correlation shrinkage on the last `lookback` rows (annualised). Missing returns -> 0."""
    X = np.where(np.isfinite(R[-lookback:]), R[-lookback:], 0.0)
    n, p = X.shape
    X = X - X.mean(axis=0)
    S = X.T @ X / max(n - 1, 1)
    sd = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    corr = S / np.outer(sd, sd)
    rbar = (corr.sum() - p) / max(p * (p - 1), 1)
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, sd ** 2)
    # shrinkage intensity (Ledoit & Wolf 2004, constant correlation target)
    Y = X ** 2
    pi_mat = (Y.T @ Y) / n - S ** 2
    pi_hat = pi_mat.sum()
    theta = ((X ** 3).T @ X) / n - np.outer(sd ** 2, np.ones(p)) * S
    rho_hat = np.trace(pi_mat) + rbar * ((np.outer(1 / sd, sd) * theta).sum() - np.trace(np.outer(1 / sd, sd) * theta))
    gamma = ((F - S) ** 2).sum()
    kappa = (pi_hat - rho_hat) / max(gamma, 1e-18)
    delta = float(np.clip(kappa / n, 0.0, 1.0))
    return (delta * F + (1 - delta) * S) * 252.0


# ====================================================================================== pairing helpers
def most_correlated(R: np.ndarray, i: int, candidates: np.ndarray) -> int | None:
    if len(candidates) == 0:
        return None
    x = R[:, i]
    Y = R[:, candidates]
    ok = np.isfinite(x)
    x = x[ok]
    Y = Y[ok]
    Y = np.where(np.isfinite(Y), Y, 0.0)
    if len(x) < 40:
        return None
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=0)
    num = Yc.T @ xc
    den = np.sqrt((Yc ** 2).sum(axis=0) * float(xc @ xc)) + 1e-12
    corr = num / den
    return int(candidates[int(np.argmax(corr))])


def sard_pairs(desc: pd.DataFrame, sectors: np.ndarray, held: list[int], pool: list[int]) -> dict[int, int]:
    """Twin for each held name: the unpaired pool member in the same sector with the minimum sum of absolute rank
    differences over size, momentum, volatility, beta and dividend yield."""
    ranks = desc.rank()
    twins: dict[int, int] = {}
    used: set[int] = set()
    order = sorted(held, key=lambda i: -desc.loc[i, "size"] if i in desc.index else 0.0)   # big names pick first
    pool_set = [j for j in pool if j in desc.index]
    for i in order:
        if i not in desc.index:
            continue
        cands = [j for j in pool_set if j not in used and j != i and sectors[j] == sectors[i]]
        if not cands:
            cands = [j for j in pool_set if j not in used and j != i]
        if not cands:
            continue
        d = (ranks.loc[cands] - ranks.loc[i]).abs().sum(axis=1)
        j = int(d.idxmin())
        twins[i] = j
        used.add(j)
    return twins


# ====================================================================================== the simulator
class Account:
    def __init__(self, cash: float):
        self.cash = float(cash)
        self.lots: list[Lot] = []
        self.last_loss_sale: dict[int, int] = {}     # sym -> row of the last loss sale (no re-buy inside the wash window)
        self.last_buy: dict[int, int] = {}           # sym -> row of the last purchase (no loss sale inside the wash window)
        self.trades = 0
        self.turnover_value = 0.0

    def held(self) -> dict[int, float]:
        q: dict[int, float] = {}
        for lot in self.lots:
            q[lot.sym] = q.get(lot.sym, 0.0) + lot.qty
        return q

    def value(self, px: np.ndarray) -> float:
        return self.cash + sum(lot.qty * px[lot.sym] for lot in self.lots if np.isfinite(px[lot.sym]))

    def buy(self, sym: int, dollars: float, price: float, t: int, whole: bool, min_trade: float, cost_bps: float) -> float:
        if not np.isfinite(price) or price <= 0 or dollars < max(min_trade, 1e-9):
            return 0.0
        qty = np.floor(dollars / price) if whole else dollars / price
        if qty <= 0:
            return 0.0
        spent = qty * price
        fee = spent * cost_bps / 1e4
        if spent + fee > self.cash + 1e-9:
            qty = np.floor((self.cash / (1 + cost_bps / 1e4)) / price) if whole else (self.cash / (1 + cost_bps / 1e4)) / price
            if qty <= 0:
                return 0.0
            spent = qty * price
            fee = spent * cost_bps / 1e4
        self.cash -= spent + fee
        self.lots.append(Lot(sym, float(qty), float(price), t))
        self.last_buy[sym] = t
        self.trades += 1
        self.turnover_value += spent
        return spent

    def sell_lot(self, lot: Lot, price: float, t: int, cost_bps: float, qty: float | None = None) -> tuple[float, float, bool]:
        """Sell (part of) a lot. Returns (proceeds, realised P&L, long_term)."""
        q = lot.qty if qty is None else min(qty, lot.qty)
        proceeds = q * price
        fee = proceeds * cost_bps / 1e4
        pnl = (price - lot.basis) * q - fee
        self.cash += proceeds - fee
        lot.qty -= q
        if lot.qty <= 1e-9:
            self.lots.remove(lot)
        if pnl < 0:
            self.last_loss_sale[lot.sym] = t
        self.trades += 1
        self.turnover_value += proceeds
        return proceeds, pnl, (t - lot.opened) > 365 * 252 / 365.25


def run_window(store, spec: ResearchSpec, progress=None) -> RunResult:
    say = progress or (lambda m: None)
    if spec.approach not in APPROACHES:
        raise ValueError(f"unknown approach {spec.approach}; choose from {list(APPROACHES)}")
    dates = store.dates
    t_start = store.date_pos(f"{spec.start_year}-01-01")
    t_end = min(store.date_pos(f"{spec.end_year}-01-01"), store.n_dates - 1)
    if t_start < LOOKBACK + 5:
        t_start = LOOKBACK + 5
    if t_end - t_start < 250:
        raise ValueError(f"window {spec.start_year}-{spec.end_year} has fewer than a year of data in the store")
    rebal = store.month_ends(t_start, t_end)
    if rebal[0] != t_start:
        rebal = [t_start] + rebal
    sectors = store.sector
    close = store.close
    warnings: list[str] = []

    acct = Account(spec.account_size)
    # ---- concentrated starting position
    conc_sym: int | None = None
    conc_qty0 = 0.0
    if spec.concentrated_pct > 0:
        members0 = np.asarray(store.member[t_start])
        if spec.concentrated_symbol and spec.concentrated_symbol in store.symbols and members0[store.symbols.index(spec.concentrated_symbol)]:
            conc_sym = store.symbols.index(spec.concentrated_symbol)
        else:
            cap0 = store.cap_proxy(t_start)
            conc_sym = int(np.nanargmax(np.where(members0, cap0, np.nan)))
            if spec.concentrated_symbol:
                warnings.append(f"{spec.concentrated_symbol} not an index member at the start; used {store.symbols[conc_sym]}")
        px0 = float(close[t_start, conc_sym])
        dollars = spec.account_size * spec.concentrated_pct
        conc_qty0 = np.floor(dollars / px0) if spec.whole_shares else dollars / px0
        acct.cash -= conc_qty0 * px0
        acct.lots.append(Lot(conc_sym, float(conc_qty0), px0 / (1.0 + spec.concentrated_gain), t_start - 2 * 252))   # long-term lot
    twins: dict[int, int] = {}
    last_pair_year = None

    monthly_rows = []
    daily_val = np.full(store.n_dates, np.nan)
    harvest_st = harvest_lt = gains_st = gains_lt = 0.0
    wash_blocked = 0
    year_realised = 0.0
    cur_year = None
    conc_done_t: int | None = None

    def bench_members(t):
        m = np.asarray(store.member[t]).copy()
        c = np.asarray(close[t], dtype=float)
        return m & np.isfinite(c) & (c > 0)

    for k, t in enumerate(rebal):
        px = np.asarray(close[t], dtype=float)
        members = bench_members(t)
        wb = store.bench_weights(t)
        year = dates[t].year
        if cur_year != year:
            cur_year, year_realised = year, 0.0
        R = _window_returns(store, t)
        harvested_now = 0.0
        realised_now = 0.0
        sold_syms: list[tuple[int, float]] = []          # (sym, proceeds) for the pairs approaches

        # ---- 1. delistings: sell at the last available close
        for lot in list(acct.lots):
            if not np.isfinite(px[lot.sym]):
                last = np.asarray(close[max(t - 10, 0): t + 1, lot.sym], dtype=float)
                last = last[np.isfinite(last)]
                price = float(last[-1]) if len(last) else 0.0
                _, pnl, lt = acct.sell_lot(lot, price, t, spec.cost_bps)
                realised_now += pnl
                if pnl < 0:
                    harvested_now += -pnl
                    if lt:
                        harvest_lt += -pnl
                    else:
                        harvest_st += -pnl
                elif lt:
                    gains_lt += pnl
                else:
                    gains_st += pnl
        value = acct.value(px)

        # ---- 2. harvest
        for lot in list(acct.lots):
            if lot.sym == conc_sym and conc_done_t is None and lot.basis < px[lot.sym]:
                continue
            loss = (lot.basis - px[lot.sym]) * lot.qty
            if loss <= 0:
                continue
            thresh = spec.trigger * (value if spec.trigger_basis == "account" else lot.basis * lot.qty)
            if loss < max(thresh, spec.min_harvest):
                continue
            if t - acct.last_buy.get(lot.sym, -10**9) <= spec.wash_days * 252 / 365.25 and acct.last_buy.get(lot.sym, -1) != lot.opened:
                wash_blocked += 1
                continue
            proceeds, pnl, lt = acct.sell_lot(lot, px[lot.sym], t, spec.cost_bps)
            realised_now += pnl
            harvested_now += -pnl
            if lt:
                harvest_lt += -pnl
            else:
                harvest_st += -pnl
            sold_syms.append((lot.sym, proceeds))
        year_realised += realised_now

        # ---- 3. concentrated unwind: realise gains only as far as this year's realised losses + budget cover them
        if conc_sym is not None and conc_done_t is None:
            conc_lots = [lot for lot in acct.lots if lot.sym == conc_sym]
            if conc_lots:
                lot = conc_lots[0]
                gain_ps = px[conc_sym] - lot.basis
                budget = -year_realised + spec.gain_budget * value * (1.0 if k == 0 else 1.0 / 12.0)
                if gain_ps > 0 and budget > 0:
                    q = min(lot.qty, budget / gain_ps)
                    q = np.floor(q) if spec.whole_shares else q
                    if q > 0:
                        proceeds, pnl, lt = acct.sell_lot(lot, px[conc_sym], t, spec.cost_bps, qty=q)
                        realised_now += pnl
                        year_realised += pnl
                        if lt:
                            gains_lt += pnl
                        else:
                            gains_st += pnl
                elif gain_ps <= 0:
                    proceeds, pnl, lt = acct.sell_lot(lot, px[conc_sym], t, spec.cost_bps)
                    realised_now += pnl
                    year_realised += pnl
                    harvested_now += max(-pnl, 0.0)
                conc_w = sum(lot.qty for lot in acct.lots if lot.sym == conc_sym) * px[conc_sym] / max(acct.value(px), 1e-9)
                if conc_w < 0.05:
                    conc_done_t = t
            else:
                conc_done_t = t

        # ---- 4. reinvest
        value = acct.value(px)
        held = acct.held()
        wash_no_buy = {s for s, ts in acct.last_loss_sale.items() if t - ts <= spec.wash_days * 252 / 365.25}
        eligible = members.copy()
        for s in wash_no_buy:
            eligible[s] = False
        if conc_sym is not None:
            eligible[conc_sym] = False
        if spec.approach == "twin_baskets" and (last_pair_year != year or k == 0):
            desc = descriptors(store, t, members, R)
            target = list(held) if held else _sector_stratified(wb, members, sectors, spec.basket_size, exclude=conc_sym)
            tset = set(target)
            pool = [j for j in np.where(eligible)[0] if j not in held and j not in tset]
            twins = sard_pairs(desc, sectors, target, pool)
            last_pair_year = year
        if k == 0 or not held or (acct.cash > 0.02 * value and spec.approach == "optimizer"):
            _construct(store, acct, spec, t, px, wb, members, eligible, sectors, R, exclude=conc_sym, twins=twins)
        else:
            _reinvest(store, acct, spec, t, px, wb, members, eligible, sectors, R, sold_syms, twins, exclude=conc_sym)

        # ---- 5. record
        held = acct.held()
        value = acct.value(px)
        w_port = np.zeros(store.symbols.__len__())
        for s, q in held.items():
            if np.isfinite(px[s]):
                w_port[s] = q * px[s] / max(value, 1e-9)
        te_fc = _forecast_te(R, w_port, wb, spec.cov_lookback, members)
        conc_w = float(w_port[conc_sym]) if conc_sym is not None else 0.0
        monthly_rows.append({"date": dates[t], "value": value, "cash": acct.cash, "harvested": harvested_now, "realised": realised_now,
                             "n_names": len(held), "te_forecast": te_fc, "conc_weight": conc_w,
                             "active_sector_max": _max_sector_active(w_port, wb, sectors, members),
                             "turnover_value": acct.turnover_value, "trades": acct.trades})
        acct.turnover_value = 0.0

        # ---- daily values until the next month-end
        t_next = rebal[k + 1] if k + 1 < len(rebal) else t_end
        if held:
            idx = np.array(list(held))
            qty = np.array([held[s] for s in idx])
            C = np.asarray(close[t: t_next + 1, idx], dtype=float)
            D = np.asarray(store.dividend[t: t_next + 1, idx], dtype=float)
            C = pd.DataFrame(C).ffill().fillna(0.0).values
            divs = np.cumsum((D * qty).sum(axis=1))
            divs = divs - divs[0]
            daily_val[t: t_next + 1] = C @ qty + acct.cash + divs
            acct.cash += float(divs[-1])
        else:
            daily_val[t: t_next + 1] = acct.cash
        if progress and (k % 12) == 0:
            say(f"  {dates[t].date()} value {value:,.0f} harvested to date {harvest_st + harvest_lt:,.0f} names {len(held)}")

    # ====================================================================== metrics
    monthly = pd.DataFrame(monthly_rows).set_index("date")
    dv = pd.Series(daily_val[t_start: t_end + 1], index=dates[t_start: t_end + 1]).ffill()
    port_r = dv.pct_change().fillna(0.0)
    idx_r = pd.Series(store.index_returns()[t_start: t_end + 1], index=dv.index)
    idx_r.iloc[0] = 0.0
    active = port_r - idx_r
    years = max(len(dv) / 252.0, 1e-9)
    daily = pd.DataFrame({"value": dv, "index": store.index_tr[t_start: t_end + 1] / store.index_tr[t_start] * spec.account_size, "active": active})
    harvested_total = float(monthly["harvested"].sum())
    trailing = monthly["harvested"].rolling(12).sum() / monthly["value"]
    alive = trailing[trailing >= spec.ossification_yield]
    life_months = int((alive.index[-1] - monthly.index[0]).days / 30.44) + 1 if len(alive) else 0
    cum = monthly["harvested"].cumsum()
    half = cum[cum >= 0.5 * harvested_total]
    half_life = int((half.index[0] - monthly.index[0]).days / 30.44) + 1 if len(half) and harvested_total > 0 else None
    by_year = monthly["harvested"].groupby(monthly.index.year).sum()
    te_year = active.groupby(active.index.year).std() * np.sqrt(252)
    px_end = np.asarray(close[t_end], dtype=float)
    unreal = sum((px_end[lot.sym] - lot.basis) * lot.qty for lot in acct.lots if np.isfinite(px_end[lot.sym]))
    conc_months = None
    if conc_sym is not None:
        conc_months = int((dates[conc_done_t] - dates[t_start]).days / 30.44) if conc_done_t is not None else None
    metrics = {
        "start": str(dates[t_start].date()), "end": str(dates[t_end].date()), "years": round(years, 2),
        "harvested_total": harvested_total, "harvested_pct_of_start": harvested_total / spec.account_size,
        "harvested_per_year_pct": harvested_total / spec.account_size / years,
        "harvested_short_term": harvest_st, "harvested_long_term": harvest_lt, "gains_realised": gains_st + gains_lt,
        "net_realised": gains_st + gains_lt - harvest_st - harvest_lt,
        "tax_value_of_losses": harvest_st * spec.st_rate + harvest_lt * spec.lt_rate,
        "tax_value_pct_of_start": (harvest_st * spec.st_rate + harvest_lt * spec.lt_rate) / spec.account_size,
        "harvest_life_months": life_months, "harvest_half_life_months": half_life,
        "months_with_harvest": int((monthly["harvested"] > 0).sum()), "months": int(len(monthly)),
        "te_realised": float(active.std() * np.sqrt(252)), "te_forecast_avg": float(monthly["te_forecast"].mean()),
        "te_realised_max_year": float(te_year.max()) if len(te_year) else np.nan,
        "excess_return_annual": float(((1 + port_r).prod() ** (1 / years) - 1) - ((1 + idx_r).prod() ** (1 / years) - 1)),
        "portfolio_cagr": float((1 + port_r).prod() ** (1 / years) - 1), "index_cagr": float((1 + idx_r).prod() ** (1 / years) - 1),
        "turnover_annual": float(monthly["turnover_value"].sum() / max(monthly["value"].mean(), 1e-9) / years / 2),
        "trades": int(acct.trades), "wash_blocked": int(wash_blocked),
        "names_avg": float(monthly["n_names"].mean()), "names_min": int(monthly["n_names"].min()), "names_end": int(monthly["n_names"].iloc[-1]),
        "cash_pct_avg": float((monthly["cash"] / monthly["value"]).mean()),
        "sector_active_max": float(monthly["active_sector_max"].max()),
        "unrealised_gain_end_pct": float(unreal / max(monthly["value"].iloc[-1], 1e-9)),
        "end_value": float(monthly["value"].iloc[-1]),
        "conc_months_to_diversify": conc_months, "conc_weight_end": float(monthly["conc_weight"].iloc[-1]),
        "harvested_by_year": {str(k): float(v) for k, v in by_year.items()},
        "te_by_year": {str(k): float(v) for k, v in te_year.items()},
    }
    return RunResult(spec=spec, metrics=metrics, monthly=monthly, daily=daily, warnings=warnings)


# ====================================================================================== construction / reinvestment
def _sector_stratified(wb: np.ndarray, members: np.ndarray, sectors: np.ndarray, n: int, exclude: int | None = None) -> list[int]:
    """Largest names per sector in proportion to sector weight (the non-optimizer construction)."""
    idx = np.where(members & (wb > 0))[0]
    if exclude is not None:
        idx = idx[idx != exclude]
    n = min(n, len(idx))
    secs = pd.Series(sectors[idx], index=idx)
    sw = pd.Series(wb[idx], index=idx).groupby(secs).sum()
    alloc = (sw / sw.sum() * n).round().astype(int)
    while alloc.sum() > n:
        alloc[alloc.idxmax()] -= 1
    while alloc.sum() < n:
        alloc[(sw / (alloc + 1)).idxmax()] += 1
    chosen: list[int] = []
    for sec, cnt in alloc.items():
        pool = [i for i in idx if secs[i] == sec]
        pool.sort(key=lambda i: -wb[i])
        chosen += pool[: int(cnt)]
    return chosen


def _sector_neutral_weights(chosen: list[int], wb: np.ndarray, sectors: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Weights on `chosen` = index weight within the name's sector, scaled to the sector's index weight."""
    w = np.zeros(len(wb))
    if not chosen:
        return w
    idx = np.where(members & (wb > 0))[0]
    sec_w = pd.Series(wb[idx], index=idx).groupby(pd.Series(sectors[idx], index=idx)).sum()
    ch = np.array(chosen)
    for sec, sw in sec_w.items():
        mine = ch[sectors[ch] == sec]
        if len(mine):
            w[mine] = wb[mine] / wb[mine].sum() * sw
    return w / w.sum() if w.sum() > 0 else w


def _max_sector_active(w: np.ndarray, wb: np.ndarray, sectors: np.ndarray, members: np.ndarray) -> float:
    idx = np.where(members | (w > 0))[0]
    if len(idx) == 0:
        return 0.0
    act = pd.Series(w[idx] - wb[idx], index=idx).groupby(pd.Series(sectors[idx], index=idx)).sum()
    return float(act.abs().max()) if len(act) else 0.0


def _forecast_te(R: np.ndarray, w: np.ndarray, wb: np.ndarray, lookback: int, members: np.ndarray, cap: int = 400) -> float:
    """Ex-ante TE from the calibrated covariance on held names + the largest index names (active weight on the rest is
    lumped into their combined position, which understates TE slightly)."""
    act = w - wb
    idx = np.where((w > 0) | (np.abs(act) > 0))[0]
    if len(idx) > cap:
        keep = idx[np.argsort(-np.abs(act[idx]))[:cap]]
        idx = np.sort(keep)
    if len(idx) < 2:
        return 0.0
    S = lw_covariance(R[:, idx], lookback)
    a = act[idx]
    return float(np.sqrt(max(a @ S @ a, 0.0)))


def _construct(store, acct: Account, spec: ResearchSpec, t: int, px, wb, members, eligible, sectors, R, exclude=None, twins=None) -> None:
    """Deploy the cash into a basket (initial construction, or a large cash balance under the optimizer)."""
    value = acct.value(px)
    held = acct.held()
    w_now = np.zeros(len(wb))
    for s, q in held.items():
        if np.isfinite(px[s]):
            w_now[s] = q * px[s] / value
    cash_w = acct.cash / value
    if cash_w <= 0.001:
        return
    if spec.approach == "optimizer":
        w_target = _optimize_buys(store, spec, t, px, wb, members, eligible, sectors, R, w_now, cash_w, exclude)
    else:
        n_new = max(spec.basket_size - len([s for s in held if s != exclude]), 0)
        chosen = [s for s in held if s != exclude]
        if n_new > 0:
            extra = _sector_stratified(wb * eligible, members & eligible, sectors, n_new + len(chosen), exclude=exclude)
            chosen = list(dict.fromkeys(chosen + [s for s in extra if s not in held]))[: spec.basket_size]
        w_target = _sector_neutral_weights(chosen, wb, sectors, members)          # whole-book target; buys = positive gaps
        if exclude is not None:
            w_target[exclude] = 0.0
    buys = np.clip(w_target - w_now, 0, None)
    buys[~eligible] = 0.0
    if buys.sum() <= 0:
        return
    buys = buys / buys.sum() * acct.cash * 0.995
    order = np.argsort(-buys)
    for s in order:
        if buys[s] < spec.min_trade:
            break
        acct.buy(int(s), float(buys[s]), float(px[s]), t, spec.whole_shares, spec.min_trade, spec.cost_bps)


def _reinvest(store, acct: Account, spec: ResearchSpec, t: int, px, wb, members, eligible, sectors, R, sold, twins, exclude=None) -> None:
    held = acct.held()
    if spec.approach == "optimizer":
        value = acct.value(px)
        if acct.cash / value > 0.002:
            _construct(store, acct, spec, t, px, wb, members, eligible, sectors, R, exclude=exclude)
        return
    # pairs / twins: one replacement per harvested name, funded by its proceeds
    for sym, proceeds in sold:
        if proceeds < spec.min_trade:
            continue
        cand_mask = eligible.copy()
        for s in held:
            cand_mask[s] = False
        cand_mask[sym] = False
        rep = None
        if spec.approach == "twin_baskets":
            tw = twins.get(sym)
            if tw is not None and cand_mask[tw]:
                rep = tw
            else:
                # the twin is held or blocked: reverse lookup (we hold the twin, buy back the original later) or fall back
                rev = {v: k for k, v in twins.items()}
                orig = rev.get(sym)
                if orig is not None and cand_mask[orig]:
                    rep = orig
        if rep is None:
            cands = np.where(cand_mask & (sectors == sectors[sym]))[0] if spec.approach in ("pairs_sector", "twin_baskets") else np.where(cand_mask)[0]
            if len(cands) == 0:
                cands = np.where(cand_mask)[0]
            rep = most_correlated(R, sym, cands)
        if rep is None:
            continue
        spent = acct.buy(int(rep), min(proceeds, acct.cash), float(px[rep]), t, spec.whole_shares, spec.min_trade, spec.cost_bps)
        if spent > 0:
            held[int(rep)] = held.get(int(rep), 0.0) + spent / px[rep]
            if spec.approach == "twin_baskets":
                twins[int(rep)] = sym if sym not in twins.values() else twins.get(int(rep), sym)
    # leftover cash (dividends, concentrated-unwind proceeds, rounding): top up the most underweight names
    value = acct.value(px)
    if acct.cash / value > 0.01:
        w_now = np.zeros(len(wb))
        for s, q in held.items():
            if np.isfinite(px[s]):
                w_now[s] = q * px[s] / value
        room = max(spec.basket_size - len(held), 0)
        gap = np.where(eligible, wb - w_now, 0.0)
        gap[list(held)] = np.clip(gap[list(held)], 0, None) if held else 0.0
        new_names = [s for s in np.argsort(-gap) if s not in held][:room]
        keep = set(held) | set(int(s) for s in new_names)
        gap = np.array([g if i in keep else 0.0 for i, g in enumerate(gap)])
        gap = np.clip(gap, 0, None)
        if gap.sum() > 0:
            buys = gap / gap.sum() * acct.cash * 0.995
            for s in np.argsort(-buys):
                if buys[s] < spec.min_trade:
                    break
                acct.buy(int(s), float(buys[s]), float(px[s]), t, spec.whole_shares, spec.min_trade, spec.cost_bps)


def _optimize_buys(store, spec: ResearchSpec, t: int, px, wb, members, eligible, sectors, R, w_now, cash_w, exclude) -> np.ndarray:
    """Minimum-TE buy list: total weights w >= w_now (buys only), sum(w) = 1, sector band, factor bands, name cap."""
    held_idx = np.where(w_now > 0)[0]
    room = max(spec.basket_size - len([i for i in held_idx if i != exclude]), 0)
    cand = np.where(eligible & (wb > 0))[0]
    cand = cand[np.argsort(-wb[cand])][: max(2 * spec.basket_size, 60)]
    idx = np.unique(np.concatenate([held_idx, cand]))
    if exclude is not None:
        idx = idx[idx != exclude] if exclude not in held_idx else idx
    S = lw_covariance(R[:, idx], spec.cov_lookback)
    n = len(idx)
    wb_i = wb[idx]
    w0 = w_now[idx]
    buyable = np.array([bool(eligible[i]) and i != exclude for i in idx])
    cap = np.maximum(3.0 * wb_i, 0.02)
    wv = None
    if _osqp is not None:
        wv = _solve_qp_osqp(S, wb_i, w0, buyable, np.maximum(cap, w0), sectors[idx], members, wb, sectors, spec, store, t, idx, R)
    if wv is None:
        # cvxpy fallback (only imported when OSQP is unavailable or fails)
        vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
        F = vecs * np.sqrt(np.clip(vals, 1e-10, None))
        w = cp.Variable(n)
        active = w - wb_i
        cons = [w >= w0, cp.sum(w) == 1.0, w[~buyable] == w0[~buyable], w <= np.maximum(cap, w0)]
        if spec.sector_band is not None:
            secs = pd.Series(sectors[idx], index=range(n))
            for sec in secs.unique():
                mask = (secs == sec).values.astype(float)
                sec_bench = float(wb[members & (sectors == sec)].sum())
                cons.append(cp.abs(mask @ w - sec_bench) <= spec.sector_band)
        if spec.factor_alignment:
            desc = descriptors(store, t, members, R)
            Z = desc.reindex(idx).fillna(0.0)
            zb = (desc.mul(wb[desc.index], axis=0).sum() / max(wb[desc.index].sum(), 1e-9))
            for f in FACTORS:
                cons.append(cp.abs(Z[f].values @ w - float(zb[f])) <= spec.factor_band)
        obj = cp.sum_squares(F.T @ active)
        prob = cp.Problem(cp.Minimize(obj), cons)
        for sv in ("CLARABEL", "OSQP", "SCS"):
            try:
                prob.solve(solver=sv)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception:
                continue
        if w.value is None:
            prob = cp.Problem(cp.Minimize(obj), [w >= w0, cp.sum(w) == 1.0, w[~buyable] == w0[~buyable], w <= np.maximum(cap, w0)])
            prob.solve(solver="CLARABEL")
            if w.value is None:
                return w_now
        wv = np.asarray(w.value).ravel()
    wv = np.clip(wv, 0, None)
    buys = wv - w0
    new = [i for i in range(n) if w0[i] <= 0 and buys[i] > 1e-6]
    if len(new) > room:
        drop = sorted(new, key=lambda i: buys[i])[: len(new) - room]
        for i in drop:
            buys[i] = 0.0
    total = np.zeros(len(wb))
    total[idx] = w0 + np.clip(buys, 0, None)
    return total


# ====================================================================================== direct QP (OSQP)
try:
    import osqp as _osqp
    from scipy import sparse as _sp
except Exception:  # pragma: no cover - optional
    _osqp = None


_STAGE_HINT: dict[int, int] = {}      # id(spec) -> relaxation stage that succeeded last time (starts one stage tighter each month)


def _solve_qp_osqp(S, wb_i, w0, buyable, cap, sec_i, members, wb, sectors, spec, store, t, idx, R):
    """min (w-wb)'S(w-wb)  s.t. sum w = 1, w0 <= w <= cap (fixed where not buyable), sector band, factor bands.
    Built as a sparse QP and handed straight to OSQP: about 10 ms for 300 names. Returns None if OSQP cannot solve."""
    n = len(w0)
    P = _sp.csc_matrix(2.0 * S)
    q = -2.0 * S @ wb_i
    rows, lo, hi = [], [], []

    def add(row, lower, upper_):
        rows.append(np.atleast_2d(np.asarray(row, dtype=float)))
        lo.append(np.atleast_1d(np.asarray(lower, dtype=float)))
        hi.append(np.atleast_1d(np.asarray(upper_, dtype=float)))

    add(np.ones((1, n)), 1.0, 1.0)
    upper = np.where(buyable, cap, w0)
    add(np.eye(n), w0, np.maximum(upper, w0))
    if spec.sector_band is not None:
        for sec in pd.unique(sec_i):
            mask = (sec_i == sec).astype(float)
            b = float(wb[members & (sectors == sec)].sum())
            add(mask[None, :], b - spec.sector_band, b + spec.sector_band)
    if spec.factor_alignment:
        desc = descriptors(store, t, members, R)
        Z = desc.reindex(idx).fillna(0.0)
        wsum = max(wb[desc.index].sum(), 1e-9)
        zb = desc.mul(wb[desc.index], axis=0).sum() / wsum
        for f in FACTORS:
            add(Z[f].values[None, :], float(zb[f]) - spec.factor_band, float(zb[f]) + spec.factor_band)
    n_sec = len(rows) - 2 - (len(FACTORS) if spec.factor_alignment else 0)
    A = _sp.csc_matrix(np.vstack(rows))
    lb = np.concatenate(lo)
    ub = np.concatenate(hi)

    def attempt(lo_v, hi_v, A_):
        prob = _osqp.OSQP()
        try:
            prob.setup(P=P, q=q, A=A_, l=lo_v, u=hi_v, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, polish=True)
            prob.warm_start(x=np.maximum(w0, wb_i / max(wb_i.sum(), 1e-9) * (1 - w0.sum()) + w0))
            res = prob.solve()
        except Exception:
            return None
        if res.x is None or "solved" not in getattr(res.info, "status", "").lower():
            return None
        return np.asarray(res.x, dtype=float)

    # staged relaxation: bands as set -> bands x2 -> bands x4 -> sector only x4 -> box + budget only. The stage that worked
    # last month is tried first (one stage tighter, so the bands re-tighten when they can), which avoids repeated failures.
    band_rows = np.arange(1 + n, len(lb))
    mid = 0.5 * (lb[band_rows] + ub[band_rows])
    half = 0.5 * (ub[band_rows] - lb[band_rows])
    stages = (1.0, 2.0, 4.0)
    key = id(spec)
    first = max(_STAGE_HINT.get(key, 0) - 1, 0)
    for si in range(first, len(stages)):
        scale = stages[si]
        lo_v, hi_v = lb.copy(), ub.copy()
        lo_v[band_rows] = mid - half * scale
        hi_v[band_rows] = mid + half * scale
        x = attempt(lo_v, hi_v, A)
        if x is not None:
            _STAGE_HINT[key] = si
            return x
    _STAGE_HINT[key] = len(stages)
    keep = 1 + n + n_sec
    lo_v, hi_v = lb[:keep].copy(), ub[:keep].copy()
    if n_sec:
        lo_v[1 + n:] = mid[:n_sec] - half[:n_sec] * 4.0
        hi_v[1 + n:] = mid[:n_sec] + half[:n_sec] * 4.0
    x = attempt(lo_v, hi_v, _sp.csc_matrix(np.vstack(rows[:2 + n_sec])))
    if x is not None:
        return x
    return attempt(lb[: 1 + n], ub[: 1 + n], _sp.csc_matrix(np.vstack(rows[:2])))
