"""Harvest optimizer: wash-safe sell/replace trade list that maximises after-tax harvested loss subject to
tracking-error, factor-drift, sector-drift, turnover and minimum-trade constraints.

Formulation (convex QP, cvxpy)
------------------------------
Variables   s_l in [0,1]  fraction of candidate loss lot l to sell (taxable accounts only)
            b_j >= 0      dollars to buy of replacement candidate j
Post-trade  h' = h - S s + B b        (dollar holdings by symbol; S, B are incidence matrices)
            w' = h' / V,  active = w' - w_bench
Objective   maximise  w_tax * sum_l benefit_l s_l / V
                      - w_te * (TE(active)/te_budget)^2
                      - w_fac * ||style_drift / drift_budget||^2
                      - cost_bps * turnover
Constraints cash neutrality (sum B b = sum S s, tolerance), no shorts, per-name max weight,
            optional hard TE cap, sector drift bounds, turnover budget.
Minimum trade size is enforced by a second pass: sub-threshold trades are fixed at zero and the QP re-solved.

Wash-sale safety is a *pre-filter*, never a soft term: candidate sells that fail `screen_proposed_sale` and
candidate buys that fail `screen_proposed_buy` never enter the problem. A final post-solve re-screen of the
actual trade list is asserted before results are returned.

This module is AI-editable (ai/registry.py). Keep `HarvestConfig`, `HarvestInputs`, `HarvestResult` stable.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..data.substitutes import SubstituteMap
from ..lazy import lazy_module
from ..risk.model import FittedRiskModel
from ..tax.lots import Lot
from ..tax.rates import TaxProfile
from ..tax.washsale import (
    Acquisition,
    LossSale,
    SubstantiallyIdentical,
    screen_proposed_buy,
    screen_proposed_sale,
)

cp = lazy_module("cvxpy")          # imported on first use (saves ~1.7 s at launch)
log = logging.getLogger(__name__)

PRIORITIES = ("tax", "tracking_error", "factor_neutrality")


@dataclass
class HarvestConfig:
    mode: str = "opportunistic"                  # opportunistic | full_rebalance
    priority: tuple[str, str, str] = ("tax", "tracking_error", "factor_neutrality")
    priority_weights: tuple[float, float, float] = (1.0, 0.3, 0.1)   # weight for 1st/2nd/3rd priority
    te_budget: float = 0.01                       # annualised, fraction (1% = 0.01)
    te_hard: bool = False
    factor_drift_budget: float = 0.10             # style z-score drift per factor treated as "one unit"
    factor_drift_hard: float | None = None        # optional hard cap per style factor
    sector_drift_max: float = 0.02                # max abs change in any sector weight
    turnover_max: float = 0.30                    # fraction of portfolio value
    max_position_weight: float = 0.15
    min_trade_value: float = 500.0
    min_loss_value: float = 250.0                 # ignore lots with smaller unrealised loss
    cost_bps: float = 5.0
    cash_tolerance: float = 0.005                 # |buys - sells| <= tol * V
    max_replacements_per_sell: int = 3
    peer_min_correlation: float = 0.55
    tax_horizon_years: float = 10.0
    target_loss: float | None = None              # stop harvesting beyond this $ of loss (soft)
    solver: str = "CLARABEL"

    def weights(self) -> dict[str, float]:
        return dict(zip(self.priority, self.priority_weights, strict=True))


@dataclass
class HarvestInputs:
    as_of: date
    lots: list[Lot]                               # all open lots of the entity (all accounts)
    prices: pd.Series                             # last price by symbol
    model: FittedRiskModel
    benchmark: pd.Series                          # weights by symbol
    tax: TaxProfile
    groups: SubstantiallyIdentical
    substitutes: SubstituteMap
    acquisitions: list[Acquisition]               # all past + scheduled acquisitions of the entity
    recent_loss_sales: list[LossSale]             # realised loss sales in the last 30 days
    returns: pd.DataFrame | None = None           # date x symbol daily returns for correlation of substitutes
    securities: pd.DataFrame | None = None        # indexed by symbol (gics_* columns) for peer search
    universe: list[str] | None = None             # buyable universe (defaults to model symbols)
    sellable_account_types: tuple[str, ...] = ("taxable",)


@dataclass
class HarvestResult:
    trades: pd.DataFrame
    blocked: pd.DataFrame
    replacements: pd.DataFrame
    summary: dict
    exposures_before: pd.Series
    exposures_after: pd.Series
    te_decomp_before: pd.DataFrame
    te_decomp_after: pd.DataFrame
    sector_before: pd.Series
    sector_after: pd.Series
    config: HarvestConfig
    solver_status: str
    weights_before: pd.Series = field(default_factory=pd.Series)
    weights_after: pd.Series = field(default_factory=pd.Series)

    def to_dict(self) -> dict:
        return {"summary": self.summary, "config": asdict(self.config), "solver_status": self.solver_status}


# ====================================================================================== main entry
def run_harvest(inp: HarvestInputs, cfg: HarvestConfig) -> HarvestResult:
    px = inp.prices.astype(float)
    model = inp.model
    known = set(model.symbols)

    # ---- holdings by symbol (all accounts; TE is measured on the whole entity) --------------------
    lots = [lot for lot in inp.lots if lot.quantity_open > 1e-9 and lot.symbol in px.index]
    hold = pd.Series(0.0, index=sorted({lot.symbol for lot in lots}))
    for lot in lots:
        hold[lot.symbol] += lot.quantity_open * px[lot.symbol]
    unknown = [s for s in hold.index if s not in known]
    if unknown:
        log.warning("holdings not in risk model (treated as cash-like, excluded from TE): %s", unknown)
    V = float(hold.sum())
    if V <= 0:
        raise ValueError("portfolio has no market value")

    # ---- candidate sells ------------------------------------------------------------------------
    cand_lots: list[Lot] = []
    blocked_rows: list[dict] = []
    sell_info: list[dict] = []
    for lot in lots:
        if lot.account_type not in inp.sellable_account_types or lot.symbol not in known:
            continue
        ug = lot.unrealized_gain(px[lot.symbol])
        if ug >= -cfg.min_loss_value:
            continue
        loss = -ug
        det = screen_proposed_sale(lot.assetid, lot.symbol, lot.account_id, inp.as_of, lot.quantity_open, loss,
                                   inp.acquisitions, inp.groups, lot_id=lot.id)
        term = lot.term_at(inp.as_of)
        row = {"lot_id": lot.id, "account_id": lot.account_id, "symbol": lot.symbol, "quantity": lot.quantity_open,
               "price": px[lot.symbol], "market_value": lot.market_value(px[lot.symbol]), "loss": loss, "term": term,
               "wash_status": det.status, "wash_explanation": det.explanation}
        if det.status != "SAFE":
            blocked_rows.append(row)
            continue
        benefit = inp.tax.tax_alpha(loss, term, horizon_years=cfg.tax_horizon_years)
        row["tax_benefit"] = inp.tax.benefit_of_loss(loss, term)
        row["tax_alpha"] = benefit
        cand_lots.append(lot)
        sell_info.append(row)
    blocked = pd.DataFrame(blocked_rows)

    # ---- candidate buys -----------------------------------------------------------------------------
    sell_symbols = sorted({r["symbol"] for r in sell_info})
    proposed_sales = [LossSale(next(lot.assetid for lot in cand_lots if lot.symbol == s), s, 0, inp.as_of, 1.0, 1.0)
                      for s in sell_symbols]
    blocked_sales = inp.recent_loss_sales + proposed_sales
    repl_rows: list[dict] = []
    buy_syms: list[str] = []
    replaces: dict[str, list[str]] = {}
    universe = [s for s in (inp.universe or model.symbols) if s in known and s in px.index]
    held_groups = {inp.groups.group_of(lot.assetid) for lot in lots}
    for s in sell_symbols:
        cands = _replacement_candidates(s, inp, cfg, universe, held_groups)
        kept = []
        for c, corr, why in cands:
            aid = _assetid_for(c, inp)
            scr = screen_proposed_buy(aid, c, inp.as_of, blocked_sales, inp.groups) if aid is not None else None
            status = scr.status if scr else "UNKNOWN_ASSETID"
            repl_rows.append({"sold_symbol": s, "candidate": c, "correlation": corr, "source": why,
                              "wash_status": status, "wash_explanation": scr.explanation if scr else "assetid unresolved"})
            if status == "SAFE":
                kept.append(c)
            if len(kept) >= cfg.max_replacements_per_sell:
                break
        replaces[s] = kept
        for c in kept:
            if c not in buy_syms:
                buy_syms.append(c)
    if cfg.mode == "full_rebalance":
        # Full-rebalance mode may also top up existing holdings and buy benchmark / model-portfolio constituents
        # (e.g. when the benchmark is a saved basket), provided each purchase is wash-safe.
        bench_names = [s for s in inp.benchmark.index if inp.benchmark[s] > 0]
        for s in list(hold.index) + bench_names:
            if s in known and s in px.index and s not in sell_symbols and s not in buy_syms:
                aid = _assetid_for(s, inp)
                if aid is not None and screen_proposed_buy(aid, s, inp.as_of, blocked_sales, inp.groups).status == "SAFE":
                    buy_syms.append(s)
    replacements = pd.DataFrame(repl_rows)

    # ---- assemble the QP ------------------------------------------------------------------------------
    syms = sorted(set(hold.index) | set(buy_syms) | set(inp.benchmark.index))
    syms = [s for s in syms if s in known]
    n = len(syms)
    pos = {s: i for i, s in enumerate(syms)}
    h = np.array([hold.get(s, 0.0) for s in syms])
    wb = np.array([inp.benchmark.get(s, 0.0) for s in syms])
    wb = wb / wb.sum() if wb.sum() > 0 else wb
    X = model.exposures.loc[syms]
    F = model.factor_cov.values
    D = model.specific_var.loc[syms].values
    L = np.linalg.cholesky(F + 1e-12 * np.eye(len(F)))
    style_cols = [c for c in model.factors if c in model.spec.styles]
    sector_cols = [c for c in model.factors if c.startswith(("sec:", "ind:"))]
    Xs = X[style_cols].values if style_cols else np.zeros((n, 0))
    Xsec = X[sector_cols].values if sector_cols else np.zeros((n, 0))

    nl, nb = len(cand_lots), len(buy_syms)
    S = np.zeros((n, nl))
    for j, lot in enumerate(cand_lots):
        S[pos[lot.symbol], j] = lot.quantity_open * px[lot.symbol]
    B = np.zeros((n, nb))
    for j, s in enumerate(buy_syms):
        B[pos[s], j] = 1.0
    benefit = np.array([r["tax_alpha"] for r in sell_info])
    wts = cfg.weights()

    def solve(fixed_zero_s: set[int], fixed_zero_b: set[int]) -> tuple[np.ndarray, np.ndarray, str]:
        s = cp.Variable(nl) if nl else None
        b = cp.Variable(nb) if nb else None
        hp = h.copy()
        terms = []
        cons = []
        sold = 0
        bought = 0
        if nl:
            hp = hp - S @ s
            cons += [s >= 0, s <= 1]
            for j in fixed_zero_s:
                cons.append(s[j] == 0)
            sold = cp.sum(S @ s)
            terms.append(wts["tax"] * (benefit @ s) / V)
        if nb:
            hp = hp + B @ b
            cons += [b >= 0]
            for j in fixed_zero_b:
                cons.append(b[j] == 0)
            bought = cp.sum(b)
            if cfg.mode == "opportunistic" and nl:
                # Minimal-disruption mode: the replacements for a harvested name may together absorb at most
                # 125% of that name's sale proceeds, and a candidate may only be bought as a replacement.
                # This stops the optimizer from using replacements to rebalance the whole book.
                for sold_sym, cands in replaces.items():
                    lots_idx = [k for k, lot in enumerate(cand_lots) if lot.symbol == sold_sym]
                    buy_idx = [j for j, sym in enumerate(buy_syms) if sym in cands]
                    if lots_idx and buy_idx:
                        cons.append(sum(b[j] for j in buy_idx) <= 1.25 * sum(S[:, k].sum() * s[k] for k in lots_idx))
                for j, sym in enumerate(buy_syms):
                    if not any(sym in c for c in replaces.values()):
                        cons.append(b[j] == 0)
        w = hp / V
        active = w - wb
        te2 = cp.sum_squares(L.T @ (X.values.T @ active)) + cp.sum_squares(cp.multiply(np.sqrt(D), active))
        terms.append(-wts["tracking_error"] * te2 / (cfg.te_budget ** 2))
        if style_cols:
            drift = Xs.T @ (w - h / V)
            terms.append(-wts["factor_neutrality"] * cp.sum_squares(drift) / (cfg.factor_drift_budget ** 2))
            if cfg.factor_drift_hard is not None:
                cons.append(cp.abs(drift) <= cfg.factor_drift_hard)
        turnover = (sold + bought) / V if (nl or nb) else 0
        if nl or nb:
            terms.append(-(cfg.cost_bps / 1e4) * turnover)
            cons.append(turnover <= cfg.turnover_max)
            cons.append(cp.abs(bought - sold) <= cfg.cash_tolerance * V)
        cons.append(w <= cfg.max_position_weight + 1e-9 + np.maximum(h / V - cfg.max_position_weight, 0))
        cons.append(hp >= -1e-6)
        if sector_cols:
            sdrift = Xsec.T @ (w - h / V)
            cons.append(cp.abs(sdrift) <= cfg.sector_drift_max)
        if cfg.target_loss is not None and nl:
            losses = np.array([r["loss"] for r in sell_info])
            cons.append(losses @ s <= cfg.target_loss)
        te_cap = [te2 <= cfg.te_budget ** 2] if cfg.te_hard else []
        prob = cp.Problem(cp.Maximize(sum(terms)), cons + te_cap)
        status = _solve(prob, cfg.solver)
        if status not in ("optimal", "optimal_inaccurate") and te_cap:
            # The hard TE cap is infeasible from this starting portfolio; relax it and flag the result.
            prob = cp.Problem(cp.Maximize(sum(terms)), cons)
            status = _solve(prob, cfg.solver) + "_te_relaxed"
        sv = np.clip(np.asarray(s.value).ravel(), 0, 1) if nl and s.value is not None else np.zeros(nl)
        bv = np.clip(np.asarray(b.value).ravel(), 0, None) if nb and b.value is not None else np.zeros(nb)
        return sv, bv, status

    fz_s: set[int] = set()
    fz_b: set[int] = set()
    sv, bv, status = solve(fz_s, fz_b)
    # second pass: drop sub-minimum trades and re-solve with them fixed at zero
    for _ in range(2):
        small_s = {j for j in range(nl) if 0 < sv[j] * S[:, j].sum() < cfg.min_trade_value}
        small_b = {j for j in range(nb) if 0 < bv[j] < cfg.min_trade_value}
        if not small_s and not small_b:
            break
        fz_s |= small_s
        fz_b |= small_b
        sv, bv, status = solve(fz_s, fz_b)
    sv = np.where(sv * S.sum(axis=0) < cfg.min_trade_value, 0.0, sv)
    bv = np.where(bv < cfg.min_trade_value, 0.0, bv)

    # ---- convert to share quantities ------------------------------------------------------------------
    trade_rows: list[dict] = []
    for j, lot in enumerate(cand_lots):
        if sv[j] <= 1e-9:
            continue
        qty = lot.quantity_open if sv[j] > 0.995 else _round_qty(lot.quantity_open * sv[j], lot.quantity_open)
        if qty <= 0:
            continue
        info = sell_info[j]
        frac = qty / lot.quantity_open
        trade_rows.append({
            "side": "SELL", "account_id": lot.account_id, "symbol": lot.symbol, "assetid": lot.assetid,
            "lot_id": lot.id, "quantity": qty, "est_price": px[lot.symbol], "est_value": qty * px[lot.symbol],
            "realized_gain": -info["loss"] * frac, "term": info["term"], "tax_benefit": info["tax_benefit"] * frac,
            "tax_alpha": info["tax_alpha"] * frac, "wash_status": "SAFE", "wash_explanation": info["wash_explanation"],
            "replacement_for": None, "holding_start": lot.holding_start_date, "basis_per_share": lot.basis_per_share,
        })
    sells_by_symbol: dict[str, float] = {}
    for r in trade_rows:
        sells_by_symbol[r["symbol"]] = sells_by_symbol.get(r["symbol"], 0.0) + r["est_value"]
    default_acct = _default_buy_account(cand_lots, lots)
    for j, s in enumerate(buy_syms):
        if bv[j] <= 0:
            continue
        qty = float(np.floor(bv[j] / px[s]))
        if qty <= 0 or qty * px[s] < cfg.min_trade_value:
            continue
        for_syms = [k for k, v in replaces.items() if s in v and k in sells_by_symbol]
        aid = _assetid_for(s, inp)
        trade_rows.append({
            "side": "BUY", "account_id": default_acct, "symbol": s, "assetid": aid, "lot_id": None, "quantity": qty,
            "est_price": px[s], "est_value": qty * px[s], "realized_gain": None, "term": None, "tax_benefit": None,
            "tax_alpha": None, "wash_status": "SAFE",
            "wash_explanation": screen_proposed_buy(aid, s, inp.as_of, blocked_sales, inp.groups).explanation if aid else "",
            "replacement_for": ", ".join(for_syms) if for_syms else ("rebalance" if cfg.mode == "full_rebalance" else None),
            "holding_start": None, "basis_per_share": None,
        })
    trades = pd.DataFrame(trade_rows)

    # ---- final safety re-screen of the actual trade list ------------------------------------------------
    _assert_wash_safe(trades, inp, cand_lots)

    # ---- before / after analytics ---------------------------------------------------------------------------
    h_after = h.copy()
    for r in trade_rows:
        i = pos[r["symbol"]]
        h_after[i] += r["est_value"] if r["side"] == "BUY" else -r["est_value"]
    w0 = pd.Series(h / V, index=syms)
    w1 = pd.Series(h_after / h_after.sum(), index=syms) if h_after.sum() > 0 else w0
    bench = pd.Series(wb, index=syms)
    dec0 = model.te_decomposition(w0, bench)
    dec1 = model.te_decomposition(w1, bench)
    exp0 = model.portfolio_exposures(w0)
    exp1 = model.portfolio_exposures(w1)
    sec0 = pd.Series(Xsec.T @ w0.values, index=[c.split(':', 1)[1] for c in sector_cols]) if sector_cols else pd.Series(dtype=float)
    sec1 = pd.Series(Xsec.T @ w1.values, index=[c.split(':', 1)[1] for c in sector_cols]) if sector_cols else pd.Series(dtype=float)
    sells = trades[trades["side"] == "SELL"] if not trades.empty else pd.DataFrame()
    buys = trades[trades["side"] == "BUY"] if not trades.empty else pd.DataFrame()
    summary = {
        "as_of": inp.as_of.isoformat(), "mode": cfg.mode, "priority": list(cfg.priority),
        "portfolio_value": V, "n_sells": int(len(sells)), "n_buys": int(len(buys)),
        "sell_value": float(sells["est_value"].sum()) if len(sells) else 0.0,
        "buy_value": float(buys["est_value"].sum()) if len(buys) else 0.0,
        "harvested_loss": float(-sells["realized_gain"].sum()) if len(sells) else 0.0,
        "harvested_loss_st": float(-sells.loc[sells["term"] == "ST", "realized_gain"].sum()) if len(sells) else 0.0,
        "harvested_loss_lt": float(-sells.loc[sells["term"] == "LT", "realized_gain"].sum()) if len(sells) else 0.0,
        "tax_benefit": float(sells["tax_benefit"].sum()) if len(sells) else 0.0,
        "tax_alpha": float(sells["tax_alpha"].sum()) if len(sells) else 0.0,
        "tax_alpha_bps": float(sells["tax_alpha"].sum() / V * 1e4) if len(sells) else 0.0,
        "te_before": dec0.attrs["tracking_error"], "te_after": dec1.attrs["tracking_error"],
        "turnover": float((sells["est_value"].sum() if len(sells) else 0.0) + (buys["est_value"].sum() if len(buys) else 0.0)) / V,
        "n_candidate_lots": nl, "n_blocked_lots": int(len(blocked)), "n_buy_candidates": nb,
        "total_harvestable_loss": float(sum(r["loss"] for r in sell_info)),
        "max_style_drift": float(np.abs((exp1 - exp0).reindex(style_cols)).max()) if style_cols else 0.0,
        "max_sector_drift": float(np.abs(sec1 - sec0).max()) if len(sec0) else 0.0,
        "solver_status": status,
    }
    return HarvestResult(trades=trades, blocked=blocked, replacements=replacements, summary=summary,
                         exposures_before=exp0, exposures_after=exp1, te_decomp_before=dec0, te_decomp_after=dec1,
                         sector_before=sec0, sector_after=sec1, config=cfg, solver_status=status,
                         weights_before=w0, weights_after=w1)


# ====================================================================================== helpers
def _solve(prob: cp.Problem, solver: str) -> str:
    import warnings
    for sv in (solver, "OSQP", "SCS"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=sv, verbose=False)
            if prob.status == "optimal":
                return prob.status
            if prob.status == "optimal_inaccurate" and sv == "SCS":
                return prob.status
        except Exception as e:  # pragma: no cover - solver availability
            log.debug("solver %s could not solve this formulation: %s", sv, e)
    return prob.status or "failed"


def _round_qty(q: float, max_q: float) -> float:
    if abs(max_q - round(max_q)) < 1e-9:      # whole-share lot -> whole-share trade
        return float(min(np.floor(q + 1e-9), max_q))
    return float(min(round(q, 4), max_q))


def _assetid_for(symbol: str, inp: HarvestInputs) -> int | None:
    for lot in inp.lots:
        if lot.symbol == symbol:
            return lot.assetid
    if inp.securities is not None and symbol in inp.securities.index and "assetid" in inp.securities:
        v = inp.securities.loc[symbol, "assetid"]
        return int(v) if pd.notna(v) else None
    return None


def _default_buy_account(cand_lots: list[Lot], lots: list[Lot]) -> int:
    if cand_lots:
        return cand_lots[0].account_id
    taxable = [lot for lot in lots if lot.account_type == "taxable"]
    return (taxable or lots)[0].account_id


def _replacement_candidates(symbol: str, inp: HarvestInputs, cfg: HarvestConfig, universe: list[str],
                            held_groups: set[str]) -> list[tuple[str, float | None, str]]:
    """Ordered (candidate, correlation, source) list: curated substitutes first, then GICS peers by correlation."""
    out: list[tuple[str, float | None, str]] = []
    seen = set()
    corr_fn = _corr_factory(symbol, inp)

    def add(c: str, why: str) -> None:
        if c in seen or c == symbol or c not in inp.model.symbols or c not in inp.prices.index:
            return
        if inp.substitutes.same_group(c, symbol):
            return
        aid = _assetid_for(c, inp)
        if aid is not None and inp.groups.group_of(aid) in held_groups and cfg.mode == "opportunistic":
            pass  # buying more of an existing holding is allowed; just noted
        seen.add(c)
        out.append((c, corr_fn(c), why))

    for c in inp.substitutes.candidates_for(symbol):
        add(c, "curated")
    if inp.securities is not None and symbol in inp.securities.index:
        sec = inp.securities
        for level in ("gics_sub_industry", "gics_industry", "gics_sector"):
            if level not in sec or pd.isna(sec.loc[symbol, level]):
                continue
            peers = [p for p in universe if p in sec.index and sec.loc[p, level] == sec.loc[symbol, level] and p != symbol]
            scored = sorted(((corr_fn(p) or -1.0, p) for p in peers), reverse=True)
            for corr, p in scored[:8]:
                if corr is not None and corr >= cfg.peer_min_correlation:
                    add(p, f"gics:{level[5:]}")
            if len(out) >= cfg.max_replacements_per_sell * 3:
                break
    return out


def _corr_factory(symbol: str, inp: HarvestInputs):
    R = inp.returns
    if R is None or symbol not in R.columns:
        return lambda c: None
    base = R[symbol].dropna().iloc[-252:]

    def corr(c: str) -> float | None:
        if c not in R.columns:
            return None
        other = R[c].reindex(base.index)
        ok = other.notna()
        if ok.sum() < 60:
            return None
        return float(np.corrcoef(base[ok].values, other[ok].values)[0, 1])

    return corr


def _assert_wash_safe(trades: pd.DataFrame, inp: HarvestInputs, cand_lots: list[Lot]) -> None:
    """Belt and braces: re-screen every SELL against acquisitions plus this run's BUYs, and every BUY against
    realised and proposed loss sales. Raises if anything slipped through."""
    if trades.empty:
        return
    acqs = copy.deepcopy(inp.acquisitions)
    buys = trades[trades["side"] == "BUY"]
    for _, r in buys.iterrows():
        acqs.append(Acquisition(int(r["assetid"]) if pd.notna(r["assetid"]) else -1, r["symbol"], int(r["account_id"]),
                                "taxable", inp.as_of, float(r["quantity"]), kind="scheduled_buy"))
    sells = trades[trades["side"] == "SELL"]
    for _, r in sells.iterrows():
        det = screen_proposed_sale(int(r["assetid"]), r["symbol"], int(r["account_id"]), inp.as_of, float(r["quantity"]),
                                   float(-r["realized_gain"]), acqs, inp.groups, lot_id=int(r["lot_id"]))
        if det.status != "SAFE":
            raise RuntimeError(f"wash-sale safety violated for SELL {r['symbol']} lot {r['lot_id']}: {det.explanation}")
    proposed = [LossSale(int(r["assetid"]), r["symbol"], int(r["account_id"]), inp.as_of, float(r["quantity"]),
                         float(-r["realized_gain"])) for _, r in sells.iterrows()]
    for _, r in buys.iterrows():
        if pd.isna(r["assetid"]):
            continue
        scr = screen_proposed_buy(int(r["assetid"]), r["symbol"], inp.as_of, inp.recent_loss_sales + proposed, inp.groups)
        if scr.status != "SAFE":
            raise RuntimeError(f"wash-sale safety violated for BUY {r['symbol']}: {scr.explanation}")
