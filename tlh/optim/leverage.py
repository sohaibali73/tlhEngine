"""Levered-beta construction and margin modelling with leveraged / inverse ETFs (no futures, no direct shorts).

Two problems, one instrument set:

1. `levered_beta`: build a portfolio of S&P 500 stocks plus leveraged S&P ETFs (2x / 3x) and optional Reg-T margin that
   delivers a target beta (1.5 by default) while tracking `target_beta x benchmark` as closely as possible and paying as
   little as possible in ETF volatility drag, expense ratios and margin interest. Convex QP:

       minimise   (w - beta_T wb)' S (w - beta_T wb) + lambda_cost * (sum_k cost_k w_k + r_margin * m)
       s.t.       sum(w) = 1 + m,  beta' w = beta_T,  0 <= w_stock <= cap,  0 <= w_etf <= etf_cap,  0 <= m <= m_max,
                  Reg-T initial margin: m <= 0.5 * (1 + m)  (loan <= 50% of long market value),
                  maintenance with buffer: equity 1 >= (1 + buffer) * sum_i maint_i * w_i.

   The instrument's beta comes from the risk model's covariance (`_betas`), so a 3x fund is whatever the data says it is
   (about 3). Cost per $ of a k-times fund per year: expense ratio + volatility drag ~= (k^2 - k)/2 * sigma_index^2.

2. `tactical_overlay`: given the core (stocks) and a target beta from a tactical signal (0 .. 1.5), size a leveraged
   (target above core) or inverse (target below core) ETF position that moves the *total* beta to the target without
   selling the core, respecting the same margin policy, and report margin usage, carry and the tax cost avoided.

Margin policy (broker notes, Aug 2026): Reg-T 50% initial / 25% maintenance at Schwab and Fidelity; leveraged ETFs are
marginable but carry house maintenance of roughly 30% x leverage factor; futures are not available on the advisor
platforms; direct shorting is not permitted; inverse and leveraged ETFs are. This module is AI-editable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..lazy import lazy_module

cp = lazy_module("cvxpy")


@dataclass(frozen=True)
class Instrument:
    symbol: str
    leverage: float            # +2 / +3 long, -1 / -2 / -3 inverse
    index: str                 # SPX | NDX
    expense_ratio: float
    name: str = ""


INSTRUMENTS: dict[str, Instrument] = {
    "SSO": Instrument("SSO", 2.0, "SPX", 0.0089, "ProShares Ultra S&P500"),
    "SPUU": Instrument("SPUU", 2.0, "SPX", 0.0060, "Direxion Daily S&P 500 Bull 2X"),
    "UPRO": Instrument("UPRO", 3.0, "SPX", 0.0091, "ProShares UltraPro S&P500"),
    "SPXL": Instrument("SPXL", 3.0, "SPX", 0.0091, "Direxion Daily S&P 500 Bull 3X"),
    "SH": Instrument("SH", -1.0, "SPX", 0.0088, "ProShares Short S&P500"),
    "SDS": Instrument("SDS", -2.0, "SPX", 0.0090, "ProShares UltraShort S&P500"),
    "SPXU": Instrument("SPXU", -3.0, "SPX", 0.0090, "ProShares UltraPro Short S&P500"),
    "SPXS": Instrument("SPXS", -3.0, "SPX", 0.0097, "Direxion Daily S&P 500 Bear 3X"),
    "QLD": Instrument("QLD", 2.0, "NDX", 0.0095, "ProShares Ultra QQQ"),
    "TQQQ": Instrument("TQQQ", 3.0, "NDX", 0.0084, "ProShares UltraPro QQQ"),
    "PSQ": Instrument("PSQ", -1.0, "NDX", 0.0095, "ProShares Short QQQ"),
    "QID": Instrument("QID", -2.0, "NDX", 0.0095, "ProShares UltraShort QQQ"),
    "SQQQ": Instrument("SQQQ", -3.0, "NDX", 0.0095, "ProShares UltraPro Short QQQ"),
}
DEFAULT_LONG_LEVERED = ("SSO", "UPRO")
DEFAULT_INVERSE = ("SH", "SDS", "SPXU")
LEVERAGE_SYMBOLS = tuple(INSTRUMENTS)


def instrument_table() -> pd.DataFrame:
    return pd.DataFrame([{"symbol": i.symbol, "name": i.name, "leverage": i.leverage, "index": i.index, "expense_ratio": i.expense_ratio,
                          "kind": "leveraged long" if i.leverage > 1 else "inverse"} for i in INSTRUMENTS.values()])


@dataclass
class MarginPolicy:
    initial: float = 0.50              # Reg-T: loan <= 50% of purchase (buying power 2x equity)
    maintenance_stock: float = 0.30    # house maintenance on stocks (Reg-T minimum is 25%)
    maintenance_per_leverage: float = 0.30   # leveraged / inverse ETFs: ~30% x |leverage| (capped at 100%)
    buffer: float = 0.25               # keep equity >= (1 + buffer) x maintenance requirement
    margin_rate: float = 0.065         # annual interest on the loan
    max_loan: float = 0.50             # hard cap on borrowing as a fraction of equity (0 = cash only)
    allow_margin: bool = True

    def maintenance(self, symbol: str) -> float:
        inst = INSTRUMENTS.get(symbol)
        if inst is None:
            return self.maintenance_stock
        return min(1.0, self.maintenance_per_leverage * abs(inst.leverage))


DEFAULT_RF = 0.04                      # used when no money-market rate is supplied
SWAP_SPREAD = 0.004                    # over the reference rate on the swaps a leveraged fund uses for its extra exposure
INVERSE_BORROW = 0.02                  # per unit of short exposure, on top of fees and drag (interest earned is ignored: conservative)


def drag_cost(leverage: float, index_vol: float, expense_ratio: float, rf: float = DEFAULT_RF) -> float:
    """Expected annual cost per $ of a daily-rebalanced k-times fund versus k x the index:
    fees + embedded financing (k - 1) x (rf + swap spread) + volatility drag (k^2 - k)/2 x sigma^2.
    Comparable with a margin loan at `margin_rate` per $ borrowed, which buys one unit of beta."""
    k = abs(leverage)
    if leverage > 0:
        return expense_ratio + max(k - 1.0, 0.0) * (rf + SWAP_SPREAD) + 0.5 * (k * k - k) * index_vol ** 2
    return expense_ratio + k * INVERSE_BORROW + 0.5 * (k * k + k) * index_vol ** 2


@dataclass
class LeveredBetaSpec:
    """Defaults are tuned for the lowest tracking error: full replication of the index (no name cap), a stock cap that never
    binds on the largest index weight, and a small cost weight so drag/interest only break ties between equally tight books."""
    target_beta: float = 1.5
    instruments: tuple[str, ...] = DEFAULT_LONG_LEVERED
    n_max: int | None = None           # None = hold every index name (full replication -> stock sleeve tracks exactly)
    max_weight: float = 0.10           # per stock, fraction of equity; raised automatically to clear the largest index weight
    etf_max_weight: float = 0.35       # per leveraged ETF, fraction of equity
    min_weight: float = 0.0005
    cost_weight: float = 0.1           # lambda on annual cost (return units) vs tracking variance; 0 = pure tracking (TE first)
    index_vol: float = 0.16
    rf: float = DEFAULT_RF             # money-market rate embedded in leveraged-fund financing
    tracking_var: dict[str, float] = field(default_factory=dict)   # per fund: annual variance of (r_fund - k r_index), from history
    sector_band: float | None = None   # the stock sleeve is index-shaped already; a band only matters when n_max prunes
    margin: MarginPolicy = field(default_factory=MarginPolicy)
    solver: str = "CLARABEL"


@dataclass
class LeveredBetaResult:
    weights: pd.Series                 # stocks + ETFs, fraction of equity (sum = 1 + loan)
    loan: float                        # margin loan, fraction of equity
    beta: float
    tracking_error: float              # vs target_beta x benchmark
    cost: float                        # annual: ETF drag + fees + margin interest, fraction of equity
    margin: dict
    status: str
    diagnostics: dict = field(default_factory=dict)


def margin_report(weights: pd.Series, loan: float, policy: MarginPolicy) -> dict:
    """Equity = 1; long market value = 1 + loan. Maintenance requirement, excess, and the market drop that triggers a call."""
    mv = float(weights.clip(lower=0).sum())
    req = float(sum(policy.maintenance(s) * w for s, w in weights.items() if w > 0))
    equity = mv - loan
    excess = equity - req
    # a uniform drop d in all positions: equity -> (1-d) mv - loan, requirement -> (1-d) req; call when equal
    drop_to_call = (mv - loan - req) / (mv - req) if (mv - req) > 0 and loan > 0 else 1.0
    return {"long_market_value": mv, "loan": loan, "equity": equity, "maintenance_requirement": req, "maintenance_excess": excess,
            "equity_pct_of_mv": equity / mv if mv else 1.0, "market_drop_to_margin_call": float(max(min(drop_to_call, 1.0), 0.0)),
            "initial_margin_ok": loan <= policy.initial * mv + 1e-9, "buffer_ok": equity >= (1 + policy.buffer) * req - 1e-9,
            "annual_interest": loan * policy.margin_rate}


def leveraged_covariance(cov: pd.DataFrame, proxy: str | pd.Series, instruments: list[str], tracking_var: dict[str, float] | None = None) -> pd.DataFrame:
    """Replace the rows/columns of leveraged funds with k x the index. `proxy` is either the benchmark weight vector (the
    index basket itself: the fund then carries no idiosyncratic risk versus the stock sleeve, which is what a daily-reset
    fund on the same index delivers) or the symbol of an index ETF in `cov`. A daily-rebalanced k-times fund has beta k to
    its index by construction; a time-series fit on a factor model under-estimates it (1.7 for a 2x fund on live data),
    which would under-size the hedge and over-size the leverage. `tracking_var[s]` (annual variance of r_fund - k r_index,
    measured from history) goes on the diagonal so the optimizer sees the fund's real tracking noise."""
    C = cov.copy()
    tracking_var = tracking_var or {}
    if isinstance(proxy, pd.Series):
        wb = proxy.reindex(cov.index).fillna(0.0)
        if wb.sum() <= 0:
            return cov
        wb = wb / wb.sum()
        base_row = cov.values @ wb.values                       # cov(i, index)
        base_row = pd.Series(base_row, index=cov.index)
        var_b = float(wb.values @ base_row.values)
    else:
        if proxy not in cov.index:
            return cov
        base_row = cov.loc[proxy]
        var_b = float(cov.loc[proxy, proxy])
    for s in instruments:
        inst = INSTRUMENTS.get(s)
        if inst is None or s not in C.index:
            continue
        k = inst.leverage
        row = k * base_row
        C.loc[s, :] = row.reindex(C.columns).values
        C.loc[:, s] = row.reindex(C.index).values
    for a in instruments:
        if a not in C.index or a not in INSTRUMENTS:
            continue
        for b in instruments:
            if b in C.index and b in INSTRUMENTS:
                C.loc[a, b] = INSTRUMENTS[a].leverage * INSTRUMENTS[b].leverage * var_b
        C.loc[a, a] += float(tracking_var.get(a, (0.005) ** 2))
    return C


def nominal_betas(betas: pd.Series, proxy: str | None, instruments: list[str]) -> pd.Series:
    """Beta of each leveraged fund = k x beta(proxy) (1.0 if the proxy is unknown); other names unchanged."""
    b = betas.copy()
    base = float(betas[proxy]) if proxy is not None and proxy in betas.index else 1.0
    for s in instruments:
        if s in INSTRUMENTS and s in b.index:
            b[s] = INSTRUMENTS[s].leverage * base
    return b


def build_levered_beta(cov: pd.DataFrame, benchmark: pd.Series, stocks: list[str], betas: pd.Series, spec: LeveredBetaSpec,
                       sectors: pd.Series | None = None, proxy: str | None = None) -> LeveredBetaResult:
    """`cov` must cover stocks and the chosen instruments (annualised); `betas` are cov-implied betas to the benchmark.
    If `proxy` (e.g. SPY) is given, leveraged funds are modelled as k x proxy (covariance and beta), not as fitted."""
    etfs = [s for s in spec.instruments if s in cov.index and s in betas.index]
    implied = {s: float(betas[s]) for s in etfs}
    bench_w = benchmark[[s for s in benchmark.index if s in cov.index and s not in INSTRUMENTS]]
    if etfs and bench_w.sum() > 0:
        # leveraged funds = k x the index basket (beta exactly k to the benchmark, no idiosyncratic term versus the stocks)
        cov = leveraged_covariance(cov, bench_w, etfs, spec.tracking_var)
        betas = nominal_betas(betas, None, etfs)
    elif proxy is not None and proxy in cov.index:
        cov = leveraged_covariance(cov, proxy, etfs, spec.tracking_var)
        betas = nominal_betas(betas, proxy, etfs)
    if not etfs and not spec.margin.allow_margin:
        raise ValueError("no leveraged ETFs in the risk model and margin disabled: cannot reach the target beta")
    syms = [s for s in stocks if s in cov.index and s in betas.index] + etfs
    n, n_s = len(syms), len(syms) - len(etfs)
    S = cov.reindex(index=syms, columns=syms).fillna(0.0).values
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 1e-10, None)
    S = vecs @ np.diag(vals) @ vecs.T
    F = vecs * np.sqrt(vals)                               # S = F F'; tracking variance = ||F' a||^2 (SOC form: fast, stable)
    b = betas.reindex(syms).values
    wb = benchmark.reindex(syms).fillna(0.0).values
    wb = wb / wb.sum() if wb.sum() > 0 else wb
    pol = spec.margin
    # the stock cap must clear the largest index weight (scaled by the stock sleeve size) or tracking error is forced in
    stock_cap = float(spec.max_weight)
    wb_max = float(wb[:n_s].max()) if n_s else 0.0
    cap_needed = wb_max * (1.0 + (pol.max_loan if pol.allow_margin else 0.0)) * 1.05
    cap_raised = False
    if cap_needed > stock_cap:
        stock_cap, cap_raised = cap_needed, True
    cost = np.zeros(n)
    for j, s in enumerate(etfs):
        inst = INSTRUMENTS[s]
        cost[n_s + j] = drag_cost(inst.leverage, spec.index_vol, inst.expense_ratio, spec.rf)
    maint = np.array([pol.maintenance(s) for s in syms])
    is_etf = np.array([s in INSTRUMENTS for s in syms])
    secm = None
    if sectors is not None and spec.sector_band is not None:
        sec = sectors.reindex(syms)
        names = sorted(sec.dropna().unique())
        if names:
            secm = np.array([[1.0 if sec[s] == g else 0.0 for g in names] for s in syms])

    def solve(fixed_zero: np.ndarray) -> tuple[np.ndarray, float, str]:
        w = cp.Variable(n)
        m = cp.Variable()
        active = w - spec.target_beta * wb
        cons = [w >= 0, m >= 0, cp.sum(w) == 1 + m, b @ w == spec.target_beta]
        cons += [w[~is_etf] <= stock_cap] if (~is_etf).any() else []
        cons += [w[is_etf] <= spec.etf_max_weight] if is_etf.any() else []
        max_loan = pol.max_loan if pol.allow_margin else 0.0
        cons += [m <= max_loan, m <= pol.initial * (1 + m), 1.0 >= (1 + pol.buffer) * (maint @ w)]
        z = np.where(fixed_zero)[0]
        if len(z):
            cons.append(w[z] == 0)
        if secm is not None:
            # sector neutrality of the stock sleeve relative to the (scaled) benchmark, ETFs excluded
            stock_active = w - (cp.sum(w[~is_etf]) if (~is_etf).any() else 1.0) * wb
            cons.append(cp.abs(secm.T @ cp.multiply((~is_etf).astype(float), stock_active)) <= spec.sector_band)
        obj = cp.sum_squares(F.T @ active) + spec.cost_weight * (cost @ w + pol.margin_rate * m) * 1e-2
        prob = cp.Problem(cp.Minimize(obj), cons)
        for sv in (spec.solver, "OSQP", "SCS"):
            try:
                t0 = time.perf_counter()
                prob.solve(solver=sv, verbose=False)
                solve_log.append({"solver": sv, "status": prob.status, "seconds": round(time.perf_counter() - t0, 3)})
                if prob.status in ("optimal", "optimal_inaccurate"):
                    return np.clip(np.asarray(w.value).ravel(), 0, None), float(m.value), prob.status
            except Exception as e:  # solver availability
                solve_log.append({"solver": sv, "status": f"error: {type(e).__name__}", "seconds": 0.0})
                continue
        raise RuntimeError(f"levered-beta construction infeasible ({prob.status}); loosen margin, ETF cap or beta target")

    solve_log: list[dict] = []
    fixed = np.zeros(n, dtype=bool)
    wv, loan, status = solve(fixed)
    if spec.n_max and spec.n_max < n_s:
        for _ in range(5):
            stock_idx = np.where(~is_etf & (wv > 1e-6))[0]
            if len(stock_idx) <= spec.n_max:
                break
            keep = stock_idx[np.argsort(wv[stock_idx])[::-1][: spec.n_max]]
            fixed = ~is_etf
            fixed[keep] = False
            wv, loan, status = solve(fixed)
    if spec.min_weight > 0:
        # names below the minimum are removed and the book is re-solved with them fixed at zero, so beta and tracking
        # error stay optimal for the names actually held (pro-rata re-scaling would break both)
        for _ in range(3):
            small = (wv > 1e-9) & (wv < spec.min_weight) & ~is_etf
            if not small.any():
                break
            fixed = fixed | small
            wv, loan, status = solve(fixed)
    wv = np.where(wv < 1e-7, 0.0, wv)
    target_sum = 1.0 + max(loan, 0.0)
    deficit = target_sum - float(np.sum(wv))                 # numerical crumbs only
    stock_sum = float(np.sum(wv[~is_etf]))
    if abs(deficit) > 1e-12 and stock_sum > 0:
        wv[~is_etf] *= (stock_sum + deficit) / stock_sum
    loan = max(float(np.sum(wv) - 1.0), 0.0)
    w = pd.Series(wv, index=syms)
    w = w[w > 0]
    active = w.reindex(syms).fillna(0.0).values - spec.target_beta * wb
    te = float(np.sqrt(max(active @ S @ active, 0.0)))
    beta = float(b @ w.reindex(syms).fillna(0.0).values)
    total_cost = float(cost @ w.reindex(syms).fillna(0.0).values + pol.margin_rate * loan)
    rep = margin_report(w, loan, pol)
    etf_w = {s: round(float(w[s]), 4) for s in etfs if s in w.index}
    diag = {"n_stocks": int(sum(1 for s in w.index if s not in INSTRUMENTS)), "etf_weights": etf_w, "stock_weight": float(sum(v for s, v in w.items() if s not in INSTRUMENTS)),
            "beta_from_stocks": float(sum(b[syms.index(s)] * v for s, v in w.items() if s not in INSTRUMENTS)),
            "beta_from_etfs": float(sum(b[syms.index(s)] * v for s, v in w.items() if s in INSTRUMENTS)),
            "etf_betas": {s: round(float(betas[s]), 3) for s in etfs}, "etf_betas_model_implied": {s: round(v, 3) for s, v in implied.items()},
            "beta_source": "nominal leverage x index basket" if (etfs and bench_w.sum() > 0) else ("nominal leverage x proxy beta" if (proxy is not None and proxy in cov.index) else "risk-model covariance"),
            "etf_cost_per_year": {s: round(float(cost[n_s + j]), 4) for j, s in enumerate(etfs)}, "margin_rate": pol.margin_rate,
            "target_beta": spec.target_beta, "stock_cap_used": stock_cap, "stock_cap_raised": cap_raised, "n_index_names": int((wb[:n_s] > 0).sum()),
            "replication": "full" if not spec.n_max or spec.n_max >= int((wb[:n_s] > 0).sum()) else f"sampled ({spec.n_max})",
            "solves": solve_log, "cost_breakdown": {
                "etf_drag_and_fees": float(cost @ w.reindex(syms).fillna(0.0).values), "margin_interest": float(pol.margin_rate * loan)}}
    return LeveredBetaResult(weights=w.sort_values(ascending=False), loan=loan, beta=beta, tracking_error=te, cost=total_cost, margin=rep, status=status, diagnostics=diag)


def realised_tracking(returns: pd.DataFrame, weights: pd.Series, benchmark: pd.Series, target_beta: float, loan: float = 0.0,
                      margin_rate: float = 0.065, rebalance: str = "M", proxy: str | None = None) -> dict:
    """Historical check with the *actual* ETF price histories (daily-reset compounding included): hold `weights` (fraction of
    equity, sum = 1 + loan) rebalanced at `rebalance` frequency, pay interest on the loan, and compare with
    target_beta x the index's daily return. The index leg is the real index ETF (`proxy`, e.g. SPY) when its history is
    available, because that is what the leveraged funds track; otherwise today's cap weights applied historically (which
    carries look-ahead in the stock sleeve). Returns annualised tracking error, realised beta, correlation and the
    daily-rebalanced TE as a lower bound. Names without history are dropped from both legs (reported)."""
    R = returns.copy()
    held = [s for s in weights.index if s in R.columns]
    missing = [s for s in weights.index if s not in R.columns]
    bcols = [s for s in benchmark.index if s in R.columns and benchmark[s] > 0]
    use_proxy = proxy is not None and proxy in R.columns and R[proxy].notna().sum() >= 60
    if not held or (not bcols and not use_proxy):
        return {"available": False, "missing": missing}
    R = R[sorted(set(held) | set(bcols) | ({proxy} if use_proxy else set()))].dropna(how="all")
    R = R.loc[R[held].notna().all(axis=1)]
    if len(R) < 60:
        return {"available": False, "missing": missing, "n_days": int(len(R))}
    R = R.fillna(0.0)
    if use_proxy:
        r_b = R[proxy].astype(float)
    else:
        wb = benchmark.reindex(bcols).fillna(0.0)
        wb = wb / wb.sum()
        r_b = (R[bcols] @ wb.values).astype(float)
    w0 = weights.reindex(held).fillna(0.0)
    daily_rate = margin_rate / 252.0
    periods = R.index.to_period(rebalance).astype(str).values
    tgt = target_beta * r_b
    var_b = float(np.var(r_b.values, ddof=1))
    out = {"available": True, "missing": missing, "n_days": int(len(R)), "start": str(R.index.min().date()), "end": str(R.index.max().date()),
           "rebalance": rebalance, "target_beta": target_beta, "index_leg": proxy if use_proxy else "today's cap weights (look-ahead)"}

    def stats(prefix: str, Rmat: np.ndarray, wvec: np.ndarray) -> None:
        r_d = pd.Series(Rmat @ wvec - loan * daily_rate, index=R.index)                      # rebalanced to target every day
        r_p = pd.Series(_periodic_returns(Rmat, wvec, loan, daily_rate, periods), index=R.index)   # drift within each period
        for label, r in ((f"{prefix}daily", r_d), (f"{prefix}periodic", r_p)):
            act = r - tgt
            out[f"te_{label}"] = float(act.std() * np.sqrt(252))
            out[f"beta_{label}"] = float(np.cov(r.values, r_b.values)[0, 1] / var_b) if var_b > 0 else np.nan
            out[f"corr_{label}"] = float(np.corrcoef(r.values, tgt.values)[0, 1]) if tgt.std() > 0 else np.nan
            out[f"excess_{label}"] = float(act.mean() * 252)

    # full book with today's weights (the stock sleeve carries look-ahead: today's index weights favour the past winners)
    stats("", R[held].values, w0.values)
    if use_proxy:
        # structure only: the stock sleeve is assumed to track the index (full replication does), so this isolates the
        # leveraged funds' daily-reset compounding, fees and the loan's interest: the tracking error the *leverage* adds
        etf_cols = [s for s in held if s in INSTRUMENTS]
        stock_w = float(sum(w0[s] for s in held if s not in INSTRUMENTS))
        cols = [proxy] + etf_cols
        wvec = np.array([stock_w] + [float(w0[s]) for s in etf_cols])
        stats("structure_", R[cols].values, wvec)
        out["structure_note"] = "stock sleeve replaced by the index ETF; measures what leveraged funds, fees and margin interest add"
    return out


def _periodic_returns(Rmat: np.ndarray, w0: np.ndarray, loan: float, daily_rate: float, periods: np.ndarray) -> np.ndarray:
    """Equity returns of a book reset to weights `w0` (sum = 1 + loan) at the start of each period and left to drift inside it."""
    out = np.zeros(len(Rmat))
    vals = w0.copy()
    ln = loan
    cur = None
    for i in range(len(Rmat)):
        if periods[i] != cur:
            vals, ln, cur = w0.copy(), loan, periods[i]
        rr = Rmat[i]
        eq0 = float(vals.sum() - ln)
        pnl = float(vals @ rr) - ln * daily_rate
        vals = vals * (1 + rr)
        ln = ln * (1 + daily_rate)
        out[i] = pnl / max(eq0, 1e-9)
    return out


# ====================================================================================== tactical overlay sizing
@dataclass
class OverlayTicket:
    symbol: str
    side: str                          # BUY | SELL
    weight: float                      # fraction of equity (positive)
    beta_contribution: float
    annual_cost: float
    note: str = ""


def tactical_overlay(core_value: float, core_beta: float, target_beta: float, policy: MarginPolicy | None = None,
                     long_instruments: tuple[str, ...] = DEFAULT_LONG_LEVERED, inverse_instruments: tuple[str, ...] = DEFAULT_INVERSE,
                     instrument_betas: dict[str, float] | None = None, cash: float = 0.0, index_vol: float = 0.16,
                     existing_overlay: dict[str, float] | None = None, core_gain_frac: float = 0.3, lt_rate: float = 0.238,
                     prices: pd.Series | None = None, rf: float = DEFAULT_RF) -> dict:
    """Move total beta from `core_beta` (core held at `core_value`) to `target_beta` with one leveraged or inverse ETF,
    never selling the core. Chooses the instrument with the lowest annual cost per unit of beta that fits the margin
    policy; reports margin usage and the tax that selling core instead would have cost."""
    policy = policy or MarginPolicy()
    equity = core_value + cash
    existing_overlay = existing_overlay or {}
    total_beta_now = core_beta * core_value / equity + sum((instrument_betas or {}).get(s, INSTRUMENTS[s].leverage) * v / equity for s, v in existing_overlay.items())
    gap = target_beta - total_beta_now                          # beta units of equity to add (negative = remove)
    candidates = list(long_instruments if gap > 0 else inverse_instruments)
    rows = []
    for s in candidates:
        inst = INSTRUMENTS[s]
        beta_s = (instrument_betas or {}).get(s, inst.leverage)
        if gap > 0 and beta_s <= 0 or gap < 0 and beta_s >= 0:
            continue
        w_needed = gap / beta_s                                   # fraction of equity to buy
        cost = drag_cost(inst.leverage, index_vol, inst.expense_ratio, rf) * w_needed
        # margin: overlay funded from cash first, remainder borrowed
        loan = max(w_needed * equity - cash, 0.0) / equity
        w_all = pd.Series({**{"CORE": core_value / equity}, **{k: v / equity for k, v in existing_overlay.items()}, s: w_needed})
        # maintenance: treat CORE as stocks
        req = policy.maintenance_stock * w_all["CORE"] + sum(policy.maintenance(k) * v for k, v in w_all.items() if k != "CORE")
        eq = float(w_all.sum() - loan)
        ok_initial = loan <= policy.initial * float(w_all.sum()) + 1e-9
        ok_maint = eq >= (1 + policy.buffer) * req - 1e-9
        ok_cap = loan <= (policy.max_loan if policy.allow_margin else 0.0) + 1e-9
        rows.append({"symbol": s, "leverage": inst.leverage, "beta_used": beta_s, "weight": w_needed, "notional": w_needed * equity,
                     "annual_cost": cost + loan * policy.margin_rate, "loan": loan, "feasible": ok_initial and ok_maint and ok_cap,
                     "maintenance_excess": eq - req, "market_drop_to_call": float(max(min((eq - req) / max(float(w_all.sum()) - req, 1e-9), 1.0), 0.0)) if loan > 0 else 1.0})
    table = pd.DataFrame(rows)
    if table.empty or abs(gap) < 1e-4:
        return {"target_beta": target_beta, "beta_now": total_beta_now, "gap": gap, "tickets": [], "table": table, "note": "no change needed" if abs(gap) < 1e-4 else "no instrument can move beta in that direction"}
    feas = table[table["feasible"]]
    if feas.empty:
        # best effort: the instrument with the largest feasible partial move under the margin cap
        best = table.sort_values("annual_cost").iloc[0]
        max_loan = policy.max_loan if policy.allow_margin else 0.0
        w_max = (cash / equity) + max_loan
        scale = min(1.0, w_max / max(best["weight"], 1e-9))
        ticket = OverlayTicket(best["symbol"], "BUY", best["weight"] * scale, gap * scale, best["annual_cost"] * scale,
                               f"margin cap reached: only {scale:.0%} of the target move is feasible; achieved beta {total_beta_now + gap * scale:.2f}")
        chosen = best
    else:
        chosen = feas.sort_values("annual_cost").iloc[0]
        ticket = OverlayTicket(chosen["symbol"], "BUY", chosen["weight"], gap, chosen["annual_cost"], "cheapest feasible instrument")
    # tax cost avoided: reducing beta by selling core instead would realise gains on the sold slice
    tax_if_sold_core = 0.0
    if gap < 0 and core_beta > 0:
        sell_frac = min(1.0, -gap * equity / (core_beta * core_value))
        tax_if_sold_core = sell_frac * core_value * core_gain_frac * lt_rate
    tickets = [ticket]
    # unwind of existing overlay legs that point the wrong way
    for s, v in existing_overlay.items():
        lev = INSTRUMENTS.get(s, Instrument(s, 1.0, "SPX", 0.0)).leverage
        if (gap > 0 and lev < 0) or (gap < 0 and lev > 0):
            tickets.insert(0, OverlayTicket(s, "SELL", v / equity, -lev * v / equity, 0.0, "existing overlay leg points the wrong way"))
    shares = None
    if prices is not None and ticket.symbol in prices.index and prices[ticket.symbol] > 0:
        shares = int(round(ticket.weight * equity / float(prices[ticket.symbol])))
    return {"target_beta": target_beta, "beta_now": total_beta_now, "gap": gap, "equity": equity,
            "tickets": [t.__dict__ | ({"shares": shares} if t is ticket else {}) for t in tickets], "chosen": chosen.to_dict(), "table": table,
            "beta_after": total_beta_now + ticket.beta_contribution, "tax_avoided_vs_selling_core": tax_if_sold_core,
            "annual_cost": ticket.annual_cost * equity, "note": ticket.note}


def simulate_tactical(index_returns: pd.Series, target_beta: pd.Series, core_returns: pd.Series | None = None, policy: MarginPolicy | None = None,
                      long_instrument: str = "SSO", inverse_instrument: str = "SDS", index_vol: float | None = None,
                      rebalance_threshold: float = 0.1, cost_bps: float = 5.0) -> dict:
    """Daily simulation: core (stock sleeve, beta 1 unless `core_returns` given) held throughout; each day the overlay ETF
    weight is reset when the target beta changes by more than `rebalance_threshold`. Leveraged ETF daily return =
    k x index return - expense/252 (daily rebalancing captured exactly by compounding). Margin interest accrues on the loan.
    Returns equity curve, realised beta, turnover, cost and overlay P&L (a proxy for the short-term gains/losses it books)."""
    policy = policy or MarginPolicy()
    r_idx = index_returns.dropna()
    tb = target_beta.reindex(r_idx.index).ffill().fillna(1.0)
    core = (core_returns.reindex(r_idx.index).fillna(0.0) if core_returns is not None else r_idx)
    iv = index_vol or float(r_idx.std() * np.sqrt(252))
    lev_l, lev_i = INSTRUMENTS[long_instrument], INSTRUMENTS[inverse_instrument]
    equity = 1.0
    core_v, ov_v, loan = 1.0, 0.0, 0.0
    ov_basis = 0.0
    ov_sym = None
    cur_target = None
    eq_curve, betas, turn, costs, ov_pnl, realised = [], [], [], [], [], []
    for d in r_idx.index:
        t = float(tb[d])
        r_i, r_c = float(r_idx[d]), float(core[d])
        # rebalance overlay if target moved
        if cur_target is None or abs(t - cur_target) > rebalance_threshold:
            eq = core_v + ov_v - loan
            gap = t - core_v / eq                                 # core beta assumed 1
            inst = lev_l if gap > 0 else lev_i
            w_ov = gap / inst.leverage if abs(gap) > 1e-6 else 0.0
            new_ov = w_ov * eq
            trade = abs(new_ov - (ov_v if ov_sym == inst.symbol else 0.0)) + (ov_v if ov_sym not in (None, inst.symbol) else 0.0)
            # realised P&L: switching instrument closes the leg; reducing realises the sold fraction pro rata
            if ov_sym is not None and ov_sym != inst.symbol:
                realised.append(ov_v - ov_basis)
                ov_basis = 0.0
            elif ov_sym == inst.symbol and new_ov < ov_v and ov_v > 0:
                frac = (ov_v - new_ov) / ov_v
                realised.append((ov_v - ov_basis) * frac)
                ov_basis *= 1 - frac
            else:
                realised.append(0.0)
            # margin cap
            max_loan = (policy.max_loan if policy.allow_margin else 0.0) * eq
            loan = max(core_v + new_ov - eq, 0.0)
            if loan > max_loan:
                new_ov -= loan - max_loan
                loan = max_loan
            if ov_sym != inst.symbol or new_ov > ov_v:
                ov_basis += max(new_ov - (ov_v if ov_sym == inst.symbol else 0.0), 0.0)
            ov_v, ov_sym = new_ov, inst.symbol
            costs.append(trade * cost_bps / 1e4)
            turn.append(trade / eq)
            cur_target = t
        else:
            costs.append(0.0)
            turn.append(0.0)
            realised.append(0.0)
        inst = INSTRUMENTS.get(ov_sym) if ov_sym else None
        r_ov = (inst.leverage * r_i - inst.expense_ratio / 252.0) if inst else 0.0
        ov_start = ov_v
        core_v *= 1 + r_c
        ov_v *= 1 + r_ov
        interest = loan * policy.margin_rate / 252.0
        loan += interest
        equity = core_v + ov_v - loan - costs[-1]
        core_v -= costs[-1]
        eq_curve.append(equity)
        betas.append((core_v + (inst.leverage if inst else 0.0) * ov_v) / equity if equity > 0 else np.nan)
        ov_pnl.append(ov_v - ov_start)
    eq = pd.Series(eq_curve, index=r_idx.index)
    bench = (1 + r_idx).cumprod()
    core_only = (1 + core).cumprod()
    rets = eq.pct_change().dropna()
    yrs = len(rets) / 252.0
    dd = float((eq / eq.cummax() - 1).min())
    beta_realised = float(np.cov(rets.values, r_idx.reindex(rets.index).values)[0, 1] / np.var(r_idx.reindex(rets.index).values, ddof=1))
    out = {"equity": eq, "benchmark": bench, "core_only": core_only, "target_beta": tb, "realised_beta_series": pd.Series(betas, index=r_idx.index),
           "overlay_pnl": pd.Series(ov_pnl, index=r_idx.index),
           "metrics": {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan, "vol": float(rets.std() * np.sqrt(252)), "max_drawdown": dd,
                       "bench_cagr": float(bench.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan, "bench_max_drawdown": float((bench / bench.cummax() - 1).min()),
                       "core_cagr": float(core_only.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan,
                       "realised_beta": beta_realised, "avg_target_beta": float(tb.mean()), "annual_turnover": float(np.sum(turn) / max(yrs, 1e-9)),
                       "total_costs": float(np.sum(costs)), "n_rebalances": int(sum(1 for x in turn if x > 0)),
                       "overlay_unrealised_pnl_total": float(np.sum(ov_pnl)), "overlay_realised_pnl": float(np.sum(realised)),
                       "overlay_losses_booked": float(-np.sum([p for p in realised if p < 0])), "overlay_gains_booked": float(np.sum([p for p in realised if p > 0])),
                       "index_vol_used": iv}}
    return out
