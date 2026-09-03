"""Multi-period tax-aware diversification of a concentrated position (glide path) and Monte Carlo evaluation.

Glide path (convex program, cvxpy)
----------------------------------
Decide dollars d_t to sell in each of T periods to minimise
    sum_t disc_t * [ tax_t(d_t) + cost * d_t ] + terminal_tax(remaining)
    + risk_aversion/2 * sum_t disc_t * sigma_eps^2 * R_t^2 / W
    - sum_t disc_t * alpha * R_t
where R_t is the concentrated dollars still held after period t (affine in d), tax_t is the bracket tax on the taxable
gain stacked on other income (convex piecewise-linear via `convex_pieces`), losses available for offset (harvested
losses, carryforwards) are used through variables u_t with a cumulative availability constraint, short-term lots are
taxed at ordinary rates until they turn long-term, and the terminal position is taxed at the LT rate unless a basis
step-up occurs (probability p_stepup). Optional constraints: annual gain budget, minimum diversification by a date.

Monte Carlo compares policies (hold / sell now / equal instalments / optimised schedule) on after-tax terminal wealth.
This module is AI-editable (ai/registry.py).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..lazy import lazy_module
from ..tax.concentration import BracketSchedule, convex_pieces, ltcg_tax, ordinary_tax, tax_from_pieces

cp = lazy_module("cvxpy")          # imported on first use (saves ~1.7 s at launch)
log = logging.getLogger(__name__)


@dataclass
class GlidePathSpec:
    horizon_years: int = 5
    periods_per_year: int = 1
    other_taxable_income: float = 300_000.0          # stacked below the gains each year
    discount_rate: float = 0.04
    risk_aversion: float = 4.0                        # lambda on idiosyncratic variance (in wealth units)
    alpha_view: float = 0.0                           # expected excess return of the stock vs the diversified alternative (annual)
    expected_return: float = 0.07                     # stock drift used to grow the position (annual)
    cost_bps: float = 10.0
    p_stepup: float = 0.0                             # probability of a basis step-up (death) within the horizon
    annual_gain_budget: float | None = None           # cap on realised gain per year
    min_sold_by: dict[int, float] = field(default_factory=dict)   # {year: cumulative fraction sold at least}
    losses_by_year: dict[int, float] = field(default_factory=dict)  # expected harvestable losses available per year (>=0)
    carryforward: float = 0.0                         # loss carryforward available at t=0
    solver: str = "CLARABEL"


@dataclass
class PositionFacts:
    symbol: str
    value: float
    basis: float
    st_value: float = 0.0                             # portion of value in short-term lots
    st_basis: float = 0.0
    years_to_lt: float = 1.0                          # when the ST portion turns LT (years from now)
    specific_vol: float = 0.30                        # annualised idiosyncratic vol from the risk model
    beta: float = 1.0
    total_vol: float = 0.35
    total_wealth: float | None = None                 # portfolio value for risk scaling (default = value * 2)
    weight: float | None = None


@dataclass
class GlidePathResult:
    schedule: pd.DataFrame
    objective: float
    status: str
    summary: dict
    comparison: pd.DataFrame


def _periods(spec: GlidePathSpec) -> int:
    return int(spec.horizon_years * spec.periods_per_year)


def solve_glidepath(pos: PositionFacts, spec: GlidePathSpec, sched: BracketSchedule) -> GlidePathResult:
    T = _periods(spec)
    dt = 1.0 / spec.periods_per_year
    W = pos.total_wealth or pos.value * 2.0
    growth = np.array([(1 + spec.expected_return) ** (dt * (t + 1)) for t in range(T)])
    disc = np.array([(1 + spec.discount_rate) ** (-dt * (t + 1)) for t in range(T)])
    V0 = pos.value
    # gain fraction per dollar sold in period t (basis fixed, value grows): LT and ST portions
    lt_value0, lt_basis = V0 - pos.st_value, pos.basis - pos.st_basis
    gamma_lt = np.clip(1 - lt_basis / np.maximum(lt_value0 * growth, 1e-9), -1, 1) if lt_value0 > 0 else np.zeros(T)
    gamma_st = np.clip(1 - pos.st_basis / np.maximum(pos.st_value * growth, 1e-9), -1, 1) if pos.st_value > 0 else np.zeros(T)
    st_frac0 = pos.st_value / V0 if V0 > 0 else 0.0
    st_still_st = np.array([(t + 1) * dt < pos.years_to_lt for t in range(T)])
    d = cp.Variable(T, nonneg=True)                   # dollars sold (today's dollars scaled by growth handled below)
    # sold fraction f_t = d_t / (V0*growth_t); remaining fraction after t: 1 - sum f
    f = cp.multiply(d, 1.0 / (V0 * growth))
    cum_f = cp.cumsum(f)
    remaining_val = cp.multiply(1 - cum_f, V0 * growth)   # R_t
    cons = [cum_f <= 1.0]
    # realised gain: ST portion (while ST) taxed ordinary, otherwise LT
    gain_lt = cp.multiply(d, np.where(st_still_st, (1 - st_frac0) * gamma_lt, (1 - st_frac0) * gamma_lt + st_frac0 * gamma_st))
    gain_st = cp.multiply(d, np.where(st_still_st, st_frac0 * gamma_st, 0.0))
    # loss offsets: u_t used against gains, cumulative availability
    losses = np.array([spec.losses_by_year.get(int(t * dt) + 1, 0.0) / spec.periods_per_year for t in range(T)])
    u = cp.Variable(T, nonneg=True)
    cons += [u <= gain_lt + gain_st, cp.cumsum(u) <= np.cumsum(losses) + spec.carryforward]
    taxable_lt = cp.pos(gain_lt - u)
    taxable_st = gain_st                               # apply offsets to LT first (conservative; ST offset handled by netting order in ledger)
    inc = spec.other_taxable_income / spec.periods_per_year
    pieces_lt = convex_pieces(sched, inc, "ltcg")
    pieces_st = convex_pieces(sched, inc, "ordinary")
    tax_t = sum(sl * cp.pos(taxable_lt - k) for k, sl in pieces_lt) + sum(sl * cp.pos(taxable_st - k) for k, sl in pieces_st)
    # terminal: remaining position taxed at LT (unless step-up)
    term_gain = cp.multiply(remaining_val[-1], gamma_lt[-1] if lt_value0 > 0 else gamma_st[-1])
    term_rate = ltcg_tax(max(V0 - pos.basis, 1.0), inc, sched)["marginal_rate"]
    terminal_tax = (1 - spec.p_stepup) * term_rate * term_gain * disc[-1]
    risk = spec.risk_aversion / 2 * pos.specific_vol ** 2 * dt * cp.sum(cp.multiply(disc, cp.square(remaining_val))) / W
    alpha_term = spec.alpha_view * dt * cp.sum(cp.multiply(disc, remaining_val))
    cost = (spec.cost_bps / 1e4) * cp.sum(cp.multiply(disc, d))
    obj = cp.sum(cp.multiply(disc, tax_t)) + terminal_tax + risk - alpha_term + cost
    if spec.annual_gain_budget is not None:
        for y in range(spec.horizon_years):
            idx = list(range(y * spec.periods_per_year, (y + 1) * spec.periods_per_year))
            cons.append(cp.sum(gain_lt[idx] + gain_st[idx]) <= spec.annual_gain_budget)
    for year, frac in spec.min_sold_by.items():
        t_idx = min(int(year * spec.periods_per_year) - 1, T - 1)
        if t_idx >= 0:
            cons.append(cum_f[t_idx] >= float(frac))
    prob = cp.Problem(cp.Minimize(obj), cons)
    status = "failed"
    for sv in (spec.solver, "SCS", "OSQP"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=sv, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                status = prob.status
                break
        except Exception as e:  # noqa: BLE001
            log.debug("glidepath solver %s: %s", sv, e)
    if d.value is None:
        raise RuntimeError(f"glide path optimisation failed: {prob.status}")
    dv = np.clip(np.asarray(d.value).ravel(), 0, None)
    dv[dv < 1.0] = 0.0
    sched_df = evaluate_schedule(dv, pos, spec, sched)
    comp = compare_policies(pos, spec, sched, dv)
    s = sched_df
    summary = {
        "objective": float(prob.value), "status": status, "total_sold": float(s["sold"].sum()), "total_tax": float(s["tax"].sum()),
        "pv_tax": float((s["tax"] * s["discount"]).sum()), "total_gain_realised": float(s["realised_gain"].sum()),
        "losses_used": float(s["loss_offset_used"].sum()), "final_weight": float(s["weight_after"].iloc[-1]),
        "final_remaining_value": float(s["remaining_value"].iloc[-1]), "years": spec.horizon_years,
        "effective_tax_rate": float(s["tax"].sum() / s["realised_gain"].sum()) if s["realised_gain"].sum() > 0 else 0.0,
    }
    return GlidePathResult(sched_df, float(prob.value), status, summary, comp)


def evaluate_schedule(d: np.ndarray, pos: PositionFacts, spec: GlidePathSpec, sched: BracketSchedule) -> pd.DataFrame:
    """Deterministic year-by-year accounting of a sell schedule (dollars per period)."""
    T = _periods(spec)
    dt = 1.0 / spec.periods_per_year
    W = pos.total_wealth or pos.value * 2.0
    V0 = pos.value
    lt_value0, lt_basis = V0 - pos.st_value, pos.basis - pos.st_basis
    st_frac0 = pos.st_value / V0 if V0 > 0 else 0.0
    rows = []
    cum = 0.0
    carry = spec.carryforward
    inc = spec.other_taxable_income / spec.periods_per_year
    for t in range(T):
        g = (1 + spec.expected_return) ** (dt * (t + 1))
        disc = (1 + spec.discount_rate) ** (-dt * (t + 1))
        value_t = V0 * g
        sold = float(min(d[t], max(value_t * (1 - cum), 0.0)))
        frac = sold / value_t if value_t > 0 else 0.0
        gamma_lt = 1 - lt_basis / max(lt_value0 * g, 1e-9) if lt_value0 > 0 else 0.0
        gamma_st = 1 - pos.st_basis / max(pos.st_value * g, 1e-9) if pos.st_value > 0 else 0.0
        still_st = (t + 1) * dt < pos.years_to_lt
        gain_st = sold * st_frac0 * gamma_st if still_st else 0.0
        gain_lt = sold * ((1 - st_frac0) * gamma_lt + (0.0 if still_st else st_frac0 * gamma_st))
        carry += spec.losses_by_year.get(int(t * dt) + 1, 0.0) / spec.periods_per_year
        used = min(carry, max(gain_lt, 0.0) + max(gain_st, 0.0))
        carry -= used
        taxable_lt = max(gain_lt - used, 0.0)
        taxable_st = max(gain_st - max(used - max(gain_lt, 0.0), 0.0), 0.0)
        tax_lt = ltcg_tax(taxable_lt, inc, sched)["total"] if taxable_lt > 0 else 0.0
        tax_st = ordinary_tax(taxable_st, inc + taxable_lt, sched)["total"] if taxable_st > 0 else 0.0
        cum += frac
        remaining = value_t * (1 - cum)
        rows.append({"period": t + 1, "year": int(t * dt) + 1, "position_value": value_t, "sold": sold, "sold_fraction": frac,
                     "cumulative_fraction": cum, "realised_gain": gain_lt + gain_st, "loss_offset_used": used, "taxable_gain": taxable_lt + taxable_st,
                     "tax": tax_lt + tax_st, "marginal_rate": marginal_rate_for(taxable_lt, inc, sched), "remaining_value": remaining,
                     "weight_after": remaining / (W - V0 + value_t) if W > 0 else 0.0, "discount": disc,
                     "risk_cost": spec.risk_aversion / 2 * pos.specific_vol ** 2 * dt * remaining ** 2 / W,
                     "cost": sold * spec.cost_bps / 1e4})
    return pd.DataFrame(rows)


def marginal_rate_for(taxable: float, inc: float, sched: BracketSchedule) -> float:
    from ..tax.concentration import marginal_ltcg_rate
    return marginal_ltcg_rate(taxable, inc, sched)


def compare_policies(pos: PositionFacts, spec: GlidePathSpec, sched: BracketSchedule, optimised: np.ndarray) -> pd.DataFrame:
    T = _periods(spec)
    V0 = pos.value
    growth = np.array([(1 + spec.expected_return) ** ((t + 1) / spec.periods_per_year) for t in range(T)])
    policies = {
        "optimised": optimised,
        "sell all now": np.array([V0 * growth[0]] + [0.0] * (T - 1)),
        "equal instalments": V0 * growth / T,
        "hold to horizon": np.zeros(T),
    }
    rows = []
    for name, d in policies.items():
        s = evaluate_schedule(d, pos, spec, sched)
        pv_tax = float((s["tax"] * s["discount"]).sum())
        # terminal tax on remainder at horizon (unless step-up)
        rem = float(s["remaining_value"].iloc[-1])
        gamma = 1 - pos.basis / max(V0 * growth[-1], 1e-9)
        term_tax = (1 - spec.p_stepup) * ltcg_tax(max(rem * gamma, 0.0), spec.other_taxable_income / spec.periods_per_year, sched)["total"] * float(s["discount"].iloc[-1])
        risk = float((s["risk_cost"] * s["discount"]).sum())
        alpha = float(spec.alpha_view / spec.periods_per_year * (s["remaining_value"] * s["discount"]).sum())
        feasible = True
        if spec.annual_gain_budget is not None and (s.groupby("year")["realised_gain"].sum() > spec.annual_gain_budget + 1e-6).any():
            feasible = False
        for year, frac in spec.min_sold_by.items():
            t_idx = min(int(year * spec.periods_per_year) - 1, T - 1)
            if t_idx >= 0 and s["cumulative_fraction"].iloc[t_idx] < frac - 1e-6:
                feasible = False
        rows.append({"policy": name, "feasible": feasible, "pv_tax_paid": pv_tax, "pv_terminal_tax": term_tax, "pv_risk_cost": risk, "pv_alpha_forgone": -alpha,
                     "pv_costs": float((s["cost"] * s["discount"]).sum()), "total_objective": pv_tax + term_tax + risk - alpha + float((s["cost"] * s["discount"]).sum()),
                     "final_weight": float(s["weight_after"].iloc[-1]), "total_sold": float(s["sold"].sum())})
    return pd.DataFrame(rows).sort_values(["feasible", "total_objective"], ascending=[False, True]).reset_index(drop=True)


# ====================================================================================== Monte Carlo
@dataclass
class MonteCarloSpec:
    n_paths: int = 5000
    horizon_years: int = 5
    market_return: float = 0.07
    market_vol: float = 0.16
    rf: float = 0.04
    seed: int = 11


def monte_carlo(pos: PositionFacts, gp: GlidePathSpec, mc: MonteCarloSpec, sched: BracketSchedule, optimised: np.ndarray | None = None) -> dict:
    """Simulate stock (beta to market + idiosyncratic) and diversified market; apply sell policies annually with bracket
    taxes; terminal liquidation taxed at LT unless step-up. Returns fan percentiles and terminal statistics per policy."""
    rng = np.random.default_rng(mc.seed)
    T = mc.horizon_years
    N = mc.n_paths
    zm = rng.standard_normal((N, T))
    ze = rng.standard_normal((N, T))
    mu_m = mc.market_return
    mu_s = mc.rf + pos.beta * (mu_m - mc.rf) + gp.alpha_view
    r_m = np.exp((mu_m - 0.5 * mc.market_vol ** 2) + mc.market_vol * zm) - 1
    sig_e = pos.specific_vol
    r_s = np.exp((mu_s - 0.5 * (pos.beta ** 2 * mc.market_vol ** 2 + sig_e ** 2)) + pos.beta * mc.market_vol * zm + sig_e * ze) - 1
    inc = gp.other_taxable_income
    per_year = gp.periods_per_year
    if optimised is not None and len(optimised) == T * per_year:
        opt_frac = np.array([optimised[y * per_year:(y + 1) * per_year].sum() for y in range(T)]) / np.maximum(pos.value * (1 + gp.expected_return) ** np.arange(1, T + 1), 1e-9)
        opt_frac = np.clip(opt_frac, 0, 1)
    else:
        opt_frac = np.zeros(T)
    policies = {"hold": np.zeros(T), "sell all now": np.array([1.0] + [0.0] * (T - 1)), "equal instalments": np.array([1.0 / (T - y) for y in range(T)]),
                "optimised": opt_frac}
    gamma_rate = ltcg_tax(max(pos.value - pos.basis, 1.0), inc, sched)["marginal_rate"]
    out = {}
    for name, fr in policies.items():
        stock_val = np.full(N, pos.value)
        basis = np.full(N, pos.basis)
        divers = np.zeros(N)
        fan = np.zeros((T + 1, 5))
        fan[0] = pos.value
        for y in range(T):
            frac = fr[y] if name != "optimised" else min(opt_frac[y] / max(1 - opt_frac[:y].sum(), 1e-9), 1.0) if y else opt_frac[0]
            sell = stock_val * frac
            gain = np.maximum(sell - basis * frac, 0.0)
            tax = np.array([ltcg_tax(g, inc, sched)["total"] if g > 0 else 0.0 for g in gain]) if frac > 0 else np.zeros(N)
            divers += sell - tax
            basis = basis * (1 - frac)
            stock_val = stock_val - sell
            stock_val *= 1 + r_s[:, y]
            divers *= 1 + r_m[:, y]
            total = stock_val + divers
            fan[y + 1] = np.percentile(total, [5, 25, 50, 75, 95])
        term_gain = np.maximum(stock_val - basis, 0.0)
        term_tax = (1 - gp.p_stepup) * term_gain * gamma_rate
        wealth = stock_val + divers - term_tax
        out[name] = {"terminal_after_tax": wealth, "fan": fan, "mean": float(wealth.mean()), "median": float(np.median(wealth)),
                     "p5": float(np.percentile(wealth, 5)), "p95": float(np.percentile(wealth, 95)), "std": float(wealth.std()),
                     "cvar5": float(wealth[wealth <= np.percentile(wealth, 5)].mean())}
    base = out["sell all now"]["terminal_after_tax"]
    for o in out.values():
        o["p_beats_sell_now"] = float((o["terminal_after_tax"] > base).mean())
        o.pop("terminal_after_tax")
    return {"policies": out, "years": T, "mu_stock": mu_s, "mu_market": mu_m, "sigma_stock": float(np.sqrt(pos.beta ** 2 * mc.market_vol ** 2 + sig_e ** 2))}


def tax_curve(gains: np.ndarray, other_income: float, sched: BracketSchedule) -> pd.DataFrame:
    rows = [{"gain": float(g), "tax": ltcg_tax(float(g), other_income, sched)["total"], "marginal_rate": marginal_rate_for(float(g), other_income, sched),
             "pieces_tax": tax_from_pieces(float(g), convex_pieces(sched, other_income, "ltcg"))} for g in gains]
    return pd.DataFrame(rows)


# ====================================================================================== completion portfolio
def completion_portfolio(model, locked: pd.Series, benchmark: pd.Series, universe: list[str], n_max: int = 60, max_weight: float = 0.05,
                         sector_band: float | None = 0.03, solver: str = "CLARABEL") -> dict:
    """Hold `locked` names at fixed weights (fractions of total portfolio) and choose the rest of the book to minimise
    tracking error to `benchmark`. Returns weights for the free sleeve, the full portfolio, and TE before/after."""
    locked = locked[locked > 0]
    free_budget = 1.0 - float(locked.sum())
    if free_budget <= 0:
        raise ValueError("locked names already use the whole portfolio")
    syms = sorted({s for s in universe if s in model.symbols and s not in locked.index} | {s for s in benchmark.index if s in model.symbols and s not in locked.index})
    all_syms = list(locked.index) + syms
    X = model.exposures.loc[all_syms]
    F = model.factor_cov.values
    D = model.specific_var.loc[all_syms].values
    L = np.linalg.cholesky(F + 1e-12 * np.eye(len(F)))
    wb = benchmark.reindex(all_syms).fillna(0.0).values
    wb = wb / wb.sum() if wb.sum() > 0 else wb
    nf = len(syms)
    buyable = np.array([s in set(universe) for s in syms])
    sector_cols = [c for c in model.factors if str(c).startswith(("sec:", "ind:"))]
    Xsec = X[sector_cols].values if sector_cols else np.zeros((len(all_syms), 0))

    def solve(fixed_zero: np.ndarray, band: float | None):
        w = cp.Variable(nf)
        full = cp.hstack([cp.Constant(locked.values), w])
        active = full - wb
        te2 = cp.sum_squares(L.T @ (X.values.T @ active)) + cp.sum_squares(cp.multiply(np.sqrt(D), active))
        cons = [cp.sum(w) == free_budget, w >= 0, w <= max_weight]
        z = np.where(fixed_zero | ~buyable)[0]
        if len(z):
            cons.append(w[z] == 0)
        if band is not None and sector_cols:
            cons.append(cp.abs(Xsec.T @ active) <= band)
        prob = cp.Problem(cp.Minimize(te2), cons)
        for sv in (solver, "SCS", "OSQP"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prob.solve(solver=sv, verbose=False)
                if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                    return np.clip(np.asarray(w.value).ravel(), 0, None), prob.status
            except Exception as e:  # noqa: BLE001
                log.debug("completion solver %s: %s", sv, e)
        raise RuntimeError(f"completion optimisation failed: {prob.status}")

    fixed = np.zeros(nf, dtype=bool)
    wv, status = solve(fixed, sector_band)
    for _ in range(5):
        nz = np.where(wv > 1e-6)[0]
        if len(nz) <= n_max:
            break
        keep = nz[np.argsort(wv[nz])[::-1][:n_max]]
        fixed = np.ones(nf, dtype=bool)
        fixed[keep] = False
        band = sector_band
        for attempt in range(3):
            try:
                wv, status = solve(fixed, band)
                break
            except RuntimeError:
                band = None if attempt else (band * 2 if band else None)
        else:
            t = np.zeros(nf)
            t[keep] = wv[keep]
            wv = t / t.sum() * free_budget
            status = "truncated_heuristic"
            break
    wv = np.where(wv < 1e-6, 0.0, wv)
    wv = wv / wv.sum() * free_budget if wv.sum() > 0 else wv
    free = pd.Series(wv, index=syms)
    free = free[free > 0].sort_values(ascending=False)
    full = pd.concat([locked, free])
    bench_full = pd.Series(wb, index=all_syms)
    te_after = model.tracking_error(full.reindex(all_syms).fillna(0.0), bench_full)
    locked_only = locked.reindex(all_syms).fillna(0.0)
    te_locked_alone = model.tracking_error(locked_only / locked_only.sum(), bench_full)
    return {"free_weights": free, "full_weights": full, "locked": locked, "te_after": float(te_after), "te_locked_only": float(te_locked_alone),
            "free_budget": free_budget, "n_free_names": int((free > 0).sum()), "status": status}
