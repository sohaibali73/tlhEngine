"""Portfolio construction strategies (model portfolios beyond min-TE).

All strategies share one contract: `run_strategy(spec, inputs) -> StrategyResult` with long-only weights that sum to
one and respect `max_weight`, an optional `n_max` name cap, and an optional sector band vs the benchmark.

Strategies
----------
equal_weight, cap_weight            baselines
min_variance                        argmin w'Σw
max_diversification                 Choueifaty-Coignard: max (w'σ)/sqrt(w'Σw)  (solved as min w'Σw s.t. w'σ = 1)
risk_parity                         equal risk contribution via the convex log-barrier form
hrp                                 hierarchical risk parity (Lopez de Prado): cluster, quasi-diagonalise, bisect
mean_variance                       Grinold alphas from style signals (mu = IC * sigma * z), benchmark-relative MVO
black_litterman                     equilibrium prior from benchmark weights + views (absolute / relative) -> MVO
min_cvar                            Rockafellar-Uryasev LP on trailing scenario returns
stratified_index                    direct-indexing sampler: sector x size strata, then min-TE reweight
factor_tilt                         min-TE with target active style exposures
tax_aware_transition                move toward a target while keeping net realised gains under a budget

Covariance comes from the caller (factor model or shrunk sample); the strategies are agnostic.
This module is AI-editable (ai/registry.py). Keep `StrategySpec`, `StrategyInputs`, `StrategyResult` stable.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STRATEGIES: dict[str, str] = {
    "equal_weight": "1/N over the (capped) universe",
    "cap_weight": "market-cap weights over the universe",
    "min_variance": "minimum total variance",
    "max_diversification": "maximum diversification ratio (Choueifaty-Coignard)",
    "risk_parity": "equal risk contribution",
    "hrp": "hierarchical risk parity (clustered inverse-variance)",
    "mean_variance": "benchmark-relative mean-variance with signal alphas (momentum/value/quality/lowvol/size/growth)",
    "black_litterman": "Black-Litterman posterior from benchmark equilibrium + views, then mean-variance",
    "min_cvar": "minimum conditional value-at-risk on trailing scenarios",
    "stratified_index": "direct-index sampling: sector x size strata, min-TE reweight",
    "factor_tilt": "min tracking error with target active style exposures",
    "tax_aware_transition": "min TE to a target portfolio subject to a net realised-gain budget and turnover cap",
}


@dataclass
class StrategySpec:
    kind: str = "min_variance"
    n_max: int | None = 50
    max_weight: float = 0.10
    min_weight: float = 0.002
    sector_band: float | None = None
    # mean-variance / BL
    risk_aversion: float = 5.0
    signal_weights: dict[str, float] = field(default_factory=lambda: {"momentum": 1.0})
    ic: float = 0.05
    benchmark_relative: bool = True
    te_penalty: float = 0.0                     # extra active-risk penalty for absolute strategies (0 = off)
    views: list[dict] = field(default_factory=list)   # [{"assets": {"AAPL": 1, "MSFT": -1}, "return": 0.03, "confidence": 0.5}]
    tau: float = 0.05
    market_sharpe: float = 0.4
    # CVaR
    cvar_alpha: float = 0.95
    # factor tilt
    tilts: dict[str, float] = field(default_factory=dict)
    tilt_weight: float = 5.0
    # transition
    target_weights: dict[str, float] | None = None
    gain_budget: float = 0.01                    # net realised gains allowed, fraction of portfolio value
    turnover_max: float | None = 0.5
    cost_bps: float = 5.0
    # stratified
    size_buckets: int = 3
    exclude: list[str] = field(default_factory=list)
    solver: str = "CLARABEL"


@dataclass
class StrategyInputs:
    symbols: list[str]
    cov: pd.DataFrame                              # annualised covariance over symbols (superset ok)
    benchmark: pd.Series                           # weights
    returns: pd.DataFrame | None = None            # trailing daily returns (for CVaR / HRP)
    signals: pd.DataFrame | None = None            # symbols x style z-scores
    exposures: pd.DataFrame | None = None          # symbols x factors (styles + sec:*), for tilts / bands
    sectors: pd.Series | None = None               # symbol -> sector
    mktcap: pd.Series | None = None
    current_weights: pd.Series | None = None
    gain_frac: pd.Series | None = None             # unrealised gain per $ of position (transition)
    rf: float = 0.0


@dataclass
class StrategyResult:
    weights: pd.Series
    kind: str
    status: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_names(self) -> int:
        return int((self.weights > 0).sum())


# ====================================================================================== entry point
def run_strategy(spec: StrategySpec, inp: StrategyInputs) -> StrategyResult:
    fn = _DISPATCH.get(spec.kind)
    if fn is None:
        raise ValueError(f"unknown strategy '{spec.kind}'; choose from {sorted(STRATEGIES)}")
    syms = [s for s in inp.symbols if s in inp.cov.index and s not in set(spec.exclude)]
    if len(syms) < 2:
        raise ValueError("universe too small")
    res = fn(spec, inp, syms)
    w = res.weights[res.weights > 0].sort_values(ascending=False)
    res.weights = w / w.sum()
    res.diagnostics.setdefault("n_names", int(len(w)))
    res.diagnostics.setdefault("max_weight", float(w.max()))
    if inp.benchmark is not None:
        a = _align(res.weights, inp.benchmark)
        S = inp.cov.reindex(index=a.index, columns=a.index).fillna(0.0).values
        res.diagnostics["tracking_error"] = float(np.sqrt(max(a.values @ S @ a.values, 0.0)))
    Sw = inp.cov.reindex(index=res.weights.index, columns=res.weights.index).values
    res.diagnostics["volatility"] = float(np.sqrt(max(res.weights.values @ Sw @ res.weights.values, 0.0)))
    return res


def _align(w: pd.Series, b: pd.Series) -> pd.Series:
    idx = w.index.union(b.index)
    return w.reindex(idx).fillna(0.0) - (b.reindex(idx).fillna(0.0) / max(b.sum(), 1e-12))


# ====================================================================================== generic QP core
def _solve(prob: cp.Problem, solver: str) -> str:
    for sv in (solver, "SCS", "OSQP"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=sv, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return prob.status
        except Exception as e:  # solver capability / availability
            log.debug("solver %s: %s", sv, e)
    raise RuntimeError(f"optimisation failed: {prob.status}")


def _sector_matrix(inp: StrategyInputs, syms: list[str]) -> tuple[np.ndarray, list[str]] | None:
    if inp.sectors is None:
        return None
    sec = inp.sectors.reindex(syms)
    names = sorted(sec.dropna().unique())
    if not names:
        return None
    M = np.array([[1.0 if sec[s] == g else 0.0 for g in names] for s in syms])
    return M, names


def _qp(spec: StrategySpec, inp: StrategyInputs, syms: list[str], objective, extra_cons=None, kind: str = "",
        prune: bool = True) -> StrategyResult:
    """Solve min objective(w) over long-only, budget, cap, sector-band constraints with n_max pruning.

    `objective(w, active)` returns a cvxpy expression; `extra_cons(w)` optional list of constraints."""
    n = len(syms)
    wb = inp.benchmark.reindex(syms).fillna(0.0).values if inp.benchmark is not None else np.zeros(n)
    bench_in_universe = float(wb.sum()) > 1e-9
    wb = wb / wb.sum() if bench_in_universe else wb
    secm = _sector_matrix(inp, syms)
    if not bench_in_universe and spec.sector_band is not None:
        # e.g. benchmark is a single ETF: no stock-level weights to be neutral against, so the band is meaningless
        spec = StrategySpec(**{**spec.__dict__, "sector_band": None})
        log.info("sector band ignored: benchmark has no constituents inside the universe")

    def solve(fixed_zero: np.ndarray, band: float | None) -> tuple[np.ndarray, str]:
        w = cp.Variable(n)
        active = w - wb
        cons = [cp.sum(w) == 1, w >= 0, w <= spec.max_weight]
        z = np.where(fixed_zero)[0]
        if len(z):
            cons.append(w[z] == 0)
        if band is not None and secm is not None:
            M, _ = secm
            cons.append(cp.abs(M.T @ active) <= band)
        if extra_cons:
            cons += extra_cons(w)
        prob = cp.Problem(cp.Minimize(objective(w, active)), cons)
        status = _solve(prob, spec.solver)
        return np.clip(np.asarray(w.value).ravel(), 0, None), status

    fixed = np.zeros(n, dtype=bool)
    wv, status = solve(fixed, spec.sector_band)
    if prune and spec.n_max:
        for _ in range(6):
            nz = np.where(wv > 1e-6)[0]
            if len(nz) <= spec.n_max and (wv[nz] >= spec.min_weight).all():
                break
            keep = nz[np.argsort(wv[nz])[::-1][: spec.n_max]]
            fixed = np.ones(n, dtype=bool)
            fixed[keep] = False
            band = spec.sector_band
            for attempt in range(4):
                try:
                    wv, status = solve(fixed, band)
                    if attempt:
                        status += f" (sector band {f'relaxed to {band:.3f}' if band is not None else 'dropped'})"
                    break
                except RuntimeError:
                    band = None if (band is None or attempt >= 2) else band * 2
            else:
                t = np.zeros(n)
                t[keep] = wv[keep]
                wv, status = t / t.sum(), "truncated_heuristic"
                break
    wv = np.where(wv < spec.min_weight * 0.5, 0.0, wv)
    return StrategyResult(pd.Series(wv / wv.sum(), index=syms), kind or spec.kind, status)


def _psd(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 1e-10, None)
    return vecs @ np.diag(vals) @ vecs.T


def _cov(inp: StrategyInputs, syms: list[str]) -> np.ndarray:
    return _psd(inp.cov.reindex(index=syms, columns=syms).fillna(0.0).values)


# ====================================================================================== strategies
def equal_weight(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    pick = syms
    if spec.n_max and len(syms) > spec.n_max:
        mc = inp.mktcap.reindex(syms).fillna(0.0) if inp.mktcap is not None else pd.Series(1.0, index=syms)
        pick = list(mc.sort_values(ascending=False).index[: spec.n_max])
    w = pd.Series(1.0 / len(pick), index=pick).clip(upper=spec.max_weight)
    return StrategyResult(w / w.sum(), "equal_weight", "closed_form")


def cap_weight(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    if inp.mktcap is None:
        raise ValueError("cap_weight needs mktcap")
    mc = inp.mktcap.reindex(syms).dropna()
    mc = mc[mc > 0].sort_values(ascending=False)
    if spec.n_max:
        mc = mc.iloc[: spec.n_max]
    w = mc / mc.sum()
    for _ in range(20):                                  # iterative capping
        over = w > spec.max_weight
        if not over.any():
            break
        excess = (w[over] - spec.max_weight).sum()
        w[over] = spec.max_weight
        under = ~over
        w[under] += excess * w[under] / w[under].sum()
    return StrategyResult(w, "cap_weight", "closed_form")


def min_variance(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    S = _cov(inp, syms)
    return _qp(spec, inp, syms, lambda w, a: cp.quad_form(w, cp.psd_wrap(S)) + spec.te_penalty * cp.quad_form(a, cp.psd_wrap(S)))


def max_diversification(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    """max (w'sigma)/sqrt(w'Sw)  <=>  min y'Sy s.t. y'sigma = 1, y >= 0; w = y / sum(y). Caps applied after."""
    S = _cov(inp, syms)
    sig = np.sqrt(np.diag(S))
    n = len(syms)
    y = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(S))), [y >= 0, sig @ y == 1])
    _solve(prob, spec.solver)
    w0 = np.clip(np.asarray(y.value).ravel(), 0, None)
    w0 = w0 / w0.sum()
    # enforce caps / n_max / sector band by a min-distance QP around the unconstrained solution
    res = _qp(spec, inp, syms, lambda w, a: cp.sum_squares(w - w0) + 1e-3 * cp.quad_form(w, cp.psd_wrap(S)), kind="max_diversification")
    wv = res.weights.values
    res.diagnostics["diversification_ratio"] = float((wv @ sig[[syms.index(s) for s in res.weights.index]]) / np.sqrt(wv @ _cov(inp, list(res.weights.index)) @ wv))
    return res


def risk_parity(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    """Equal risk contribution: min 0.5 w'Sw - (1/n) sum log w_i, then normalise. Names chosen by min-variance
    pruning when n_max < universe (ERC over hundreds of names is rarely intended)."""
    pick = syms
    if spec.n_max and len(syms) > spec.n_max:
        pre = _qp(StrategySpec(kind="min_variance", n_max=spec.n_max, max_weight=max(spec.max_weight, 2.0 / spec.n_max),
                               sector_band=spec.sector_band, solver=spec.solver), inp, syms,
                  lambda w, a: cp.quad_form(w, cp.psd_wrap(_cov(inp, syms))))
        pick = list(pre.weights[pre.weights > 1e-6].index)
    S = _cov(inp, pick)
    n = len(pick)
    y = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(S)) - (1.0 / n) * cp.sum(cp.log(y))), [y >= 1e-8])
    status = _solve(prob, spec.solver)
    w = np.clip(np.asarray(y.value).ravel(), 1e-10, None)
    w = w / w.sum()
    w = np.minimum(w, spec.max_weight)
    w = w / w.sum()
    rc = w * (S @ w) / (w @ S @ w)
    return StrategyResult(pd.Series(w, index=pick), "risk_parity", status,
                          {"risk_contrib_min": float(rc.min()), "risk_contrib_max": float(rc.max())})


def hrp(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    pick = syms
    if spec.n_max and len(syms) > spec.n_max:
        mc = inp.mktcap.reindex(syms).fillna(0.0) if inp.mktcap is not None else pd.Series(1.0, index=syms)
        pick = list(mc.sort_values(ascending=False).index[: spec.n_max])
    S = _cov(inp, pick)
    sd = np.sqrt(np.diag(S))
    C = np.clip(S / np.outer(sd, sd), -1, 1)
    D = np.sqrt(np.clip(0.5 * (1 - C), 0, None))
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="single")
    order = list(leaves_list(Z))

    def ivp(idx):
        v = np.diag(S)[idx]
        w = 1.0 / v
        return w / w.sum()

    def cvar(idx):
        w = ivp(idx)
        return float(w @ S[np.ix_(idx, idx)] @ w)

    w = pd.Series(1.0, index=range(len(pick)))
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            a, b = c[:half], c[half:]
            va, vb = cvar(a), cvar(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    weights = pd.Series(w.values, index=pick)
    weights = weights.clip(upper=spec.max_weight)
    weights = weights / weights.sum()
    return StrategyResult(weights, "hrp", "closed_form", {"n_clusters_leaf_order": len(order)})


def _alphas(spec: StrategySpec, inp: StrategyInputs, syms: list[str], S: np.ndarray) -> np.ndarray:
    if inp.signals is None or inp.signals.empty:
        raise ValueError("mean_variance needs style signals (inp.signals)")
    z = pd.Series(0.0, index=syms)
    tot = 0.0
    for k, wgt in spec.signal_weights.items():
        if k in inp.signals:
            z = z + wgt * inp.signals[k].reindex(syms).fillna(0.0)
            tot += abs(wgt)
    z = z / max(tot, 1e-12)
    z = (z - z.mean()) / (z.std(ddof=0) + 1e-12)
    sig = np.sqrt(np.diag(S))
    return spec.ic * sig * z.values                       # Grinold: alpha = IC * vol * score


def mean_variance(spec: StrategySpec, inp: StrategyInputs, syms: list[str], mu: np.ndarray | None = None) -> StrategyResult:
    S = _cov(inp, syms)
    mu = _alphas(spec, inp, syms, S) if mu is None else mu
    lam = spec.risk_aversion

    def obj(w, a):
        risk = cp.quad_form(a, cp.psd_wrap(S)) if spec.benchmark_relative else cp.quad_form(w, cp.psd_wrap(S))
        return -(mu @ w) + 0.5 * lam * risk

    res = _qp(spec, inp, syms, obj, kind=spec.kind)
    res.diagnostics["expected_alpha"] = float(mu[[syms.index(s) for s in res.weights.index]] @ res.weights.values)
    return res


def black_litterman(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    S = _cov(inp, syms)
    wb = inp.benchmark.reindex(syms).fillna(0.0).values
    wb = wb / wb.sum() if wb.sum() > 0 else np.ones(len(syms)) / len(syms)
    sigma_b = float(np.sqrt(wb @ S @ wb))
    delta = spec.market_sharpe / max(sigma_b, 1e-6)
    pi = delta * S @ wb                                    # equilibrium excess returns
    if not spec.views:
        mu = pi
    else:
        P, Q, conf = [], [], []
        for v in spec.views:
            row = np.zeros(len(syms))
            for sym, wt in (v.get("assets") or {}).items():
                if sym.upper() in syms:
                    row[syms.index(sym.upper())] = float(wt)
            if not row.any():
                continue
            P.append(row)
            Q.append(float(v.get("return", 0.0)))
            conf.append(float(np.clip(v.get("confidence", 0.5), 0.01, 0.99)))
        if P:
            P = np.array(P)
            Q = np.array(Q)
            tS = spec.tau * S
            omega = np.diag(np.diag(P @ tS @ P.T) * (1 - np.array(conf)) / np.array(conf))
            M = np.linalg.inv(np.linalg.inv(tS) + P.T @ np.linalg.inv(omega) @ P)
            mu = M @ (np.linalg.inv(tS) @ pi + P.T @ np.linalg.inv(omega) @ Q)
        else:
            mu = pi
    spec2 = StrategySpec(**{**spec.__dict__, "kind": "black_litterman", "risk_aversion": delta if spec.risk_aversion <= 0 else spec.risk_aversion,
                            "benchmark_relative": False})
    res = mean_variance(spec2, inp, syms, mu=mu)
    res.kind = "black_litterman"
    res.diagnostics.update({"delta": float(delta), "n_views": int(len(spec.views)),
                            "posterior_mu_top": {syms[i]: float(mu[i]) for i in np.argsort(mu)[::-1][:5]}})
    return res


def min_cvar(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    if inp.returns is None:
        raise ValueError("min_cvar needs trailing returns")
    R = inp.returns.reindex(columns=syms).dropna(how="all").fillna(0.0)
    if len(R) < 60:
        raise ValueError("min_cvar needs >= 60 return observations")
    T = len(R)
    Rv = R.values
    alpha = spec.cvar_alpha
    var_ = cp.Variable()
    u = cp.Variable(T)

    def obj(w, a):
        return var_ + cp.sum(u) / ((1 - alpha) * T)

    def cons(w):
        return [u >= 0, u >= -(Rv @ w) - var_]

    res = _qp(spec, inp, syms, obj, extra_cons=cons, kind="min_cvar")
    port = Rv @ res.weights.reindex(syms).fillna(0.0).values
    q = np.quantile(port, 1 - alpha)
    res.diagnostics.update({"daily_var": float(-q), "daily_cvar": float(-port[port <= q].mean())})
    return res


def stratified_index(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    """Pick names per (sector, size bucket) in proportion to benchmark weight, then min-TE reweight."""
    if inp.sectors is None or inp.mktcap is None:
        raise ValueError("stratified_index needs sectors and mktcap")
    n_target = spec.n_max or 50
    bw = inp.benchmark.reindex(syms).fillna(0.0)
    bw = bw / bw.sum() if bw.sum() > 0 else pd.Series(1.0 / len(syms), index=syms)
    df = pd.DataFrame({"w": bw, "sector": inp.sectors.reindex(syms), "mc": inp.mktcap.reindex(syms)}).dropna()
    df["bucket"] = pd.qcut(df["mc"].rank(method="first"), q=min(spec.size_buckets, max(len(df) // 10, 1)), labels=False)
    picks: list[str] = []
    strata = df.groupby(["sector", "bucket"])["w"].sum().sort_values(ascending=False)
    alloc = np.maximum(np.round(strata / strata.sum() * n_target), 1).astype(int)
    for (sector, bucket), k in alloc.items():
        grp = df[(df["sector"] == sector) & (df["bucket"] == bucket)].sort_values("mc", ascending=False)
        picks += list(grp.index[: int(k)])
    picks = list(dict.fromkeys(picks))[: max(n_target, 1)]
    S = _cov(inp, picks)
    sub = StrategyInputs(symbols=picks, cov=inp.cov, benchmark=inp.benchmark, sectors=inp.sectors, mktcap=inp.mktcap)
    spec2 = StrategySpec(**{**spec.__dict__, "kind": "stratified_index", "n_max": None})
    res = _qp(spec2, sub, picks, lambda w, a: cp.quad_form(a, cp.psd_wrap(S)), kind="stratified_index", prune=False)
    res.diagnostics["strata"] = int(len(alloc))
    return res


def factor_tilt(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    if inp.exposures is None:
        raise ValueError("factor_tilt needs exposures")
    S = _cov(inp, syms)
    X = inp.exposures.reindex(syms).fillna(0.0)

    def obj(w, a):
        o = cp.quad_form(a, cp.psd_wrap(S))
        for k, tgt in spec.tilts.items():
            if k in X.columns:
                o = o + spec.tilt_weight * cp.square(X[k].values @ a - float(tgt))
        return o

    return _qp(spec, inp, syms, obj, kind="factor_tilt")


def tax_aware_transition(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    """min (w - wt)'S(w - wt) + cost*turnover  s.t.  net realised gain <= gain_budget, turnover <= cap.

    Trades are explicit: w = w0 - sell + buy with 0 <= sell <= w0, buy >= 0, so realised gain = gain_frac . sell is
    LINEAR (losses offset gains inside the budget). A name may not be sold and bought in the same rebalance (that
    would be a wash sale, not a harvest): after each solve, names with both legs active get buy fixed to zero and
    the problem is re-solved (at most 4 passes)."""
    if inp.current_weights is None or spec.target_weights is None:
        raise ValueError("tax_aware_transition needs current_weights and target_weights")
    tgt = pd.Series(spec.target_weights, dtype=float)
    tgt.index = [str(s).upper() for s in tgt.index]
    universe = sorted(set(syms) | set(inp.current_weights.index) | set(tgt.index))
    universe = [s for s in universe if s in inp.cov.index]
    S = _cov(inp, universe)
    w0 = inp.current_weights.reindex(universe).fillna(0.0).values
    w0 = w0 / w0.sum() if w0.sum() > 0 else w0
    wt = tgt.reindex(universe).fillna(0.0).values
    wt = wt / wt.sum() if wt.sum() > 0 else wt
    g = inp.gain_frac.reindex(universe).fillna(0.0).values if inp.gain_frac is not None else np.zeros(len(universe))
    n = len(universe)
    cap = max(spec.max_weight, float(np.max(w0)) + 1e-9)
    no_buy = np.zeros(n, dtype=bool)
    status = "failed"
    for _ in range(4):
        sell = cp.Variable(n)
        buy = cp.Variable(n)
        w = w0 - sell + buy
        cons = [sell >= 0, sell <= w0, buy >= 0, cp.sum(w) == 1, w <= cap, g @ sell <= spec.gain_budget]
        if no_buy.any():
            cons.append(buy[np.where(no_buy)[0]] == 0)
        turnover = (cp.sum(sell) + cp.sum(buy)) / 2
        if spec.turnover_max is not None:
            cons.append(turnover <= spec.turnover_max)
        d = w - wt
        prob = cp.Problem(cp.Minimize(cp.quad_form(d, cp.psd_wrap(S)) + (spec.cost_bps / 1e4) * turnover), cons)
        status = _solve(prob, spec.solver)
        sv = np.clip(np.asarray(sell.value).ravel(), 0, None)
        bv = np.clip(np.asarray(buy.value).ravel(), 0, None)
        churn = (sv > 1e-6) & (bv > 1e-6)
        if not churn.any():
            break
        no_buy |= churn
    wv = np.clip(w0 - sv + bv, 0, None)
    wv = np.where(wv < 1e-6, 0.0, wv)
    wv = wv / wv.sum()
    sold_v = np.clip(w0 - wv, 0, None)
    gains_idx, loss_idx = g > 0, g < 0
    res = StrategyResult(pd.Series(wv, index=universe), "tax_aware_transition", status, {
        "realised_gain_frac": float((g * sold_v).sum()),
        "realised_gains_only": float((g[gains_idx] * sold_v[gains_idx]).sum()),
        "realised_losses_only": float((g[loss_idx] * sold_v[loss_idx]).sum()),
        "gain_budget": float(spec.gain_budget),
        "turnover": float(np.abs(wv - w0).sum() / 2),
        "te_to_target": float(np.sqrt(max((wv - wt) @ S @ (wv - wt), 0.0))),
        "te_to_target_before": float(np.sqrt(max((w0 - wt) @ S @ (w0 - wt), 0.0))),
        "names_blocked_from_rebuy": int(no_buy.sum()),
    })
    return res


_DISPATCH = {
    "equal_weight": equal_weight, "cap_weight": cap_weight, "min_variance": min_variance,
    "max_diversification": max_diversification, "risk_parity": risk_parity, "hrp": hrp, "mean_variance": mean_variance,
    "black_litterman": black_litterman, "min_cvar": min_cvar, "stratified_index": stratified_index,
    "factor_tilt": factor_tilt, "tax_aware_transition": tax_aware_transition,
}


# ====================================================================================== covariance helpers
def shrunk_sample_cov(returns: pd.DataFrame, min_obs: int = 60) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance of daily returns, annualised."""
    from sklearn.covariance import LedoitWolf
    R = returns.dropna(axis=1, thresh=min_obs).fillna(0.0)
    lw = LedoitWolf().fit(R.values)
    return pd.DataFrame(lw.covariance_ * 252, index=R.columns, columns=R.columns)
