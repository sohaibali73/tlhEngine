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

import numpy as np
import pandas as pd

from ..lazy import lazy_module

cp = lazy_module("cvxpy")          # imported on first use (saves ~1.7 s at launch)

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
    "multi_factor": "integrated multi-factor (value + momentum + quality + low-vol composite, sector-neutral scores) with benchmark-relative risk control",
    "defensive_equity": "low-beta, high-quality, low-volatility equity with a beta cap (AQR Defensive-style) and TE awareness",
    "quality_momentum": "quality x momentum composite tilt, benchmark-relative",
    "long_short_extension": "130/30-style: long core + long extension - short extension, beta-neutral extension, sector/factor-neutral, for continuous tax-loss generation",
    "overlay_neutral": "market-neutral long/short extension on top of existing holdings (DEALS Overlay-style): no sales of appreciated core, no shorts of held names",
    "levered_beta": "S&P 500 stocks + leveraged S&P ETFs (2x/3x) + optional Reg-T margin to hit a target beta (1.5), tracking target_beta x benchmark at minimum drag/interest cost; no futures, no shorts",
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
    # multi-factor / defensive
    integrated: bool = True                      # composite score (integrated) vs averaged single-factor sleeves (mixed)
    sector_neutral_scores: bool = True
    beta_cap: float = 0.85                       # defensive_equity: max portfolio beta to the market factor
    # long/short extension
    extension: float = 0.30                      # 130/30 -> 0.30 (long 1+ext, short ext); overlay_neutral: ext/ext
    short_max_weight: float = 0.02
    beta_target: float = 1.0                     # net beta of the long/short book
    beta_tolerance: float = 0.03
    extension_neutral: bool = True               # extension neutral to sectors (band) and style factors
    # levered beta (leveraged ETFs + margin, no futures)
    target_beta: float = 1.5
    lev_instruments: tuple[str, ...] = ("SSO", "UPRO")
    etf_max_weight: float = 0.35
    margin_max: float = 0.50                     # max loan as a fraction of equity (0 = cash only)
    margin_rate: float = 0.065
    margin_buffer: float = 0.25                  # equity >= (1 + buffer) x maintenance requirement
    cost_weight: float = 0.1                     # drag/interest weight vs tracking variance (0 = pure tracking); TE first
    replicate: bool = True                       # levered_beta: ignore n_max/sector_band and hold every index name (lowest TE)


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
    if res.diagnostics.get("levered"):
        w = res.weights[res.weights > 1e-9]
        res.weights = w.sort_values(ascending=False)          # sums to 1 + loan (fraction of equity)
        res.diagnostics.setdefault("n_names", int(len(w)))
        res.diagnostics.setdefault("max_weight", float(w.max()))
    elif res.diagnostics.get("long_short"):
        w = res.weights[res.weights.abs() > 1e-9]
        res.weights = w.sort_values(ascending=False)          # signed: net weights sum to 1
        res.diagnostics.setdefault("n_names", int((w > 0).sum()))
        res.diagnostics.setdefault("n_short", int((w < 0).sum()))
        res.diagnostics.setdefault("max_weight", float(w.max()))
    else:
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


# ====================================================================================== multi-factor family
DEFAULT_FACTOR_MIX = {"value": 1.0, "momentum": 1.0, "quality": 1.0, "lowvol": 1.0}


def _composite(spec: StrategySpec, inp: StrategyInputs, syms: list[str], weights: dict[str, float] | None = None) -> pd.Series:
    """Integrated composite z-score of the requested signals; optionally sector-neutralised (demeaned within GICS sector)."""
    if inp.signals is None or inp.signals.empty:
        raise ValueError("multi-factor strategies need style signals (inp.signals)")
    mix = weights or spec.signal_weights or DEFAULT_FACTOR_MIX
    avail = {k: v for k, v in mix.items() if k in inp.signals.columns}
    if not avail:                                    # ERM naming: lowvol -> resvol (inverted)
        alias = {"lowvol": "resvol", "value": "value", "momentum": "momentum", "quality": "quality"}
        avail = {alias.get(k, k): (-v if k == "lowvol" and alias.get(k) == "resvol" else v) for k, v in mix.items() if alias.get(k, k) in inp.signals.columns}
    if not avail:
        raise ValueError(f"none of the signals {list(mix)} are in the model; available: {list(inp.signals.columns)}")
    z = pd.Series(0.0, index=syms)
    for k, wgt in avail.items():
        s = inp.signals[k].reindex(syms).fillna(0.0)
        s = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
        z = z + wgt * s.clip(-3, 3)
    z = z / sum(abs(v) for v in avail.values())
    if spec.sector_neutral_scores and inp.sectors is not None:
        sec = inp.sectors.reindex(syms)
        z = z - z.groupby(sec).transform("mean").fillna(0.0)
    z = (z - z.mean()) / (z.std(ddof=0) + 1e-12)
    z.attrs["signals_used"] = avail
    return z


def _betas(inp: StrategyInputs, syms: list[str], S: np.ndarray | None = None) -> np.ndarray:
    """Beta of each name to the benchmark portfolio from the covariance: beta = S w_b / (w_b' S w_b). (The `market`
    column of a Barra-style exposure matrix is an intercept of ones, not a beta.)"""
    S = _cov(inp, syms) if S is None else S
    wb = inp.benchmark.reindex(syms).fillna(0.0).values if inp.benchmark is not None else np.ones(len(syms))
    wb = wb / wb.sum() if wb.sum() > 0 else np.ones(len(syms)) / len(syms)
    var_b = float(wb @ S @ wb)
    return (S @ wb) / var_b if var_b > 0 else np.ones(len(syms))


def multi_factor(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    """Integrated multi-factor: alpha = IC * sigma * composite z, benchmark-relative MVO under caps / bands / n_max.
    `integrated=False` builds one sleeve per factor and averages them (the 'mixed' approach AQR argues against)."""
    S = _cov(inp, syms)
    sig = np.sqrt(np.diag(S))
    mix = spec.signal_weights or DEFAULT_FACTOR_MIX
    if spec.integrated:
        z = _composite(spec, inp, syms, mix)
        mu = spec.ic * sig * z.values
        res = mean_variance(StrategySpec(**{**spec.__dict__, "kind": "multi_factor"}), inp, syms, mu=mu)
        res.kind = "multi_factor"
        res.diagnostics.update({"approach": "integrated", "signals_used": z.attrs.get("signals_used", {}),
                                "composite_exposure": float(z.reindex(res.weights.index).fillna(0.0) @ res.weights.values)})
        return res
    sleeves = []
    for k, wgt in mix.items():
        try:
            zk = _composite(spec, inp, syms, {k: 1.0})
        except ValueError:
            continue
        r = mean_variance(StrategySpec(**{**spec.__dict__, "kind": "multi_factor"}), inp, syms, mu=spec.ic * sig * zk.values)
        sleeves.append((abs(wgt), r.weights))
    if not sleeves:
        raise ValueError("no usable signals for the mixed multi-factor portfolio")
    tot = sum(w for w, _ in sleeves)
    w = sum(wt / tot * s.reindex(syms).fillna(0.0) for wt, s in sleeves)
    w = w[w > spec.min_weight * 0.5]
    return StrategyResult(w / w.sum(), "multi_factor", "mixed_sleeves", {"approach": "mixed", "n_sleeves": len(sleeves)})


def quality_momentum(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    spec2 = StrategySpec(**{**spec.__dict__, "kind": "quality_momentum", "signal_weights": spec.signal_weights or {"quality": 1.0, "momentum": 1.0}})
    res = multi_factor(spec2, inp, syms)
    res.kind = "quality_momentum"
    return res


def defensive_equity(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    """Low beta + quality + low residual vol with a hard beta cap: min w'Sw - kappa * score'w (+ TE penalty), beta(w) <= beta_cap."""
    S = _cov(inp, syms)
    z = _composite(spec, inp, syms, spec.signal_weights or {"lowvol": 1.0, "quality": 1.0})
    sig = np.sqrt(np.diag(S))
    mu = spec.ic * sig * z.values
    beta = _betas(inp, syms, S)

    def obj(w, a):
        o = cp.quad_form(w, cp.psd_wrap(S)) - (mu @ w) * (spec.risk_aversion / 2.0)
        if spec.te_penalty:
            o = o + spec.te_penalty * cp.quad_form(a, cp.psd_wrap(S))
        return o

    def cons(w):
        return [beta @ w <= spec.beta_cap] if beta is not None else []

    res = _qp(spec, inp, syms, obj, extra_cons=cons, kind="defensive_equity")
    if beta is not None:
        res.diagnostics["beta"] = float(beta[[syms.index(s) for s in res.weights.index]] @ res.weights.values)
    res.diagnostics["composite_exposure"] = float(z.reindex(res.weights.index).fillna(0.0) @ res.weights.values)
    return res


def _long_short_core(spec: StrategySpec, inp: StrategyInputs, syms: list[str], base: np.ndarray, long_budget: float, short_budget: float,
                     forbid_short: np.ndarray | None, kind: str) -> StrategyResult:
    """Shared solver for long/short books. Variables l >= 0 (long book / long extension), s >= 0 (shorts); net = base + l - s.

    Names are split by composite score before solving: the bottom tranche is short-eligible, the rest long-eligible, so a
    name is never held long and short at once and the budgets can be equalities. Neutrality (sector band, style penalty)
    applies to what is added to the core: the active book for a pure long/short, the l - s extension for an overlay."""
    S = _cov(inp, syms)
    n = len(syms)
    wb = inp.benchmark.reindex(syms).fillna(0.0).values
    wb = wb / wb.sum() if wb.sum() > 0 else wb
    z = _composite(spec, inp, syms, spec.signal_weights or DEFAULT_FACTOR_MIX)
    sig = np.sqrt(np.diag(S))
    mu = spec.ic * sig * z.values
    beta = _betas(inp, syms, S)
    secm = _sector_matrix(inp, syms)
    style_cols = [c for c in (inp.signals.columns if inp.signals is not None else []) if c in (inp.exposures.columns if inp.exposures is not None else [])]
    Xs = inp.exposures[style_cols].reindex(syms).fillna(0.0).values if style_cols else None
    pure_ls = not np.any(base > 1e-12)
    forbid = np.zeros(n, dtype=bool) if forbid_short is None else forbid_short.copy()
    zv = z.values
    need_short = int(np.ceil(short_budget / max(spec.short_max_weight, 1e-9))) + 2

    def solve(q: float, neutral_on: bool) -> tuple[np.ndarray, np.ndarray, str]:
        cutoff = np.quantile(zv[~forbid], q) if (~forbid).sum() > need_short else np.inf
        short_ok = (zv <= cutoff) & ~forbid
        if short_ok.sum() < need_short:                      # not enough eligible names below the cutoff: take the lowest scores
            order = np.argsort(np.where(forbid, np.inf, zv))
            short_ok = np.zeros(n, dtype=bool)
            short_ok[order[:need_short]] = True
            short_ok &= ~forbid
        long_ok = ~short_ok | (base > 1e-12)
        l_ = cp.Variable(n)
        s_ = cp.Variable(n)
        net = base + l_ - s_
        ext = l_ - s_
        active = net - wb
        cons = [l_ >= 0, s_ >= 0, cp.sum(l_) == long_budget, cp.sum(s_) == short_budget,
                base + l_ <= spec.max_weight, s_ <= spec.short_max_weight,
                l_[np.where(~long_ok)[0]] == 0, s_[np.where(~short_ok)[0]] == 0]
        cons += [beta @ net <= spec.beta_target + spec.beta_tolerance, beta @ net >= spec.beta_target - spec.beta_tolerance]
        neutral = active if pure_ls else ext
        if neutral_on and secm is not None:
            M, _ = secm
            band = spec.sector_band if spec.sector_band is not None else 0.02
            cons.append(cp.abs(M.T @ neutral) <= band)
        # NB: regularise the *net* book, never l and s separately
        obj = -(mu @ net) + 0.5 * spec.risk_aversion * cp.quad_form(active, cp.psd_wrap(S)) + 1e-4 * cp.sum_squares(net)
        if neutral_on and Xs is not None:
            obj = obj + 5.0 * cp.sum_squares(Xs.T @ neutral) / max(len(style_cols), 1)
        prob = cp.Problem(cp.Minimize(obj), cons)
        status = _solve(prob, spec.solver)
        return np.clip(np.asarray(l_.value).ravel(), 0, None), np.clip(np.asarray(s_.value).ravel(), 0, None), status

    attempts = [(0.4, spec.extension_neutral), (0.5, spec.extension_neutral), (0.6, spec.extension_neutral), (0.5, False)]
    err: Exception | None = None
    lv = sv = None
    status = "failed"
    for q, neutral_on in attempts:
        try:
            lv, sv, status = solve(q, neutral_on)
            if not neutral_on and spec.extension_neutral:
                status += " (neutrality relaxed)"
            break
        except RuntimeError as e:  # infeasible under this split: widen the short tranche, then relax neutrality
            err = e
    if lv is None:
        raise RuntimeError(f"long/short construction infeasible: {err}")
    lv = np.where(lv < spec.min_weight * 0.5, 0.0, lv)
    sv = np.where(sv < spec.min_weight * 0.5, 0.0, sv)
    if lv.sum() > 0:
        lv *= long_budget / lv.sum()
    if sv.sum() > 0:
        sv *= short_budget / sv.sum()
    net = base + lv - sv
    w = pd.Series(net, index=syms)
    long_w = pd.Series(base + lv, index=syms)
    short_w = pd.Series(sv, index=syms)
    diag = {"long_short": True, "gross": float(long_w.sum() + short_w.sum()), "net": float(w.sum()),
            "long_exposure": float(long_w.sum()), "short_exposure": float(-short_w.sum()),
            "n_long": int((long_w > 1e-9).sum()), "n_short": int((short_w > 1e-9).sum()),
            "long_weights": {k: round(float(v), 5) for k, v in long_w[long_w > 1e-9].sort_values(ascending=False).head(400).items()},
            "short_weights": {k: round(float(-v), 5) for k, v in short_w[short_w > 1e-9].sort_values(ascending=False).head(400).items()},
            "expected_alpha": float(mu @ net), "signals_used": z.attrs.get("signals_used", {}),
            "beta": float(beta @ net), "extension_beta": float(beta @ (lv - sv))}
    if secm is not None:
        M, _names = secm
        diag["extension_sector_max_abs"] = float(np.abs(M.T @ ((net - wb) if pure_ls else (lv - sv))).max())
    return StrategyResult(w, kind, status, diag)


def long_short_extension(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    """130/30 (or 145/45, 175/75, 200/100 via `extension`): long book 1+ext, short book ext, net 1, beta ~ beta_target,
    extension sector-/style-neutral so the shorts are pure alpha and a permanent source of harvestable losses."""
    n = len(syms)
    return _long_short_core(spec, inp, syms, np.zeros(n), 1.0 + spec.extension, spec.extension, None, "long_short_extension")


def overlay_neutral(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    """Market-neutral ext/ext extension around the client's existing holdings (held at current weights): the core is
    never sold (no realised gains), held names are never shorted (constructive-sale / wash risk), the extension has
    ~zero beta and neutral sectors, and every leg is a candidate loss generator."""
    if inp.current_weights is None or inp.current_weights.empty:
        raise ValueError("overlay_neutral needs current_weights (the existing holdings)")
    universe = sorted(set(syms) | set(inp.current_weights.index))
    universe = [s for s in universe if s in inp.cov.index]
    base = inp.current_weights.reindex(universe).fillna(0.0).values
    base = base / base.sum() if base.sum() > 0 else base
    held = base > 1e-9
    spec2 = StrategySpec(**{**spec.__dict__, "kind": "overlay_neutral", "beta_target": (spec.beta_target if spec.beta_target != 1.0 else float("nan"))})
    # beta target for the whole book = current beta (extension beta-neutral): compute from exposures
    beta_now = float(_betas(inp, universe) @ base)
    spec2.beta_target = beta_now
    spec2.max_weight = max(spec.max_weight, float(base.max()) + spec.short_max_weight)
    res = _long_short_core(spec2, inp, universe, base, spec.extension, spec.extension, held, "overlay_neutral")
    res.diagnostics["core_untouched"] = True
    res.diagnostics["beta_before"] = beta_now
    return res


def levered_beta(spec: StrategySpec, inp: StrategyInputs, syms: list[str]) -> StrategyResult:
    """Target beta with S&P stocks + leveraged ETFs + optional margin (optim/leverage.py). ETFs must be in the covariance
    (i.e. in the risk model); stocks are the strategy universe. Weights are fractions of equity and sum to 1 + loan."""
    from .leverage import INSTRUMENTS, LeveredBetaSpec, MarginPolicy, build_levered_beta

    etfs = [e for e in spec.lev_instruments if e in inp.cov.index]
    if not etfs and spec.margin_max <= 0:
        raise ValueError(f"none of {list(spec.lev_instruments)} are in the risk model (refresh data so the snapshot includes them) and margin is off")
    # the benchmark is the *index*: leveraged / inverse funds never belong in it (a saved levered basket set as the house
    # benchmark would otherwise make the model track itself)
    bench = inp.benchmark[[s for s in inp.benchmark.index if s not in INSTRUMENTS]] if inp.benchmark is not None else None
    if bench is None or bench.sum() <= 0:
        raise ValueError("levered_beta needs an index benchmark (S&P 500 cap weights) with no leveraged funds in it")
    bench = bench / bench.sum()
    inp = StrategyInputs(**{**inp.__dict__, "benchmark": bench})
    stocks = [x for x in syms if x not in INSTRUMENTS]
    universe = stocks + etfs
    betas = pd.Series(_betas(inp, universe), index=universe)
    pol = MarginPolicy(margin_rate=spec.margin_rate, max_loan=max(spec.margin_max, 0.0), allow_margin=spec.margin_max > 0, buffer=spec.margin_buffer)
    n_max = None if spec.replicate else spec.n_max
    band = None if spec.replicate else spec.sector_band
    min_w = 0.0 if spec.replicate else spec.min_weight          # replication holds every index name, however small
    lspec = LeveredBetaSpec(target_beta=spec.target_beta, instruments=tuple(etfs), n_max=n_max, max_weight=spec.max_weight,
                            etf_max_weight=spec.etf_max_weight, min_weight=min_w, cost_weight=spec.cost_weight, sector_band=band,
                            margin=pol, solver=spec.solver)
    proxy = next((p for p in ("SPY", "IVV", "VOO") if p in inp.cov.index), None)
    lspec.rf = float(inp.rf) if inp.rf else lspec.rf
    if inp.returns is not None:
        bw = inp.benchmark.reindex([c for c in inp.returns.columns if c in inp.benchmark.index]).fillna(0.0)
        if bw.sum() > 0:
            lspec.index_vol = float((inp.returns[bw.index].fillna(0.0) @ (bw / bw.sum())).std() * np.sqrt(252)) or lspec.index_vol
        if proxy is not None and proxy in inp.returns.columns:
            # each fund's real tracking noise versus k x the index, so the optimizer prices it (a loan on stocks has none)
            rp = inp.returns[proxy]
            for e in etfs:
                if e in inp.returns.columns:
                    d = (inp.returns[e] - INSTRUMENTS[e].leverage * rp).dropna()
                    if len(d) >= 60:
                        lspec.tracking_var[e] = float(d.var() * 252)
    out = build_levered_beta(inp.cov, inp.benchmark, stocks, betas, lspec, sectors=inp.sectors, proxy=proxy)
    diag = {"levered": True, "loan": out.loan, "gross": float(out.weights.sum()), "beta": out.beta, "target_beta": spec.target_beta,
            "te_vs_levered_benchmark": out.tracking_error, "annual_cost": out.cost, "margin": out.margin, **out.diagnostics}
    if inp.returns is not None:
        from .leverage import realised_tracking
        try:
            diag["realised"] = realised_tracking(inp.returns, out.weights, inp.benchmark, spec.target_beta, loan=out.loan, margin_rate=spec.margin_rate, proxy=proxy)
        except Exception as e:  # diagnostics never fail a build
            diag["realised"] = {"available": False, "error": repr(e)}
    return StrategyResult(out.weights, "levered_beta", out.status, diag)


_DISPATCH = {
    "equal_weight": equal_weight, "cap_weight": cap_weight, "min_variance": min_variance,
    "max_diversification": max_diversification, "risk_parity": risk_parity, "hrp": hrp, "mean_variance": mean_variance,
    "black_litterman": black_litterman, "min_cvar": min_cvar, "stratified_index": stratified_index,
    "factor_tilt": factor_tilt, "tax_aware_transition": tax_aware_transition,
    "multi_factor": multi_factor, "defensive_equity": defensive_equity, "quality_momentum": quality_momentum,
    "long_short_extension": long_short_extension, "overlay_neutral": overlay_neutral, "levered_beta": levered_beta,
}


# ====================================================================================== covariance helpers
def shrunk_sample_cov(returns: pd.DataFrame, min_obs: int = 60) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance of daily returns, annualised."""
    from sklearn.covariance import LedoitWolf
    R = returns.dropna(axis=1, thresh=min_obs).fillna(0.0)
    lw = LedoitWolf().fit(R.values)
    return pd.DataFrame(lw.covariance_ * 252, index=R.columns, columns=R.columns)
