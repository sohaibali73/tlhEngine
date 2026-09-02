"""Model-portfolio (basket) construction.

Given a fitted risk model, a benchmark and a buyable universe, build a long-only basket that minimises tracking error
to the benchmark subject to: name count, max/min weight, sector neutrality band, optional style tilts (target active
exposures), exclusions. Also exposes `analyze_basket` for TE / exposures / sector diagnostics of any weight vector.

Formulation (cvxpy QP):
    minimise   TE^2(w - wb) + lam_tilt * sum_k (x_k'(w - wb) - tilt_k)^2 + lam_conc * ||w||^2
    subject to sum w = 1, 0 <= w <= max_weight, |sector active| <= sector_band, w[excluded] = 0
Name count is enforced by iterative pruning: solve, keep the largest `n_max` weights, fix the rest at zero, re-solve.

This module is AI-editable (ai/registry.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from ..risk.model import FittedRiskModel

log = logging.getLogger(__name__)


@dataclass
class BasketSpec:
    n_max: int = 50
    max_weight: float = 0.08
    min_weight: float = 0.002            # weights below this are pruned in the final pass
    sector_band: float = 0.02            # |sector active weight| cap (None disables)
    tilts: dict[str, float] = field(default_factory=dict)     # style -> target active exposure (z units)
    tilt_weight: float = 5.0
    concentration_weight: float = 0.05
    exclude: list[str] = field(default_factory=list)
    include_only: list[str] | None = None                      # restrict universe to these symbols
    exclude_etps: bool = True
    solver: str = "CLARABEL"


@dataclass
class BasketResult:
    weights: pd.Series
    tracking_error: float
    exposures: pd.DataFrame          # factor x [basket, benchmark, active]
    sectors: pd.DataFrame            # sector x [basket, benchmark, active]
    n_names: int
    status: str
    spec: BasketSpec

    def metrics(self) -> dict:
        return {"tracking_error": self.tracking_error, "n_names": self.n_names, "status": self.status,
                "max_weight": float(self.weights.max()) if len(self.weights) else 0.0,
                "active_style": {k: float(v) for k, v in self.exposures["active"].items() if not str(k).startswith(("sec:", "ind:")) and k != "market"},
                "max_sector_active": float(self.sectors["active"].abs().max()) if len(self.sectors) else 0.0}


def build_basket(model: FittedRiskModel, benchmark: pd.Series, universe: list[str] | None, spec: BasketSpec,
                 securities: pd.DataFrame | None = None) -> BasketResult:
    known = set(model.symbols)
    if universe is None:
        universe = list(model.symbols)
    uni = [s for s in universe if s in known and s not in set(spec.exclude)]
    if spec.include_only:
        inc = set(spec.include_only)
        uni = [s for s in uni if s in inc]
    if spec.exclude_etps and securities is not None and "subtype1" in securities:
        etp = securities["subtype1"].fillna("").str.lower().str.startswith("exchange traded")
        uni = [s for s in uni if not (s in etp.index and bool(etp.get(s, False)))]
    syms = sorted(set(uni) | {s for s in benchmark.index if s in known})
    if len(syms) < 2:
        raise ValueError("universe too small after filters")
    buyable = np.array([s in set(uni) for s in syms])
    n = len(syms)
    wb = benchmark.reindex(syms).fillna(0.0).values
    wb = wb / wb.sum() if wb.sum() > 0 else wb
    X = model.exposures.loc[syms]
    F = model.factor_cov.values
    D = model.specific_var.loc[syms].values
    L = np.linalg.cholesky(F + 1e-12 * np.eye(len(F)))
    sector_cols = [c for c in model.factors if c.startswith(("sec:", "ind:"))]
    Xsec = X[sector_cols].values if sector_cols else np.zeros((n, 0))

    def solve(fixed_zero: np.ndarray, sector_band: float | None = spec.sector_band) -> tuple[np.ndarray, str]:
        w = cp.Variable(n)
        active = w - wb
        te2 = cp.sum_squares(L.T @ (X.values.T @ active)) + cp.sum_squares(cp.multiply(np.sqrt(D), active))
        obj = te2 + spec.concentration_weight * cp.sum_squares(w)
        for k, tgt in spec.tilts.items():
            if k in X.columns:
                obj = obj + spec.tilt_weight * cp.square(X[k].values @ active - float(tgt))
        cons = [cp.sum(w) == 1, w >= 0, w <= spec.max_weight]
        zero_idx = np.where(fixed_zero | ~buyable)[0]
        if len(zero_idx):
            cons.append(w[zero_idx] == 0)
        if sector_cols and sector_band is not None:
            cons.append(cp.abs(Xsec.T @ active) <= sector_band)
        prob = cp.Problem(cp.Minimize(obj), cons)
        import warnings
        for sv in (spec.solver, "OSQP", "SCS"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prob.solve(solver=sv, verbose=False)
                if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                    return np.clip(np.asarray(w.value).ravel(), 0, None), prob.status
            except Exception as e:  # pragma: no cover
                log.warning("basket solver %s failed: %s", sv, e)
        raise RuntimeError(f"basket optimisation failed: {prob.status}")

    fixed = np.zeros(n, dtype=bool)
    wv, status = solve(fixed)
    for _ in range(6):
        nz = np.where(wv > 1e-6)[0]
        if len(nz) <= spec.n_max and (wv[nz] >= spec.min_weight).all():
            break
        keep = nz[np.argsort(wv[nz])[::-1][: spec.n_max]]
        keep = keep[wv[keep] >= spec.min_weight] if len(keep) > spec.n_max // 2 else keep
        fixed = np.ones(n, dtype=bool)
        fixed[keep] = False
        # With few names the sector band can become infeasible: relax it progressively, then fall back to a
        # renormalised truncation of the last feasible solution (flagged in status).
        band = spec.sector_band
        for attempt in range(4):
            try:
                wv, status = solve(fixed, band)
                if attempt:
                    status = f"{status} (sector band relaxed to {band:.3f})" if band is not None else f"{status} (sector band dropped)"
                break
            except RuntimeError:
                band = None if (band is None or attempt >= 2) else band * 2
        else:
            trunc = np.zeros(n)
            trunc[keep] = wv[keep]
            wv = trunc / trunc.sum()
            status = "truncated_heuristic"
            break
    wv = np.where(wv < spec.min_weight * 0.5, 0.0, wv)
    wv = wv / wv.sum()
    weights = pd.Series(wv, index=syms)
    weights = weights[weights > 0].sort_values(ascending=False)
    return analyze_basket(model, weights, benchmark, spec=spec, status=status)


def analyze_basket(model: FittedRiskModel, weights: pd.Series, benchmark: pd.Series, spec: BasketSpec | None = None,
                   status: str = "given") -> BasketResult:
    syms = sorted({s for s in weights.index if s in model.symbols} | {s for s in benchmark.index if s in model.symbols})
    w = weights.reindex(syms).fillna(0.0)
    w = w / w.sum() if w.sum() > 0 else w
    b = benchmark.reindex(syms).fillna(0.0)
    b = b / b.sum() if b.sum() > 0 else b
    te = model.tracking_error(w, b)
    pe, be = model.portfolio_exposures(w), model.portfolio_exposures(b)
    exp = pd.DataFrame({"basket": pe, "benchmark": be})
    exp["active"] = exp["basket"] - exp["benchmark"]
    sector_cols = [c for c in model.factors if c.startswith(("sec:", "ind:"))]
    sec = exp.loc[sector_cols].copy() if sector_cols else pd.DataFrame(columns=["basket", "benchmark", "active"])
    sec.index = [c.split(':', 1)[1] for c in sec.index]
    exp = exp.drop(index=sector_cols)
    return BasketResult(weights=w[w > 0].sort_values(ascending=False), tracking_error=te, exposures=exp, sectors=sec,
                        n_names=int((w > 0).sum()), status=status, spec=spec or BasketSpec())
