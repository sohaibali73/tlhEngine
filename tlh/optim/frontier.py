"""Tax-alpha vs tracking-error frontier and constraint-hierarchy comparison.

`frontier()` sweeps the TE budget (hard cap) across a grid and records harvested loss, tax alpha and realised TE.
`priority_comparison()` runs the optimizer under each of the six orderings of the constraint hierarchy so the
operator can see how the trade list changes when tax alpha, tracking error or factor neutrality comes first.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from .harvest import PRIORITIES, HarvestConfig, HarvestInputs, HarvestResult, run_harvest


def frontier(inp: HarvestInputs, cfg: HarvestConfig, te_grid: list[float] | None = None) -> pd.DataFrame:
    grid = te_grid or list(np.round(np.linspace(0.0025, 0.03, 12), 5))
    rows = []
    for te in grid:
        c = replace(cfg, te_budget=float(te), te_hard=True)
        try:
            res = run_harvest(inp, c)
            s = res.summary
            rows.append({"te_budget": te, "te_after": s["te_after"], "harvested_loss": s["harvested_loss"],
                         "tax_alpha": s["tax_alpha"], "tax_alpha_bps": s["tax_alpha_bps"], "turnover": s["turnover"],
                         "n_trades": s["n_sells"] + s["n_buys"], "status": res.solver_status})
        except Exception as e:  # keep the sweep alive
            rows.append({"te_budget": te, "status": f"error: {e}"})
    return pd.DataFrame(rows)


def priority_comparison(inp: HarvestInputs, cfg: HarvestConfig) -> tuple[pd.DataFrame, dict[str, HarvestResult]]:
    results: dict[str, HarvestResult] = {}
    rows = []
    for order in itertools.permutations(PRIORITIES):
        key = " > ".join(_short(p) for p in order)
        c = replace(cfg, priority=order)
        try:
            res = run_harvest(inp, c)
            results[key] = res
            s = res.summary
            rows.append({"priority": key, "harvested_loss": s["harvested_loss"], "tax_alpha": s["tax_alpha"],
                         "te_after": s["te_after"], "max_style_drift": s["max_style_drift"],
                         "max_sector_drift": s["max_sector_drift"], "turnover": s["turnover"],
                         "n_trades": s["n_sells"] + s["n_buys"], "status": res.solver_status})
        except Exception as e:
            rows.append({"priority": key, "status": f"error: {e}"})
    return pd.DataFrame(rows), results


def _short(p: str) -> str:
    return {"tax": "Tax", "tracking_error": "TE", "factor_neutrality": "Factor"}[p]
