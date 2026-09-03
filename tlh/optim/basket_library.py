"""Sample model portfolios: a library of named baskets any advisor can build with one click.

Each entry is a construction recipe (strategy kind + parameters + benchmark) with a plain-English pitch. Building the
library runs every recipe against the live snapshot and active risk model through the same StrategyService the
Strategy lab uses, so each result is a normal saved basket that can become the benchmark or a harvest target.

This module is AI-editable (ai/registry.py): YANG can add recipes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BasketRecipe:
    name: str
    kind: str
    pitch: str                                   # one sentence for a client
    params: dict = field(default_factory=dict)
    benchmark: str | None = None                 # None -> household default benchmark
    audience: str = "core"                       # core | income | defensive | growth | tax | long_short
    cov_source: str = "model"


LIBRARY: list[BasketRecipe] = [
    BasketRecipe("Core 500 Tracker (50)", "stratified_index",
                 "Fifty stocks that track the S&P 500 within about 1% a year: the direct-indexing core that harvesting runs on.",
                 {"n_max": 50, "max_weight": 0.06, "sector_band": 0.015, "size_buckets": 3}),
    BasketRecipe("Core 500 Tracker (100)", "stratified_index",
                 "A hundred-name index sleeve with tighter tracking and more lots to harvest.",
                 {"n_max": 100, "max_weight": 0.05, "sector_band": 0.01, "size_buckets": 4}),
    BasketRecipe("Multi-Factor Integrated (60)", "multi_factor",
                 "Value, momentum, quality and low-volatility combined into one score per stock, sector-neutral, risk-controlled versus the index.",
                 {"n_max": 60, "max_weight": 0.05, "sector_band": 0.03, "ic": 0.05, "risk_aversion": 8.0, "integrated": True}, audience="growth"),
    BasketRecipe("Multi-Factor Mixed Sleeves (60)", "multi_factor",
                 "The same four factors built as separate sleeves and averaged, for comparison with the integrated version.",
                 {"n_max": 60, "max_weight": 0.05, "sector_band": 0.03, "integrated": False}, audience="growth"),
    BasketRecipe("Defensive Equity (45)", "defensive_equity",
                 "Lower-beta, higher-quality, calmer stocks with a hard beta cap: equity exposure with a smaller drawdown profile.",
                 {"n_max": 45, "max_weight": 0.06, "sector_band": 0.05, "beta_cap": 0.8, "risk_aversion": 4.0, "te_penalty": 2.0}, audience="defensive"),
    BasketRecipe("Quality Momentum (40)", "quality_momentum",
                 "Profitable companies whose prices are already trending: the two factors that historically pair best.",
                 {"n_max": 40, "max_weight": 0.06, "sector_band": 0.04, "ic": 0.06, "risk_aversion": 6.0}, audience="growth"),
    BasketRecipe("Minimum Variance (40)", "min_variance",
                 "The lowest-risk combination of forty stocks the model can find, with a light pull toward the index.",
                 {"n_max": 40, "max_weight": 0.06, "sector_band": 0.06, "te_penalty": 0.5}, audience="defensive"),
    BasketRecipe("Maximum Diversification (50)", "max_diversification",
                 "Weights set so no single risk dominates: the most diversified fifty-name portfolio.",
                 {"n_max": 50, "max_weight": 0.06}, audience="defensive"),
    BasketRecipe("Risk Parity Equities (35)", "risk_parity",
                 "Every holding contributes the same amount of risk.",
                 {"n_max": 35, "max_weight": 0.08}, audience="defensive"),
    BasketRecipe("Hierarchical Risk Parity (50)", "hrp",
                 "Clusters similar stocks, then spreads risk across clusters: robust to estimation error.",
                 {"n_max": 50, "max_weight": 0.06}, audience="defensive"),
    BasketRecipe("Value Tilt (50)", "factor_tilt",
                 "Index-like portfolio leaning toward cheaper stocks by about half a standard deviation.",
                 {"n_max": 50, "max_weight": 0.06, "sector_band": 0.02, "tilts": {"value": 0.5}, "tilt_weight": 5.0}, audience="core"),
    BasketRecipe("Low-Vol Income Tilt (50)", "factor_tilt",
                 "Index-like with a lean to steadier, lower-volatility names favoured by income-oriented clients.",
                 {"n_max": 50, "max_weight": 0.06, "sector_band": 0.03, "tilts": {"lowvol": 0.5, "quality": 0.25}, "tilt_weight": 5.0}, audience="income"),
    BasketRecipe("Growth Tilt (50)", "factor_tilt",
                 "Index-like with a lean to faster-growing companies.",
                 {"n_max": 50, "max_weight": 0.06, "sector_band": 0.03, "tilts": {"growth": 0.5, "momentum": 0.25}, "tilt_weight": 5.0}, audience="growth"),
    BasketRecipe("Minimum CVaR (40)", "min_cvar",
                 "Built to shrink the worst 5% of months rather than average volatility.",
                 {"n_max": 40, "max_weight": 0.06, "cvar_alpha": 0.95}, audience="defensive"),
    BasketRecipe("Black-Litterman Equilibrium (60)", "black_litterman",
                 "Starts from what the index implies about expected returns and lets the optimiser express it in sixty names.",
                 {"n_max": 60, "max_weight": 0.05, "risk_aversion": 0.0}, audience="core"),
    BasketRecipe("Long/Short 130/30 Tax Engine", "long_short_extension",
                 "A full index-like long book plus a 30% long / 30% short extension that keeps generating short-term losses in up and down markets.",
                 {"n_max": 150, "max_weight": 0.04, "short_max_weight": 0.015, "extension": 0.30, "sector_band": 0.02, "ic": 0.05, "risk_aversion": 10.0},
                 audience="long_short"),
    BasketRecipe("Levered Beta 1.5 (S&P stocks + 2x/3x ETFs)", "levered_beta",
                 "Every S&P 500 stock at index weight plus leveraged S&P ETFs and up to 50% Reg-T margin to run a 1.5 beta at the lowest tracking error to 1.5x the index; no futures, no shorts.",
                 {"replicate": True, "max_weight": 0.10, "target_beta": 1.5, "lev_instruments": ["SSO", "UPRO"], "margin_max": 0.5, "cost_weight": 0.1},
                 audience="growth"),
    BasketRecipe("Levered Beta 1.5 cash-only (S&P stocks + 2x/3x ETFs)", "levered_beta",
                 "The same 1.5 beta with no borrowing at all: leveraged ETFs carry the extra exposure.",
                 {"replicate": True, "max_weight": 0.10, "target_beta": 1.5, "lev_instruments": ["SSO", "UPRO"], "margin_max": 0.0, "cost_weight": 0.1},
                 audience="growth"),
    BasketRecipe("Long/Short 145/45 Tax Engine", "long_short_extension",
                 "More extension, more loss generation (and more financing and tracking error).",
                 {"n_max": 180, "max_weight": 0.04, "short_max_weight": 0.015, "extension": 0.45, "sector_band": 0.02, "ic": 0.05, "risk_aversion": 10.0},
                 audience="long_short"),
]


def recipes(audience: str | None = None) -> list[BasketRecipe]:
    return [r for r in LIBRARY if audience is None or r.audience == audience]


def recipe_table():
    import pandas as pd
    return pd.DataFrame([{"name": r.name, "strategy": r.kind, "audience": r.audience, "pitch": r.pitch, "params": r.params} for r in LIBRARY])
