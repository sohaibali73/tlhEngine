"""Which files the embedded co-pilot may read and propose changes to, and which tests gate each one.

Anything not listed here is read-only to the AI (it can still be read via `read_module` if in READ_ONLY).
The wash-sale suite is always run as a canary regardless of the module touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CANARY_TESTS = ["tests/test_washsale.py", "tests/test_holding.py", "tests/test_ledger.py"]


@dataclass(frozen=True)
class EditableModule:
    path: str
    description: str
    tests: list[str] = field(default_factory=list)
    kind: str = "python"          # python | yaml
    reload_hint: str = ""


AI_EDITABLE: dict[str, EditableModule] = {
    "tlh/risk/factors.py": EditableModule(
        "tlh/risk/factors.py",
        "Style factor definitions (raw score functions + STYLE_DEFINITIONS registry) and exposure standardisation.",
        ["tests/test_risk.py"], reload_hint="Refit the risk model after promotion to see new factors."),
    "tlh/risk/descriptors.py": EditableModule(
        "tlh/risk/descriptors.py", "ERM descriptor library and style composites (size, beta, momentum, resvol, value, quality, growth, liquidity, leverage).",
        ["tests/test_erm.py"], reload_hint="Refit an ERM model after promotion."),
    "tlh/risk/erm.py": EditableModule(
        "tlh/risk/erm.py", "Equity risk model estimator: WLS/Huber factor returns, EWMA vol x corr + Newey-West, eigen-adjust, VRA, specific-risk shrinkage.",
        ["tests/test_erm.py"], reload_hint="Refit an ERM model after promotion."),
    "tlh/risk/analytics.py": EditableModule(
        "tlh/risk/analytics.py", "Risk decomposition, factor stress tests, historical scenarios, parametric VaR, bias tests.",
        ["tests/test_erm.py"]),
    "tlh/risk/model.py": EditableModule(
        "tlh/risk/model.py",
        "Barra-style fit: cross-sectional WLS, EWMA covariance, specific risk, ETF regression fill, macro block.",
        ["tests/test_risk.py"], reload_hint="Refit the risk model after promotion."),
    "tlh/optim/harvest.py": EditableModule(
        "tlh/optim/harvest.py",
        "Harvest optimizer formulation: objective terms, constraints, replacement search, post-processing.",
        ["tests/test_harvest.py"], reload_hint="Re-run harvest after promotion."),
    "tlh/optim/frontier.py": EditableModule(
        "tlh/optim/frontier.py", "TE-budget frontier sweep and constraint-priority comparison.",
        ["tests/test_harvest.py"]),
    "tlh/optim/basket.py": EditableModule(
        "tlh/optim/basket.py", "Model-portfolio construction: min-TE basket with name caps, sector band, style tilts.",
        ["tests/test_basket.py"], reload_hint="Rebuild baskets after promotion."),
    "tlh/optim/strategies.py": EditableModule(
        "tlh/optim/strategies.py",
        "Construction strategies: min-var, max-div, risk parity, HRP, mean-variance, Black-Litterman, min-CVaR, stratified index, factor tilt, tax-aware transition. Add a strategy = function + _DISPATCH entry + STRATEGIES description.",
        ["tests/test_strategies.py"], reload_hint="Rebuild strategy baskets after promotion."),
    "tlh/optim/pipeline.py": EditableModule(
        "tlh/optim/pipeline.py", "TLH model pipeline schema (blocks, validation, filter/rank helpers, examples) behind the drag-and-drop builder.",
        ["tests/test_pipeline.py"]),
    "tlh/tax/concentration.py": EditableModule(
        "tlh/tax/concentration.py", "Bracket-aware tax engine (LTCG/ordinary/NIIT, convex pieces), Black-Scholes collars with §1259/§1092 flags, charitable/gift/exchange-fund/step-up comparisons.",
        ["tests/test_concentration.py"]),
    "tlh/research/engine.py": EditableModule(
        "tlh/research/engine.py", "TLH research simulator: monthly lot-level harvesting with wash windows, whole shares, pairs / twin-basket / optimizer reinvestment, metrics.",
        ["tests/test_research.py"], reload_hint="Re-run the study after promotion (results are cached per parameter set)."),
    "tlh/research/spec.py": EditableModule(
        "tlh/research/spec.py", "Research parameter grids (account sizes, basket sizes, triggers, approaches, concentrated grid) and the approach descriptions.",
        ["tests/test_research.py"]),
    "tlh/optim/glidepath.py": EditableModule(
        "tlh/optim/glidepath.py", "Multi-period tax-aware diversification glide path (convex) and Monte Carlo policy comparison.",
        ["tests/test_concentration.py"]),
    "tlh/optim/backtest.py": EditableModule(
        "tlh/optim/backtest.py", "Walk-forward backtester (rebalance schedule, trailing covariance/signals, costs, metrics).",
        ["tests/test_strategies.py"]),
    "tlh/data/substitutes.yaml": EditableModule(
        "tlh/data/substitutes.yaml",
        "Substantially-identical groups and wash-safe substitute candidates.",
        ["tests/test_substitutes.py"], kind="yaml", reload_hint="Substitute map reloads immediately on promotion."),
    "tlh/risk/statistical.py": EditableModule(
        "tlh/risk/statistical.py",
        "Statistical risk models (calibrated covariance, Ledoit-Wolf, PCA, hybrid residual factors) and dynamic covariance (GARCH, regime).",
        ["tests/test_statistical.py"], reload_hint="Refit a statistical / dynamic model after promotion."),
    "tlh/risk/calibration.py": EditableModule(
        "tlh/risk/calibration.py", "Walk-forward calibration study (lookback x weighting x estimator x horizon) and substitute-pair TE study.",
        ["tests/test_statistical.py"]),
    "tlh/optim/basket_library.py": EditableModule(
        "tlh/optim/basket_library.py", "Sample model-portfolio recipes (name, strategy, params, pitch). Add a recipe = one BasketRecipe entry.",
        ["tests/test_flagship.py"]),
    "tlh/optim/longshort.py": EditableModule(
        "tlh/optim/longshort.py", "Long/short TLH economics: loss-generation Monte Carlo, financing, tax-neutral Exchange glide.",
        ["tests/test_flagship.py"]),
    "tlh/optim/overlay.py": EditableModule(
        "tlh/optim/overlay.py", "Index-futures beta overlay sizing, margin, carry, §1256 / §1092 flags.",
        ["tests/test_flagship.py"]),
    "tlh/optim/leverage.py": EditableModule(
        "tlh/optim/leverage.py", "Levered-beta construction (leveraged ETFs + Reg-T margin, no futures), margin policy, tactical overlay sizing, daily simulator.",
        ["tests/test_leverage.py"]),
    "tlh/optim/tactical.py": EditableModule(
        "tlh/optim/tactical.py", "Tactical signal sources (manual, Potomac CSV, example rules, blends) that produce a target-beta series.",
        ["tests/test_leverage.py"], reload_hint="Re-save signals after promotion."),
    "tlh/tax/state_rates.yaml": EditableModule(
        "tlh/tax/state_rates.yaml", "State capital-gains tax treatment table (51 jurisdictions, approximate planning figures).",
        ["tests/test_flagship.py"], kind="yaml", reload_hint="Restart or re-open the Tax rates screen to reload."),
}

READ_ONLY: list[str] = [
    "tlh/tax/washsale.py", "tlh/tax/ledger.py", "tlh/tax/lots.py", "tlh/tax/holding.py", "tlh/tax/rates.py", "tlh/tax/state_rates.py",
    "tlh/risk/benchmark.py", "tlh/data/norgate.py", "tlh/data/cache.py", "tlh/data/substitutes.py",
    "tlh/services/harvest_service.py", "tlh/services/risk_service.py", "tlh/services/home_service.py", "tlh/services/import_service.py",
    "tlh/explain.py", "DECISIONS.md", "CLAUDE.md",
]

NEW_FILE_PREFIXES = ["tlh/risk/custom/"]   # the AI may create new modules here (tests required)


def is_editable(path: str) -> bool:
    p = path.replace("\\", "/")
    return p in AI_EDITABLE or any(p.startswith(pre) and p.endswith(".py") for pre in NEW_FILE_PREFIXES)


def is_readable(path: str) -> bool:
    p = path.replace("\\", "/")
    return is_editable(p) or p in READ_ONLY or p.startswith("tests/")


def tests_for(path: str) -> list[str]:
    p = path.replace("\\", "/")
    mod = AI_EDITABLE.get(p)
    own = list(mod.tests) if mod else ["tests/test_risk.py"]
    return list(dict.fromkeys(own + CANARY_TESTS))
