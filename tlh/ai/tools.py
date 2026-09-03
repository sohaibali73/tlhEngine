"""Tool schemas and executor for the embedded co-pilot.

Every tool is a method on `ToolExecutor`; `TOOLS` is the JSON schema list sent to the API. Tools never place
orders. Anything that changes state (baskets, runs, model fits, proposed changes) is written through the normal
repositories so it is audited and visible in the GUI.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..services.context import AppContext
from . import sandbox
from .registry import AI_EDITABLE, CANARY_TESTS, NEW_FILE_PREFIXES, is_editable, is_readable

log = logging.getLogger(__name__)

TOOLS: list[dict[str, Any]] = [
    # ---- read / context
    {"name": "get_portfolio_context", "description": "Current entity's positions, lot summary, harvestable losses, wash calendar, tax profile and carryforwards.",
     "input_schema": {"type": "object", "properties": {"include_lots": {"type": "boolean", "description": "Include lot-level rows (default false)."}}}},
    {"name": "get_risk_model_summary", "description": "Active risk model: spec, diagnostics, factor vols, portfolio-vs-benchmark exposures and TE decomposition.",
     "input_schema": {"type": "object", "properties": {"model_id": {"type": "integer", "description": "Specific model version; default active."}}}},
    {"name": "list_models", "description": "All fitted risk-model versions with diagnostics.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_run", "description": "A harvest run (default latest): summary, trades, blocked lots, replacement candidates.",
     "input_schema": {"type": "object", "properties": {"run_id": {"type": "integer"}}}},
    {"name": "list_runs", "description": "Recent runs (harvest, frontier, ai_plan) with summaries.", "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "search_securities", "description": "Search the data snapshot by symbol/name/GICS sector or industry. Returns symbol, name, sector, industry, market cap, last price, 1y return, 1y vol.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "substring of symbol or name (optional)"}, "sector": {"type": "string"},
                                                        "industry": {"type": "string"}, "min_mktcap_musd": {"type": "number"}, "limit": {"type": "integer"}}}},
    {"name": "get_price_stats", "description": "Return statistics for symbols over a lookback: annualised return/vol, max drawdown, correlation matrix (<= 25 symbols).",
     "input_schema": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}}, "lookback_days": {"type": "integer"}}, "required": ["symbols"]}},
    {"name": "get_substitutes", "description": "Substantially-identical group and wash-safe replacement candidates for a symbol.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "explain_wash_status", "description": "Wash-sale screen for one open lot as of today with the full determination.",
     "input_schema": {"type": "object", "properties": {"lot_id": {"type": "integer"}}, "required": ["lot_id"]}},
    {"name": "screen_trade", "description": "Pre-trade wash-sale screen for a hypothetical trade (SELL of a lot/symbol at a loss, or BUY of a symbol) as of a date.",
     "input_schema": {"type": "object", "properties": {"side": {"type": "string", "enum": ["BUY", "SELL"]}, "symbol": {"type": "string"},
                                                        "quantity": {"type": "number"}, "lot_id": {"type": "integer"}, "account_id": {"type": "integer"},
                                                        "trade_date": {"type": "string", "description": "YYYY-MM-DD, default today"}}, "required": ["side", "symbol"]}},
    # ---- harvest & plans
    {"name": "run_harvest", "description": "Run the harvest optimizer for the current entity with config overrides (mode, priority, te_budget, te_hard, sector_drift_max, turnover_max, min_trade_value, min_loss_value, cost_bps, tax_horizon_years, target_loss, benchmark). Persists a run and returns its summary + trades.",
     "input_schema": {"type": "object", "properties": {"overrides": {"type": "object", "description": "HarvestConfig fields to override; `benchmark` may be a watchlist, ETF, or 'basket:<name>'."},
                                                        "notes": {"type": "string"}}}},
    {"name": "run_frontier", "description": "Sweep hard TE budgets and return tax alpha vs TE.",
     "input_schema": {"type": "object", "properties": {"te_grid": {"type": "array", "items": {"type": "number"}}, "overrides": {"type": "object"}}}},
    {"name": "evaluate_trade_list", "description": "Evaluate a custom trade plan you design: wash-screens every trade, computes realised gain/term per sell, TE and exposures before/after, and saves it as an 'ai_plan' run the user can review/export/book. Sells should reference lot_id when possible.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"},
                                                        "trades": {"type": "array", "items": {"type": "object", "properties": {"side": {"type": "string", "enum": ["BUY", "SELL"]}, "symbol": {"type": "string"},
                                                                                                                              "quantity": {"type": "number"}, "lot_id": {"type": "integer"}, "account_id": {"type": "integer"}},
                                                                                              "required": ["side", "symbol", "quantity"]}},
                                                        "benchmark": {"type": "string"}, "rationale": {"type": "string"}}, "required": ["name", "trades"]}},
    # ---- baskets / model portfolios
    {"name": "list_baskets", "description": "Saved model portfolios (baskets).", "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_basket", "description": "Weights and metrics of a basket.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "create_basket", "description": "Save a model portfolio from explicit weights (normalised to 1). Returns TE and exposures vs benchmark.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "weights": {"type": "object", "description": "symbol -> weight"},
                                                        "description": {"type": "string"}, "benchmark": {"type": "string"}}, "required": ["name", "weights"]}},
    {"name": "optimize_basket", "description": "Construct a model portfolio that minimises TE to a benchmark subject to name count, weight caps, sector band, style tilts (target active z-exposures, e.g. {'quality': 0.3, 'lowvol': 0.2}) and exclusions. Saves it as a basket.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "benchmark": {"type": "string", "description": "watchlist, ETF or basket:<name>; default current"},
                                                        "universe": {"type": "array", "items": {"type": "string"}, "description": "restrict to these symbols (default: model universe)"},
                                                        "n_max": {"type": "integer"}, "max_weight": {"type": "number"}, "sector_band": {"type": "number"},
                                                        "tilts": {"type": "object"}, "tilt_weight": {"type": "number"}, "exclude": {"type": "array", "items": {"type": "string"}},
                                                        "exclude_held": {"type": "boolean", "description": "exclude names currently held (useful for replacement baskets)"},
                                                        "exclude_wash_blocked": {"type": "boolean", "description": "exclude names whose purchase would trigger a wash sale today (default true)"},
                                                        "description": {"type": "string"}}, "required": ["name"]}},
    {"name": "analyze_basket", "description": "TE, style/sector exposures of a basket vs a benchmark.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "benchmark": {"type": "string"}}, "required": ["name"]}},
    {"name": "set_benchmark", "description": "Set the app's active benchmark (watchlist, ETF ticker, or basket:<name>). Affects TE everywhere.",
     "input_schema": {"type": "object", "properties": {"benchmark": {"type": "string"}}, "required": ["benchmark"]}},
    # ---- strategies & backtests
    {"name": "list_strategies", "description": "Portfolio construction strategies available to build_strategy_basket / backtest_strategy, with their parameters.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "build_strategy_basket", "description": "Build a model portfolio with a named strategy (min_variance, max_diversification, risk_parity, hrp, mean_variance, black_litterman, min_cvar, stratified_index, factor_tilt, tax_aware_transition, equal_weight, cap_weight, multi_factor, defensive_equity, quality_momentum, long_short_extension, overlay_neutral, levered_beta) and save it as a basket. `params` are StrategySpec fields (n_max, max_weight, sector_band, signal_weights, ic, risk_aversion, views, cvar_alpha, tilts, target_weights, gain_budget, turnover_max, exclude...). For tax_aware_transition pass target_basket or params.target_weights. levered_beta (no futures, no shorts): params target_beta (1.5), replicate (default true: every S&P name at index weight, lowest tracking error; set false to use n_max), margin_max (loan as a fraction of equity, 0 = cash-only so 2x/3x ETFs carry the leverage), lev_instruments, etf_max_weight, cost_weight (0 = pure tracking); always built against the S&P index benchmark; diagnostics include model TE and a realised check with actual ETF histories.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "strategy": {"type": "string"}, "params": {"type": "object"},
                                                        "benchmark": {"type": "string"}, "universe": {"type": "array", "items": {"type": "string"}},
                                                        "target_basket": {"type": "string"}, "cov_source": {"type": "string", "enum": ["model", "sample"]},
                                                        "description": {"type": "string"}, "save": {"type": "boolean"}}, "required": ["name", "strategy"]}},
    {"name": "backtest_strategy", "description": "Walk-forward backtest of a strategy on the snapshot history (monthly/quarterly/weekly rebalances, trailing shrunk covariance and signals, costs). Returns CAGR, vol, Sharpe, max drawdown, TE, IR, turnover vs benchmark and saves a 'backtest' run. Note caveats in the result (survivorship, fundamental look-ahead).",
     "input_schema": {"type": "object", "properties": {"strategy": {"type": "string"}, "params": {"type": "object"}, "name": {"type": "string"},
                                                        "start": {"type": "string"}, "end": {"type": "string"}, "rebalance": {"type": "string", "enum": ["M", "Q", "W"]},
                                                        "lookback_days": {"type": "integer"}, "cost_bps": {"type": "number"},
                                                        "benchmark_symbol": {"type": "string", "description": "ETF ticker in the snapshot, e.g. SPY; omit for cap-weighted proxy"}},
                      "required": ["strategy"]}},
    {"name": "get_backtest", "description": "Metrics, warnings and last weights of a saved backtest run.",
     "input_schema": {"type": "object", "properties": {"run_id": {"type": "integer"}}, "required": ["run_id"]}},
    # ---- TLH model pipelines
    {"name": "pipeline_schema", "description": "Block types and parameters for TLH model pipelines (drag-and-drop builder). Read before save_pipeline.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_pipelines", "description": "Saved TLH model pipelines.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "save_pipeline", "description": "Save a TLH model pipeline as JSON: {name, description, nodes:[{id,type,params,x,y}]} with types universe|filter|rank|benchmark|construct|transition|harvest|output ordered by increasing x. Validates and returns errors if any.",
     "input_schema": {"type": "object", "properties": {"pipeline": {"type": "object"}}, "required": ["pipeline"]}},
    {"name": "run_pipeline", "description": "Execute a saved pipeline (by name) or an inline pipeline object end to end: screens, construction, optional tax-aware transition, optional harvest run, save/export. Returns basket name, harvest run id, weights and the run log.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "pipeline": {"type": "object"}}}},
    # ---- concentration & embedded gains
    {"name": "concentration_overview", "description": "Per-position weight, embedded gain (ST/LT), tax if liquidated (bracket-aware, stacked on other income), risk share, specific vol, beta; HHI, effective N, gain-weighted concentration, locked-in %.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "diversification_plan", "description": "Optimise a multi-year tax-aware glide path to reduce a concentrated position: bracket taxes with loss offsets and carryforwards, ST->LT timing, step-up probability, alpha view, risk aversion, annual gain budget, minimum diversification by year. Returns the schedule, policy comparison and tax curve.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "horizon_years": {"type": "integer"}, "periods_per_year": {"type": "integer"},
                                                        "other_taxable_income": {"type": "number"}, "risk_aversion": {"type": "number"}, "alpha_view": {"type": "number"},
                                                        "expected_return": {"type": "number"}, "discount_rate": {"type": "number"}, "p_stepup": {"type": "number"},
                                                        "annual_gain_budget": {"type": "number"}, "min_sold_by": {"type": "object"}, "use_expected_losses": {"type": "boolean"}},
                      "required": ["symbol"]}},
    {"name": "concentration_monte_carlo", "description": "Monte Carlo of after-tax terminal wealth for hold / sell now / equal instalments / optimised schedule on a concentrated position (stock = beta x market + idiosyncratic from the risk model).",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "horizon_years": {"type": "integer"}, "alpha_view": {"type": "number"}, "p_stepup": {"type": "number"},
                                                        "n_paths": {"type": "integer"}, "market_return": {"type": "number"}, "market_vol": {"type": "number"}}, "required": ["symbol"]}},
    {"name": "hedge_analysis", "description": "Collar (or protective put) economics for a concentrated position with Black-Scholes pricing, zero-cost call strike, payoff table and tax flags (constructive sale §1259 band, straddle §1092 on short-term lots, covered-call holding period).",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "tenor_years": {"type": "number"}, "put_strike_pct": {"type": "number"},
                                                        "call_strike_pct": {"type": "number", "description": "omit for zero-cost"}, "sigma": {"type": "number"}, "rate": {"type": "number"}, "div_yield": {"type": "number"}},
                      "required": ["symbol"]}},
    {"name": "concentration_alternatives", "description": "Sell now vs donate appreciated shares vs sell-then-donate, gifting to a lower bracket, Section 721 exchange fund breakeven, and step-up-at-death option value for a position.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "agi": {"type": "number"}, "p_stepup": {"type": "number"}, "horizon_years": {"type": "number"}}, "required": ["symbol"]}},
    {"name": "gain_offset_plan", "description": "Build a reviewable trade plan that sells a dollar amount of a concentrated name from highest-basis taxable lots and pairs it with wash-safe harvestable loss sales so net realised gain is minimised (optional replacement buy). Saved as an ai_plan run.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "sell_value": {"type": "number"}, "replacement": {"type": "string"}, "offset_with_losses": {"type": "boolean"}, "name": {"type": "string"}},
                      "required": ["symbol", "sell_value"]}},
    {"name": "completion_portfolio", "description": "Hold the named concentrated positions at their current weights and optimise the rest of the book to minimise tracking error (name cap, weight cap, sector band). Optionally save as a basket to harvest toward.",
     "input_schema": {"type": "object", "properties": {"locked_symbols": {"type": "array", "items": {"type": "string"}}, "n_max": {"type": "integer"}, "max_weight": {"type": "number"},
                                                        "sector_band": {"type": "number"}, "save_as": {"type": "string"}}, "required": ["locked_symbols"]}},
    # ---- risk analytics
    {"name": "risk_decomposition", "description": "Decompose the current holdings' total or active risk into market/style/industry/specific, per-factor contributions and per-holding marginal contributions.",
     "input_schema": {"type": "object", "properties": {"active": {"type": "boolean", "description": "vs benchmark (default true)"}}}},
    {"name": "stress_test", "description": "Factor stress test on current holdings. shocks: {factor: sigma_units} or {'<factor>:raw': return}. Factor names as in the active model (market, value, momentum, quality, size, lowvol/resvol, growth, liquidity, leverage, beta, midcap, sec:<Sector>/ind:<Industry Group>, macro:*). Presets: Market -2σ, Momentum crash, Value rally / growth sell-off, Flight to quality, Small-cap squeeze, Rates +100bp, Liquidity shock.",
     "input_schema": {"type": "object", "properties": {"shocks": {"type": "object"}, "preset": {"type": "string"}, "propagate": {"type": "boolean"}, "active": {"type": "boolean"}}}},
    {"name": "historical_scenario", "description": "Replay the model's factor returns between two dates through today's exposures.",
     "input_schema": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}, "active": {"type": "boolean"}}, "required": ["start", "end"]}},
    {"name": "parametric_var", "description": "Parametric VaR and expected shortfall of current holdings from the risk model.",
     "input_schema": {"type": "object", "properties": {"horizon_days": {"type": "integer"}, "alpha": {"type": "number"}, "active": {"type": "boolean"}}}},
    {"name": "validate_risk_model", "description": "Out-of-sample bias test: refit the model at the start of each of n periods and compare predicted vs realised volatility for the market, equal-weight, holdings and style portfolios. Slow (one fit per period).",
     "input_schema": {"type": "object", "properties": {"n_periods": {"type": "integer"}, "period_days": {"type": "integer"}, "model_overrides": {"type": "object"}}}},
    # ---- risk models
    {"name": "fit_risk_model", "description": "Fit a new risk-model version on the latest snapshot with spec overrides (lookback_days, halflife_days, styles, use_sectors, use_macro, cov_shrink, specific_shrink, exposure_refresh_days). Not activated unless activate=true. Returns diagnostics.",
     "input_schema": {"type": "object", "properties": {"overrides": {"type": "object"}, "name": {"type": "string"}, "activate": {"type": "boolean"}, "notes": {"type": "string"}}}},
    {"name": "compare_models", "description": "Compare factor vols, R² and portfolio TE between two model versions.",
     "input_schema": {"type": "object", "properties": {"model_a": {"type": "integer"}, "model_b": {"type": "integer"}}, "required": ["model_a", "model_b"]}},
    {"name": "list_style_factors", "description": "Registered style factors (built-in + custom plugins) with descriptions.", "input_schema": {"type": "object", "properties": {}}},
    # ---- code
    {"name": "list_editable_modules", "description": "Modules you may modify (with gating tests) and where new modules may be created.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "read_module", "description": "Read a repo file (editable modules, key read-only modules, tests, DECISIONS.md).",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "test_change", "description": "Apply full-file content in a sandbox and run the gating tests. Iterate here; no change record is created.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "code": {"type": "string"}, "extra_tests": {"type": "array", "items": {"type": "string"}}}, "required": ["path", "code"]}},
    {"name": "propose_change", "description": "Create a reviewable change (diff + sandbox results) for human approval. New factor modules go in tlh/risk/custom/<name>.py. Call once after tests pass.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "code": {"type": "string"}, "title": {"type": "string"}, "rationale": {"type": "string"}},
                      "required": ["path", "code", "title", "rationale"]}},
    {"name": "run_analysis", "description": "Run a Python script in the sandbox against a COPY of the state DB and read-only snapshot/model folders (SNAPSHOTS_DIR, MODELS_DIR, RUNS_DIR predefined; `tlh` importable). Print results.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}, "timeout_s": {"type": "integer"}}, "required": ["code"]}},
    # ---- model library, calibration, sample baskets, long/short, overlay, taxes, onboarding
    {"name": "risk_model_presets", "description": "The risk-model library: 13 named presets (ERM standard/short/long/robust/GARCH/regime, hybrid ERM+statistical, Potomac calibrated covariances, tight-pair sample, PCA, barra_lite) with descriptions. Use a preset name in fit_risk_model's `preset`.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_calibration_study", "description": "Walk-forward calibration of covariance lookback x weighting x estimator x horizon (port of the 2026 Potomac study): scoreboard with bias ratios, Spearman ranks, TE bias, composite Score and a recommendation. quick=true (≈1–2 min) or full grid (several minutes).",
     "input_schema": {"type": "object", "properties": {"quick": {"type": "boolean"}, "include_pca": {"type": "boolean"}, "include_holdings": {"type": "boolean"},
                                                        "lookbacks": {"type": "array", "items": {"type": "integer"}}, "horizons": {"type": "array", "items": {"type": "integer"}},
                                                        "fit_recommendation": {"type": "boolean", "description": "fit and activate the recommended calibrated model"}}}},
    {"name": "pair_te_study", "description": "Forecast vs realised tracking error of tight substitute pairs (e.g. IVV vs SPY) by estimator: shows why the sample covariance beats shrinkage for near-identical pairs.",
     "input_schema": {"type": "object", "properties": {"pairs": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}, "horizon_days": {"type": "integer"}}}},
    {"name": "build_sample_baskets", "description": "Build the sample model-portfolio library (index trackers, integrated multi-factor, defensive, quality-momentum, risk parity, HRP, style tilts, min-CVaR, Black-Litterman, 130/30 and 145/45 long/short tax engines) against the live snapshot. Optional audience filter: core | growth | defensive | income | long_short, or explicit names.",
     "input_schema": {"type": "object", "properties": {"audience": {"type": "string"}, "names": {"type": "array", "items": {"type": "string"}}, "benchmark": {"type": "string"}}}},
    {"name": "longshort_analysis", "description": "Economics of a long/short TLH extension (130/30, 145/45, 175/75, 200/100): Monte Carlo of net loss generation by year vs long-only, financing cost, tax benefit net of financing, and the published Quantinno reference profile.",
     "input_schema": {"type": "object", "properties": {"extension": {"type": "number", "description": "0.30 = 130/30"}, "years": {"type": "integer"}, "market_return": {"type": "number"},
                                                        "market_vol": {"type": "number"}, "n_paths": {"type": "integer"}}}},
    {"name": "exchange_glide", "description": "Tax-neutral divestiture of a concentrated position using a long/short extension's expected losses (DEALS Exchange-style): year-by-year sells, gains, losses, net tax ≈ 0, years to full diversification.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string", "description": "held position (uses its market value and basis)"}, "position_value": {"type": "number"},
                                                        "cost_basis": {"type": "number"}, "extension": {"type": "number"}, "years": {"type": "integer"}}}},
    {"name": "overlay_plan", "description": "Size an index-futures beta overlay (ES/MES/NQ/MNQ/RTY/M2K) to restore or reduce portfolio beta: contracts, notional, margin, carry, §1256 60/40 tax treatment and straddle flags. Uses the risk model's implied beta of current holdings unless portfolio_beta is given.",
     "input_schema": {"type": "object", "properties": {"target_beta": {"type": "number"}, "contract": {"type": "string"}, "cash": {"type": "number"}, "index_level": {"type": "number"},
                                                        "portfolio_beta": {"type": "number"}, "days": {"type": "integer"}}}},
    {"name": "state_tax_rates", "description": "Capital-gains tax treatment for every US state + DC (approximate 2026 planning figures): state ST/LT rates, treatment (ordinary / none / exclusion / flat / excise), combined federal+NIIT+state marginal rates. Pass a state for the detailed combined calculation at a given income.",
     "input_schema": {"type": "object", "properties": {"state": {"type": "string"}, "filing_status": {"type": "string"}, "other_income": {"type": "number"}, "gain": {"type": "number"}}}},
    {"name": "set_tax_setup", "description": "Set the household's tax profile from state + filing status + other taxable income (derives federal, NIIT and state marginal rates; syncs the bracket schedule). Use when the advisor gives you the client's state.",
     "input_schema": {"type": "object", "properties": {"state": {"type": "string"}, "filing_status": {"type": "string", "enum": ["single", "mfj", "mfs", "hoh"]}, "other_income": {"type": "number"}},
                      "required": ["state", "filing_status", "other_income"]}},
    {"name": "one_click_harvest", "description": "The Start-here flow: refresh data if stale, fit a risk model if none, run the wash-safe harvest with the saved configuration and return a plain-English summary plus the tickets.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "leverage_instruments", "description": "Leveraged / inverse S&P 500 and Nasdaq-100 ETFs the engine can use instead of futures (leverage, expense ratio), the margin policy (Reg-T initial, maintenance by leverage, buffer, rate, max loan) and the custodian capability table.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "tactical_signal", "description": "Create or update a target-beta signal for the tactical overlay: kind manual (one beta), csv (a Potomac strategy export with date + target_beta/state/score), rule:trend | rule:vol_regime | rule:composite | rule:drawdown (example rules, not Potomac models), or blend (weighted average of saved signals). Optionally make it active.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "kind": {"type": "string"}, "manual_beta": {"type": "number"}, "path": {"type": "string"},
                                                        "beta_min": {"type": "number"}, "beta_max": {"type": "number"},
                                                        "components": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "weight": {"type": "number"}}}},
                                                        "activate": {"type": "boolean"}, "description": {"type": "string"}}, "required": ["name", "kind"]}},
    {"name": "list_tactical_signals", "description": "Saved tactical signals with statistics and which one is active.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "tactical_overlay", "description": "Size today's leveraged / inverse ETF overlay that moves the household's total beta to the target (active signal or an explicit beta) without selling core stock, within the margin policy: ticket, candidate table, margin usage, carry, tax avoided vs selling core.",
     "input_schema": {"type": "object", "properties": {"target_beta": {"type": "number"}, "cash": {"type": "number"}}}},
    {"name": "tactical_backtest", "description": "Daily simulation of the core plus a signal-driven leveraged/inverse ETF overlay (leveraged-fund compounding, expense, margin interest, costs): equity vs core vs index, realised beta, drawdown, overlay losses booked.",
     "input_schema": {"type": "object", "properties": {"signal": {"type": "string"}, "start": {"type": "string"}, "long_instrument": {"type": "string"}, "inverse_instrument": {"type": "string"}}}},
    {"name": "import_holdings", "description": "Import a broker CSV/Excel holdings export (Schwab, Fidelity, IBKR, TradeStation, Vanguard, generic) into the current entity as tax lots. dry_run=true only previews the parsed rows and column mapping.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "account_name": {"type": "string"}, "account_type": {"type": "string"}, "dry_run": {"type": "boolean"}}, "required": ["path"]}},
]


def _js(o):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, float | np.floating) and not np.isfinite(o):
        return None
    return str(o)


def dumps(obj, **kw) -> str:
    return json.dumps(obj, default=_js, **kw)


def records(df: pd.DataFrame, limit: int | None = None, drop: tuple[str, ...] = ()) -> list[dict]:
    if df is None or df.empty:
        return []
    d = df.drop(columns=[c for c in drop if c in df.columns])
    if limit:
        d = d.head(limit)
    return json.loads(d.to_json(orient="records", date_format="iso", default_handler=str))


class ToolExecutor:
    """Executes tools; returns (result_text, change_id_or_None)."""

    def __init__(self, ctx: AppContext, conversation_id: int | None = None):
        self.ctx = ctx
        self.conversation_id = conversation_id
        from ..services.basket_service import BasketService
        from ..services.data_service import DataService
        from ..services.harvest_service import HarvestService
        from ..services.portfolio_service import PortfolioService
        from ..services.risk_service import RiskService
        self.data = DataService(ctx)
        self.portfolio = PortfolioService(ctx)
        self.risk = RiskService(ctx)
        self.harvest = HarvestService(ctx)
        self.baskets = BasketService(ctx)

    def execute(self, name: str, args: dict) -> tuple[str, int | None]:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            raise ValueError(f"unknown tool {name}")
        return fn(**args)

    @property
    def eid(self) -> int:
        eid = self.ctx.current_entity_id
        if eid is None:
            raise RuntimeError("no tax entity selected")
        return eid

    # ------------------------------------------------------------------ context
    def t_get_portfolio_context(self, include_lots: bool = False):
        ps = self.portfolio
        lots = ps.lots_view(self.eid)
        pos = ps.positions_view(lots)
        prof = self.ctx.tax.default_profile()
        out = {"as_of": date.today().isoformat(), "entity_id": self.eid,
               "accounts": [a.__dict__ for a in self.ctx.entities.accounts(self.eid)],
               "summary": ps.summary(lots), "positions": records(pos.round(4)),
               "wash_calendar": records(ps.wash_calendar(self.eid)),
               "tax_profile": {"fed_st": prof.fed_st_rate, "fed_lt": prof.fed_lt_rate, "state": prof.state_rate, "niit": prof.niit_rate,
                               "st_rate": prof.st_rate, "lt_rate": prof.lt_rate, "filing_status": prof.filing_status},
               "carryforward_prior_year": dict(zip(("st", "lt"), self.ctx.tax.carryforward(self.eid, date.today().year - 1), strict=True)),
               "realized_ytd": ps.realized(self.eid, date.today().year), "benchmark": self.risk.benchmark_name()}
        if include_lots:
            out["lots"] = records(lots.round(4), drop=("wash_explanation",))
        return dumps(out), None

    def t_get_risk_model_summary(self, model_id: int | None = None):
        if model_id is not None:
            model = self.risk.load(model_id)
            mid = model_id
        else:
            act = self.risk.active()
            if act is None:
                return dumps({"error": "no active risk model"}), None
            mid, model = act
        if model is None:
            return dumps({"error": f"model {model_id} not found"}), None
        out = {"model_version_id": mid, "as_of": model.as_of.isoformat(), "spec": asdict(model.spec), "diagnostics": model.diagnostics,
               "factor_vols": model.factor_vols().round(4).to_dict()}
        snap = self.data.latest_snapshot()
        if snap is not None:
            w = self._holding_weights(model, snap)
            if w is not None:
                bench = self.risk.benchmark_weights(snap, model)
                tab = self.risk.exposure_table(model, w, bench)
                out["exposures"] = records(tab.round(4).reset_index().rename(columns={"index": "factor"}))
                dec = model.te_decomposition(w.reindex(model.symbols).fillna(0.0), bench.reindex(model.symbols).fillna(0.0))
                out["tracking_error"] = dec.attrs["tracking_error"]
                out["te_decomposition"] = records(dec.round(6).reset_index().rename(columns={"index": "factor"}))
        return dumps(out), None

    def _holding_weights(self, model, snap) -> pd.Series | None:
        lots = self.portfolio.lots_view(self.eid, snap=snap)
        if lots.empty:
            return None
        w = lots.groupby("symbol")["market_value"].sum()
        w = w[w.index.isin(model.symbols)]
        return (w / w.sum()) if w.sum() > 0 else None

    def t_list_models(self):
        df = self.ctx.models.list()
        return dumps(records(df.drop(columns=["factor_list"]) if not df.empty else df)), None

    def t_get_run(self, run_id: int | None = None):
        if run_id is None:
            runs = self.ctx.runs.list(limit=1)
            if runs.empty:
                return dumps({"error": "no runs yet"}), None
            run_id = int(runs.iloc[0]["id"])
        run = self.harvest.load_run(run_id)
        if not run:
            return dumps({"error": f"run {run_id} not found"}), None
        return dumps({"id": run_id, "created_at": run["created_at"], "run_type": run["run_type"], "params": run["params"], "summary": run["summary"],
                      "trades": records(run["trades"], drop=("wash_explanation",)), "blocked": records(run["blocked"]),
                      "replacements": records(run["replacements"], limit=40)}), None

    def t_list_runs(self, limit: int = 20):
        df = self.ctx.runs.list(limit=limit)
        return dumps(records(df[["id", "created_at", "run_type", "as_of_date", "summary"]]) if not df.empty else []), None

    def t_search_securities(self, query: str | None = None, sector: str | None = None, industry: str | None = None,
                            min_mktcap_musd: float | None = None, limit: int = 40):
        snap = self.data.latest_snapshot()
        if snap is None:
            return dumps({"error": "no snapshot"}), None
        sec = snap.securities()
        fund = snap.fundamentals().set_index("symbol") if not snap.fundamentals().empty else pd.DataFrame()
        df = sec.copy()
        if query:
            q = query.lower()
            df = df[df["symbol"].str.lower().str.contains(q, regex=False) | df["name"].fillna("").str.lower().str.contains(q, regex=False)]
        if sector:
            df = df[df["gics_sector"].fillna("").str.lower().str.contains(sector.lower(), regex=False)]
        if industry:
            df = df[df["gics_industry"].fillna("").str.lower().str.contains(industry.lower(), regex=False) |
                    df["gics_sub_industry"].fillna("").str.lower().str.contains(industry.lower(), regex=False)]
        if not fund.empty and "mktcap" in fund:
            df["mktcap_musd"] = df["symbol"].map(fund["mktcap"])
            if min_mktcap_musd:
                df = df[df["mktcap_musd"] >= min_mktcap_musd]
        close = snap.close_matrix("close")
        px = snap.last_prices()
        sub = close[[s for s in df["symbol"] if s in close.columns]].iloc[-253:]
        ret1y = (sub.iloc[-1] / sub.iloc[0] - 1.0) if len(sub) > 20 else pd.Series(dtype=float)
        vol1y = sub.pct_change().std() * np.sqrt(252) if len(sub) > 20 else pd.Series(dtype=float)
        df["last_price"] = df["symbol"].map(px)
        df["ret_1y"] = df["symbol"].map(ret1y)
        df["vol_1y"] = df["symbol"].map(vol1y)
        cols = [c for c in ["symbol", "name", "subtype2", "gics_sector", "gics_industry", "mktcap_musd", "last_price", "ret_1y", "vol_1y"] if c in df]
        df = df[cols].sort_values("mktcap_musd", ascending=False, na_position="last") if "mktcap_musd" in df else df[cols]
        return dumps({"n_matches": int(len(df)), "rows": records(df.round(4), limit=limit)}), None

    def t_get_price_stats(self, symbols: list[str], lookback_days: int = 252):
        snap = self.data.latest_snapshot()
        if snap is None:
            return dumps({"error": "no snapshot"}), None
        syms = [s.upper() for s in symbols][:25]
        close = snap.close_matrix("close")
        have = [s for s in syms if s in close.columns]
        sub = close[have].iloc[-(lookback_days + 1):].dropna(how="all")
        r = sub.pct_change().iloc[1:]
        ann = (sub.iloc[-1] / sub.iloc[0]) ** (252 / max(len(r), 1)) - 1
        vol = r.std() * np.sqrt(252)
        dd = (sub / sub.cummax() - 1).min()
        stats = pd.DataFrame({"ann_return": ann, "ann_vol": vol, "max_drawdown": dd, "last": sub.iloc[-1]}).round(4)
        return dumps({"missing": [s for s in syms if s not in have], "period": [str(sub.index[0].date()), str(sub.index[-1].date())],
                      "stats": records(stats.reset_index().rename(columns={"index": "symbol"})),
                      "correlation": r.corr().round(3).to_dict()}), None

    def t_get_substitutes(self, symbol: str):
        sym = symbol.upper()
        sm = self.ctx.substitutes
        return dumps({"symbol": sym, "group": sm.group_key(sym), "group_members": sorted(sm.group_members(sym)),
                      "explanation": sm.explain_group(sym), "candidates": sm.candidates_for(sym)}), None

    def t_explain_wash_status(self, lot_id: int):
        lots = self.portfolio.lots_view(self.eid)
        row = lots[lots["lot_id"] == lot_id]
        if row.empty:
            return dumps({"error": f"lot {lot_id} not found or closed"}), None
        r = row.iloc[0].to_dict()
        r["group_explanation"] = self.ctx.substitutes.explain_group(r["symbol"])
        return dumps(r), None

    def t_screen_trade(self, side: str, symbol: str, quantity: float | None = None, lot_id: int | None = None,
                       account_id: int | None = None, trade_date: str | None = None):
        from ..tax.washsale import screen_proposed_buy, screen_proposed_sale
        d = date.fromisoformat(trade_date) if trade_date else date.today()
        book = self.portfolio.book(self.eid)
        sym = symbol.upper()
        aid = self.ctx.resolve_assetid(sym)
        if aid is None:
            return dumps({"error": f"unknown symbol {sym}"}), None
        if side.upper() == "BUY":
            scr = screen_proposed_buy(aid, sym, d, book.loss_sales(), book.groups)
            return dumps({"side": "BUY", "symbol": sym, "date": d.isoformat(), "status": scr.status, "explanation": scr.explanation}), None
        px = self.data.prices_for(self.data.latest_snapshot(), [sym]).get(sym) if self.data.latest_snapshot() else None
        lots = [lot for lot in book.open_lots(assetid=aid) if lot_id is None or lot.id == lot_id]
        if account_id is not None:
            lots = [lot for lot in lots if lot.account_id == account_id]
        if not lots:
            return dumps({"error": "no matching open lot"}), None
        lot = lots[0]
        qty = float(quantity or lot.quantity_open)
        loss = max(-(lot.unrealized_gain(px) if px else 0.0) * qty / lot.quantity_open, 0.0)
        det = screen_proposed_sale(aid, sym, lot.account_id, d, qty, loss, book.acquisitions(), book.groups, lot_id=lot.id)
        return dumps({"side": "SELL", "symbol": sym, "lot_id": lot.id, "quantity": qty, "price": px, "loss_at_price": loss,
                      "term": lot.term_at(d), "status": det.status, "disallowed_loss": det.disallowed_loss, "explanation": det.explanation}), None

    # ------------------------------------------------------------------ harvest & plans
    def _cfg(self, overrides: dict | None):
        cfg = self.harvest.load_config()
        ov = dict(overrides or {})
        bench = ov.pop("benchmark", None)
        if "priority" in ov:
            ov["priority"] = tuple(ov["priority"])
        if "priority_weights" in ov:
            ov["priority_weights"] = tuple(ov["priority_weights"])
        valid = {k: v for k, v in ov.items() if k in asdict(cfg)}
        return replace(cfg, **valid), bench

    def t_run_harvest(self, overrides: dict | None = None, notes: str | None = None):
        cfg, bench = self._cfg(overrides)
        rid, res = self.harvest.run(self.eid, cfg, notes=notes or "ai co-pilot", benchmark_name=bench)
        return dumps({"run_id": rid, "summary": res.summary, "trades": records(res.trades.round(4), drop=("wash_explanation", "holding_start")),
                      "blocked": records(res.blocked, drop=("wash_explanation",)), "n_replacement_candidates": int(len(res.replacements))}), None

    def t_run_frontier(self, te_grid: list[float] | None = None, overrides: dict | None = None):
        cfg, bench = self._cfg(overrides)
        df = self.harvest.frontier(self.eid, cfg, te_grid=te_grid, benchmark_name=bench)
        return dumps(records(df.round(5))), None

    def t_evaluate_trade_list(self, name: str, trades: list[dict], benchmark: str | None = None, rationale: str | None = None):
        rid, out = self.harvest.evaluate_trade_list(self.eid, name, trades, benchmark_name=benchmark, rationale=rationale,
                                                    source=f"ai:{self.conversation_id}")
        return dumps({"run_id": rid, **out}), None

    # ------------------------------------------------------------------ baskets
    def t_list_baskets(self):
        return dumps(records(self.ctx.baskets.list())), None

    def t_get_basket(self, name: str):
        b = self.ctx.baskets.get(name)
        if not b:
            return dumps({"error": f"basket '{name}' not found"}), None
        b["weights"] = b["weights"].round(5).to_dict()
        return dumps(b), None

    def t_create_basket(self, name: str, weights: dict, description: str | None = None, benchmark: str | None = None):
        res = self.baskets.create(name, pd.Series({k.upper(): float(v) for k, v in weights.items()}), description, source="ai", benchmark_name=benchmark)
        return dumps(res), None

    def t_optimize_basket(self, name: str, benchmark: str | None = None, universe: list[str] | None = None, n_max: int = 50,
                          max_weight: float = 0.08, sector_band: float = 0.02, tilts: dict | None = None, tilt_weight: float = 5.0,
                          exclude: list[str] | None = None, exclude_held: bool = False, exclude_wash_blocked: bool = True,
                          description: str | None = None):
        from ..optim.basket import BasketSpec
        spec = BasketSpec(n_max=n_max, max_weight=max_weight, sector_band=sector_band, tilts=tilts or {}, tilt_weight=tilt_weight,
                          exclude=[s.upper() for s in (exclude or [])], include_only=[s.upper() for s in universe] if universe else None)
        res = self.baskets.optimize(name, spec, benchmark_name=benchmark, description=description, source="ai",
                                    exclude_held=exclude_held, exclude_wash_blocked=exclude_wash_blocked, entity_id=self.eid)
        return dumps(res), None

    def t_analyze_basket(self, name: str, benchmark: str | None = None):
        return dumps(self.baskets.analyze(name, benchmark_name=benchmark)), None

    def t_set_benchmark(self, benchmark: str):
        snap = self.data.latest_snapshot()
        act = self.risk.active()
        if snap is None or act is None:
            return dumps({"error": "need a snapshot and an active model"}), None
        w = self.risk.benchmark_weights(snap, act[1], benchmark)     # validates
        self.ctx.set("benchmark_name", benchmark)
        self.ctx.db.audit("ai", "benchmark.set", benchmark)
        return dumps({"benchmark": benchmark, "n_names": int((w > 0).sum())}), None

    # ------------------------------------------------------------------ strategies & backtests
    def _strategy_service(self):
        from ..services.strategy_service import StrategyService
        return StrategyService(self.ctx)

    def t_list_strategies(self):
        return dumps(self._strategy_service().catalogue()), None

    def t_build_strategy_basket(self, name: str, strategy: str, params: dict | None = None, benchmark: str | None = None,
                                universe: list[str] | None = None, target_basket: str | None = None, cov_source: str = "model",
                                description: str | None = None, save: bool = True):
        from ..services.strategy_service import spec_from_params
        svc = self._strategy_service()
        spec = svc.resolve_target(spec_from_params(strategy, params), target_basket)
        out = svc.build(name, spec, benchmark_name=benchmark, universe=universe, entity_id=self.eid, description=description,
                        save=save, cov_source=cov_source, source="ai")
        return dumps(out), None

    def t_backtest_strategy(self, strategy: str, params: dict | None = None, name: str | None = None, start: str | None = None,
                            end: str | None = None, rebalance: str = "M", lookback_days: int = 252, cost_bps: float = 5.0,
                            benchmark_symbol: str | None = None):
        from ..optim.backtest import BacktestSpec
        from ..services.strategy_service import spec_from_params
        svc = self._strategy_service()
        spec = spec_from_params(strategy, params)
        bspec = BacktestSpec(start=start, end=end, rebalance=rebalance, lookback_days=int(lookback_days), cost_bps=float(cost_bps),
                             benchmark_symbol=benchmark_symbol.upper() if benchmark_symbol else None)
        rid, res = svc.backtest(spec, bspec, name=name or f"{strategy} backtest", entity_id=self.eid)
        yearly = (res.equity.resample("YE").last().pct_change().dropna()).round(4)
        byearly = (res.bench_equity.resample("YE").last().pct_change().dropna()).round(4)
        return dumps({"run_id": rid, "metrics": res.metrics, "warnings": res.warnings,
                      "calendar_year_returns": {str(k.year): {"strategy": float(v), "benchmark": float(byearly.get(k, float("nan")))} for k, v in yearly.items()},
                      "last_weights": res.weights.iloc[-1][res.weights.iloc[-1] > 0].sort_values(ascending=False).head(20).round(4).to_dict() if len(res.weights) else {}}), None

    def t_get_backtest(self, run_id: int):
        bt = self._strategy_service().load_backtest(run_id)
        if not bt:
            return dumps({"error": f"backtest run {run_id} not found"}), None
        W = bt["weights"]
        return dumps({"id": run_id, "params": bt["params"], "summary": bt["summary"],
                      "last_weights": W.iloc[-1][W.iloc[-1] > 0].sort_values(ascending=False).head(25).round(4).to_dict() if not W.empty else {}}), None

    # ------------------------------------------------------------------ pipelines
    def t_pipeline_schema(self):
        from ..optim.pipeline import EXAMPLES, NODE_TYPES, ORDER_HINT
        return dumps({"order": ORDER_HINT, "blocks": NODE_TYPES, "examples": {k: json.loads(v.to_json()) for k, v in EXAMPLES.items()}}), None

    def t_list_pipelines(self):
        return dumps(records(self.ctx.pipelines.list())), None

    def t_save_pipeline(self, pipeline: dict):
        from ..optim.pipeline import Pipeline, validate
        from ..services.pipeline_service import PipelineService
        p = Pipeline.from_json(pipeline)
        errs = validate(p)
        if errs:
            return dumps({"saved": False, "errors": errs}), None
        PipelineService(self.ctx).save(p, source="ai")
        return dumps({"saved": True, "name": p.name, "n_blocks": len(p.nodes)}), None

    def t_run_pipeline(self, name: str | None = None, pipeline: dict | None = None):
        from ..optim.pipeline import Pipeline
        from ..services.pipeline_service import PipelineService
        svc = PipelineService(self.ctx)
        if pipeline is not None:
            p = Pipeline.from_json(pipeline)
        elif name:
            p = svc.load(name)
            if p is None:
                return dumps({"error": f"pipeline '{name}' not found"}), None
        else:
            return dumps({"error": "give a saved pipeline name or an inline pipeline"}), None
        res = svc.run(p, entity_id=self.eid)
        return dumps(res.summary()), None

    # ------------------------------------------------------------------ concentration
    def _conc(self):
        from ..services.concentration_service import ConcentrationService
        return ConcentrationService(self.ctx)

    def t_concentration_overview(self):
        d = self._conc().overview(self.eid)
        return dumps({"stats": d["stats"], "positions": records(d["positions"].round(5), limit=60), "total_value": d["total_value"]}), None

    def t_diversification_plan(self, symbol: str, use_expected_losses: bool = True, **kw):
        from ..optim.glidepath import GlidePathSpec
        allowed = {k: v for k, v in kw.items() if k in GlidePathSpec.__dataclass_fields__ and v is not None}
        if "min_sold_by" in allowed:
            allowed["min_sold_by"] = {int(k): float(v) for k, v in dict(allowed["min_sold_by"]).items()}
        if "other_taxable_income" not in allowed:
            allowed["other_taxable_income"] = self._conc().other_income()
        spec = GlidePathSpec(**allowed)
        d = self._conc().plan(symbol, spec, self.eid, use_expected_losses=use_expected_losses)
        return dumps({"symbol": d["symbol"], "position": d["position"], "summary": d["summary"], "status": d["status"],
                      "schedule": records(d["schedule"].round(2)), "comparison": records(d["comparison"].round(2)), "spec": d["spec"]}), None

    def t_concentration_monte_carlo(self, symbol: str, horizon_years: int = 5, alpha_view: float = 0.0, p_stepup: float = 0.0, n_paths: int = 3000,
                                    market_return: float = 0.07, market_vol: float = 0.16):
        from ..optim.glidepath import GlidePathSpec, MonteCarloSpec
        svc = self._conc()
        gp = GlidePathSpec(horizon_years=int(horizon_years), alpha_view=alpha_view, p_stepup=p_stepup, other_taxable_income=svc.other_income())
        plan = svc.plan(symbol, gp, self.eid)
        out = svc.monte_carlo(symbol, gp, MonteCarloSpec(n_paths=int(n_paths), horizon_years=int(horizon_years), market_return=market_return, market_vol=market_vol),
                              self.eid, optimised=plan["schedule"]["sold"].values)
        return dumps({"symbol": symbol, "years": out["years"], "mu_stock": out["mu_stock"], "sigma_stock": out["sigma_stock"], "mu_market": out["mu_market"],
                      "policies": {n: {k: v for k, v in o.items() if k != "fan"} for n, o in out["policies"].items()},
                      "optimised_schedule": records(plan["schedule"][["period", "sold", "tax", "weight_after"]].round(2))}), None

    def t_hedge_analysis(self, symbol: str, tenor_years: float = 1.0, put_strike_pct: float = 0.90, call_strike_pct: float | None = None,
                         sigma: float | None = None, rate: float = 0.04, div_yield: float = 0.0):
        a = self._conc().hedge(symbol, tenor_years, put_strike_pct, call_strike_pct, sigma, rate, div_yield, self.eid)
        a["payoff"] = a["payoff"][::4]
        return dumps(a), None

    def t_concentration_alternatives(self, symbol: str, agi: float | None = None, p_stepup: float = 0.3, horizon_years: float = 10.0):
        return dumps(self._conc().alternatives(symbol, agi, p_stepup, horizon_years, self.eid)), None

    def t_gain_offset_plan(self, symbol: str, sell_value: float, replacement: str | None = None, offset_with_losses: bool = True, name: str | None = None):
        rid, out = self._conc().gain_offset_plan(symbol, float(sell_value), name, self.eid, offset_with_losses, replacement)
        return dumps({"run_id": rid, **out}), None

    def t_completion_portfolio(self, locked_symbols: list[str], n_max: int = 60, max_weight: float = 0.05, sector_band: float | None = 0.03, save_as: str | None = None):
        out = self._conc().completion(locked_symbols, int(n_max), float(max_weight), sector_band, save_as, self.eid)
        out.pop("full_weights", None)
        out["free_weights"] = dict(list(out["free_weights"].items())[:40])
        return dumps(out), None

    # ------------------------------------------------------------------ risk analytics
    def t_risk_decomposition(self, active: bool = True):
        d = self.risk.decomposition(self.eid, active=active)
        d["holdings"] = records(d["holdings"].reset_index().rename(columns={"index": "symbol"}).round(5), limit=40)
        return dumps(d), None

    def t_stress_test(self, shocks: dict | None = None, preset: str | None = None, propagate: bool = True, active: bool = False):
        from ..risk.analytics import PRESET_SHOCKS
        sh = dict(PRESET_SHOCKS.get(preset, {})) if preset else {}
        sh.update({k: float(v) for k, v in (shocks or {}).items()})
        if not sh:
            return dumps({"error": "give shocks or a preset", "presets": list(PRESET_SHOCKS)}), None
        d = self.risk.stress(sh, self.eid, active=active, propagate=propagate)
        d["holdings"] = records(d["holdings"].reset_index().rename(columns={"index": "symbol"}).round(5), limit=40)
        return dumps(d), None

    def t_historical_scenario(self, start: str, end: str, active: bool = False):
        return dumps(self.risk.scenario(start, end, self.eid, active=active)), None

    def t_parametric_var(self, horizon_days: int = 21, alpha: float = 0.99, active: bool = False):
        return dumps(self.risk.var(self.eid, int(horizon_days), float(alpha), active=active)), None

    def t_validate_risk_model(self, n_periods: int = 6, period_days: int = 21, model_overrides: dict | None = None):
        from ..risk.model import RiskModelSpec
        base = asdict(self.risk.default_spec())
        act = self.risk.active()
        if act is not None:
            base = asdict(act[1].spec)
        base.update({k: v for k, v in (model_overrides or {}).items() if k in base})
        df = self.risk.bias_test(RiskModelSpec(**base), n_periods=int(n_periods), period_days=int(period_days), entity_id=self.eid)
        return dumps({"summary": records(df.round(4)), "detail": records(df.attrs.get("detail", pd.DataFrame()).round(5), limit=100),
                      "how_to_read": "bias_stat ~1 is well-calibrated; >1 under-forecasts risk; <1 over-forecasts; band is the 95% range for n periods."}), None

    # ------------------------------------------------------------------ models
    def t_fit_risk_model(self, overrides: dict | None = None, name: str | None = None, activate: bool = False, notes: str | None = None,
                         preset: str | None = None):
        from ..risk.custom import load_all
        from ..risk.model import RISK_MODEL_PRESETS, RiskModelSpec, preset_spec
        load_all()
        if preset:
            if preset not in RISK_MODEL_PRESETS:
                return dumps({"error": f"unknown preset {preset!r}", "presets": list(RISK_MODEL_PRESETS)}), None
            spec = preset_spec(preset, **{k: v for k, v in (overrides or {}).items() if k in RiskModelSpec.__dataclass_fields__})
        else:
            base = asdict(self.risk.default_spec())
            base.update({k: v for k, v in (overrides or {}).items() if k in base})
            spec = RiskModelSpec(**base)
        snap = self.data.latest_snapshot()
        if snap is None:
            return dumps({"error": "no snapshot"}), None
        mid, model = self.risk.fit(snap, spec, name=name, notes=notes or "fitted by ai co-pilot", make_active=bool(activate))
        return dumps({"model_version_id": mid, "active": bool(activate), "as_of": model.as_of.isoformat(),
                      "diagnostics": model.diagnostics, "factor_vols": model.factor_vols().round(4).to_dict()}), None

    def t_compare_models(self, model_a: int, model_b: int):
        a, b = self.risk.load(model_a), self.risk.load(model_b)
        if a is None or b is None:
            return dumps({"error": "model not found"}), None
        out = {"vols": self.risk.compare_versions(model_a, model_b).round(4).reset_index().rename(columns={"index": "factor"}).to_dict("records"),
               "r2": {model_a: a.diagnostics.get("avg_r2"), model_b: b.diagnostics.get("avg_r2")},
               "median_specific_vol": {model_a: a.diagnostics.get("median_specific_vol"), model_b: b.diagnostics.get("median_specific_vol")}}
        snap = self.data.latest_snapshot()
        if snap is not None:
            te = {}
            for mid, m in ((model_a, a), (model_b, b)):
                w = self._holding_weights(m, snap)
                if w is not None:
                    bench = self.risk.benchmark_weights(snap, m)
                    te[mid] = m.tracking_error(w.reindex(m.symbols).fillna(0.0), bench.reindex(m.symbols).fillna(0.0))
            out["portfolio_te"] = te
        return dumps(out), None

    def t_list_style_factors(self):
        from ..risk.custom import load_all
        from ..risk.factors import STYLE_DEFINITIONS
        load_all()
        return dumps([{"name": k, "description": v.description, "needs_fundamentals": v.needs_fundamentals,
                       "module": v.fn.__module__} for k, v in STYLE_DEFINITIONS.items()]), None

    # ------------------------------------------------------------------ code
    def t_list_editable_modules(self):
        return dumps([{"path": m.path, "description": m.description, "tests": m.tests, "kind": m.kind} for m in AI_EDITABLE.values()]
                     + [{"new_files_allowed_under": NEW_FILE_PREFIXES, "canary_tests": CANARY_TESTS,
                         "note": "New style factors: create tlh/risk/custom/<name>.py registering into STYLE_DEFINITIONS; then fit_risk_model with styles including it."}]), None

    def t_read_module(self, path: str):
        p = path.replace("\\", "/")
        if not is_readable(p):
            raise PermissionError(f"{p} is not readable by the co-pilot")
        fp = REPO_ROOT / p
        if not fp.exists():
            raise FileNotFoundError(p)
        return fp.read_text(encoding="utf-8"), None

    def t_test_change(self, path: str, code: str, extra_tests: list[str] | None = None):
        from .registry import tests_for
        tests = list(dict.fromkeys(tests_for(path) + [t for t in (extra_tests or []) if t.startswith("tests/")]))
        res = sandbox.run_tests_with_change(path, code, tests=tests)
        self.ctx.db.audit("ai", "sandbox.test", path, passed=res.passed, duration=res.duration_s)
        return res.summary(), None

    def t_propose_change(self, path: str, code: str, title: str, rationale: str):
        from .copilot import propose
        cid = propose(self.ctx, path, code, title, rationale, self.conversation_id)
        ch = self.ctx.code.change(cid)
        return (f"Change #{cid} created for {path} (status={ch['status']}, sandbox_passed={bool(ch['sandbox_passed'])}). "
                f"Awaiting human approval in the AI co-pilot panel.\n\n{(ch['sandbox_stdout'] or '')[-3000:]}"), cid

    def t_run_analysis(self, code: str, timeout_s: int = 300):
        res = sandbox.run_analysis(code, timeout_s=int(timeout_s))
        self.ctx.db.audit("ai", "sandbox.analysis", None, passed=res.passed, duration=res.duration_s)
        return res.summary(max_chars=12000), None


    # ------------------------------------------------------------------ model library / calibration
    def t_risk_model_presets(self):
        from ..risk.model import preset_table
        return dumps(records(preset_table())), None

    def t_run_calibration_study(self, quick: bool = True, include_pca: bool = False, include_holdings: bool = True,
                                lookbacks: list[int] | None = None, horizons: list[int] | None = None, fit_recommendation: bool = False):
        out = self.risk.calibrate(quick=bool(quick), include_pca=bool(include_pca), include_holdings=bool(include_holdings), entity_id=self.eid,
                                  lookbacks=tuple(lookbacks) if lookbacks else None, horizons=tuple(horizons) if horizons else None)
        board = out["scoreboard"]
        cols = ["Horizon", "RankInHorizon", "Lookback", "Weighting", "Estimator", "Score", "BiasRatio", "Spearman", "CorrBias", "CorrRMSE", "CorrSpearman", "TEBiasRatio", "TESpearman", "Dates"]
        res = {"recommendation": out["recommendation"], "winners": records(out["winners"][cols].round(4)),
               "scoreboard_top": records(board.sort_values(["Horizon", "Score"]).groupby("Horizon").head(6)[cols].round(4)),
               "by_lookback": records(out["by_lookback"].round(4)), "by_weighting": records(out["by_weighting"].round(4)),
               "by_estimator": records(out["by_estimator"].round(4)), "grid": out["grid"],
               "caveats": "All specifications under-forecast on average (bias ratios > 1: fat tails); treat forecasts as a floor. Use the sample matrix for tight substitute pairs."}
        if fit_recommendation:
            spec = self.risk.spec_from_recommendation(out["recommendation"])
            snap = self.data.latest_snapshot()
            mid, m = self.risk.fit(snap, spec, notes="calibration study recommendation (YANG)", make_active=True)
            res["fitted_model_version_id"] = mid
            res["fitted_diagnostics"] = {k: v for k, v in m.diagnostics.items() if not isinstance(v, dict | list)}
        return dumps(res), None

    def t_pair_te_study(self, pairs: list[list[str]] | None = None, horizon_days: int = 63):
        df = self.risk.pair_study([tuple(p) for p in pairs] if pairs else None, horizon=int(horizon_days))
        if df.empty:
            return dumps({"error": "no pairs with enough history in the snapshot"}), None
        best = df.sort_values("abs_bias_dev").groupby("pair").head(1)
        return dumps({"least_biased_per_pair": records(best.round(4)), "all": records(df.round(4), limit=200)}), None

    # ------------------------------------------------------------------ sample baskets / long-short / overlay
    def t_build_sample_baskets(self, audience: str | None = None, names: list[str] | None = None, benchmark: str | None = None):
        df = self.baskets.build_library(names=names, audience=audience, benchmark_name=benchmark)
        return dumps({"built": records(df.round(4))}), None

    def t_longshort_analysis(self, extension: float = 0.30, years: int = 10, market_return: float = 0.06, market_vol: float = 0.16, n_paths: int = 150):
        from ..optim.longshort import LongShortSpec, simulate_loss_generation
        prof = self.ctx.tax.default_profile()
        res = simulate_loss_generation(LongShortSpec(extension=float(extension), years=int(years), market_return=float(market_return),
                                                     market_vol=float(market_vol), n_paths=int(n_paths), st_rate=prof.st_rate, lt_rate=prof.lt_rate))
        return dumps({"summary": res.summary, "financing": res.financing, "by_year": records(res.by_year.round(4)), "reference_10y_avg": res.reference,
                      "note": "Simulation, not a forecast; Quantinno reference rows are published marketing averages (Apr 2026)."}), None

    def t_exchange_glide(self, symbol: str | None = None, position_value: float | None = None, cost_basis: float | None = None,
                         extension: float = 0.30, years: int = 10):
        from ..optim.longshort import exchange_glide, years_to_diversify_table
        prof = self.ctx.tax.default_profile()
        if symbol:
            lots = self.portfolio.lots_view(self.eid, snap=self.data.latest_snapshot())
            sub = lots[lots["symbol"] == symbol.upper()]
            if sub.empty:
                return dumps({"error": f"{symbol} not held"}), None
            position_value, cost_basis = float(sub["market_value"].sum()), float(sub["cost_basis"].sum())
        if position_value is None or cost_basis is None:
            return dumps({"error": "give symbol or position_value + cost_basis"}), None
        df = exchange_glide(float(position_value), float(cost_basis), float(extension), int(years), lt_rate=prof.lt_rate, st_rate=prof.st_rate)
        return dumps({"schedule": records(df.round(2)), "years_to_full_divestiture": df.attrs.get("years_to_full_divestiture"),
                      "reference_years_table": years_to_diversify_table().reset_index().to_dict("records"),
                      "how_it_works": "each year sell the amount whose long-term tax equals the value of the extension's expected short-term losses"}), None

    def t_overlay_plan(self, target_beta: float = 1.0, contract: str = "MES", cash: float = 0.0, index_level: float | None = None,
                       portfolio_beta: float | None = None, days: int = 31):
        from ..optim.overlay import CONTRACTS, OverlayInputs, beta_from_model, plan_overlay
        snap = self.data.latest_snapshot()
        lots = self.portfolio.lots_view(self.eid, snap=snap)
        value = float(lots["market_value"].sum()) if not lots.empty else 0.0
        if portfolio_beta is None:
            act = self.risk.active()
            if act is None or lots.empty:
                return dumps({"error": "need holdings and an active risk model, or pass portfolio_beta"}), None
            w = lots.groupby("symbol")["market_value"].sum()
            portfolio_beta = beta_from_model(act[1], w)
        if index_level is None and snap is not None:
            px = snap.last_prices()
            proxy = CONTRACTS.get(contract, CONTRACTS["MES"])["proxy_etf"]
            if proxy in px.index:
                index_level = float(px[proxy]) * CONTRACTS[contract]["etf_ratio"]
        if index_level is None:
            return dumps({"error": "index_level needed (proxy ETF not in snapshot)"}), None
        prof = self.ctx.tax.default_profile()
        plan = plan_overlay(OverlayInputs(portfolio_value=value + float(cash), portfolio_beta=float(portfolio_beta), target_beta=float(target_beta), cash=float(cash),
                                          index_level=float(index_level), contract=contract, days=int(days), st_rate=prof.st_rate, lt_rate=prof.lt_rate))
        return dumps(plan.to_dict()), None

    # ------------------------------------------------------------------ taxes / onboarding
    def t_state_tax_rates(self, state: str | None = None, filing_status: str = "single", other_income: float = 300_000.0, gain: float = 0.0):
        from ..explain import explain_state
        from ..tax import state_rates as sr
        if state:
            c = sr.combined_marginal(state, filing_status, float(other_income), float(gain))
            return dumps({**c, "explanation": explain_state(c)}), None
        t = sr.table(float(other_income))
        return dumps({"year": sr.data_year(), "states": records(t.round(4)), "note": "approximate planning figures; verify before filing"}), None

    def t_set_tax_setup(self, state: str, filing_status: str, other_income: float):
        from ..services.home_service import HomeService
        out = HomeService(self.ctx).apply_tax_setup(state, filing_status, float(other_income))
        return dumps({"st_rate": out["st_rate"], "lt_rate": out["lt_rate"], "combined": out.get("combined")}), None

    def t_one_click_harvest(self):
        from ..services.home_service import HomeService
        res = HomeService(self.ctx).one_click(self.eid)
        return dumps({"run_id": res.run_id, "steps": res.steps, "summary": res.summary, "explanation": res.sentences,
                      "trades": records(res.trades, drop=("wash_explanation",))}), None

    # ------------------------------------------------------------------ tactical overlay / leverage
    def _tactical(self):
        from ..services.tactical_service import TacticalService
        return TacticalService(self.ctx)

    def t_leverage_instruments(self):
        from ..optim.overlay import custodian_capabilities
        svc = self._tactical()
        return dumps({"instruments": records(svc.instruments()), "margin_policy": asdict(svc.policy()), "custodians": records(custodian_capabilities()),
                      "note": "No futures, no direct shorts: inverse funds cut beta, leveraged funds raise it; leveraged funds decay with volatility."}), None

    def t_tactical_signal(self, name: str, kind: str, manual_beta: float = 1.0, path: str | None = None, beta_min: float = 0.0, beta_max: float = 1.5,
                          components: list[dict] | None = None, activate: bool = False, description: str = ""):
        from ..optim.tactical import SignalSpec
        svc = self._tactical()
        out = svc.save_signal(SignalSpec(name=name, kind=kind, manual_beta=float(manual_beta), path=path, beta_min=float(beta_min), beta_max=float(beta_max),
                                         components=components or [], description=description))
        if activate:
            svc.set_active(name)
        return dumps({**out, "active": svc.active_name()}), None

    def t_list_tactical_signals(self):
        svc = self._tactical()
        return dumps({"signals": records(svc.list_signals()), "active": svc.active_name(), "rules": svc.rules()}), None

    def t_tactical_overlay(self, target_beta: float | None = None, cash: float = 0.0):
        out = self._tactical().recommend(target_beta, float(cash), entity_id=self.eid)
        out["table"] = records(out["table"]) if isinstance(out.get("table"), pd.DataFrame) else []
        return dumps(out), None

    def t_tactical_backtest(self, signal: str | None = None, start: str | None = None, long_instrument: str = "SSO", inverse_instrument: str = "SDS"):
        res = self._tactical().backtest(signal, start, long_instrument, inverse_instrument, entity_id=self.eid)
        eq = res["equity"]
        yearly = eq.resample("YE").last().pct_change().dropna() if len(eq) else pd.Series(dtype=float)
        return dumps({"signal": res["signal"], "core_source": res["core_source"], "metrics": res["metrics"],
                      "yearly_returns": {str(k.year): round(float(v), 4) for k, v in yearly.items()},
                      "note": "simulation with leveraged-fund daily compounding, expense ratios, margin interest and 5 bps costs; not a forecast"}), None

    def t_import_holdings(self, path: str, account_name: str = "Imported brokerage", account_type: str = "taxable", dry_run: bool = False):
        from ..services.import_service import ImportService, plan_import
        plan = plan_import(path, default_account=account_name)
        out = {"rows": plan.n_rows, "mapping": plan.mapping, "warnings": plan.warnings, "preview": records(plan.frame, limit=50)}
        if not dry_run:
            out["result"] = ImportService(self.ctx).execute(self.eid, plan, account_type=account_type)
        return dumps(out), None


def is_editable_path(path: str) -> bool:
    return is_editable(path)
