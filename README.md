# TLH Engine

Desktop tax-loss-harvesting engine for a single quant operator: lot-level tax accounting, a wash-sale compliance
engine with full test coverage, a Barra-style factor risk model, a constrained harvest optimizer with a user-adjustable
constraint hierarchy, an embedded co-pilot YANG (built on Claude) that can propose and sandbox-test changes to the model code, and
institutional Excel output. PySide6 GUI, Norgate data, SQLite + Parquet persistence.

**Nothing places orders.** Trade tickets are the terminal output.

## Quick start

```
# 1. Norgate Data Updater must be running.
# 2. Put your key in .env (ANTHROPIC_API_KEY=...). Optional: TLH_AI_MODEL, TLH_AI_EFFORT, FRED_API_KEY.
python -m tlh --seed-demo          # seeds a demo household, pulls an S&P 500 snapshot, launches the GUI
python -m tlh                      # normal launch
python -m tlh --fit --harvest --no-gui   # headless: refresh snapshot, fit model, run harvest, print
python -m tlh --run-task "Daily harvest scan"   # run one agent task headless (Task Scheduler friendly)
python -m tlh --agent-loop                     # headless scheduler
python -m pytest                   # 150 tests; wash-sale/holding/ledger suites are the compliance gate
set PYTHONPATH=. && python scripts/smoke_tools.py     # exercise every co-pilot tool against live state (no API calls)
set PYTHONPATH=. && python scripts/smoke_gui.py       # offscreen GUI smoke
set PYTHONPATH=. && python scripts/smoke_copilot.py   # live two-turn co-pilot check (costs a few cents)
```

First launch: the app pulls a data snapshot (roughly 640 symbols, 10 years, about 20 seconds), fits the risk model
(about 3 seconds) and lands on the Portfolio screen.

## Screens

| Screen | What it does |
|---|---|
| Portfolio | Lots with live valuation, ST/LT bucketing, days-to-long-term, wash-sale status + explanation per lot; positions treemap; harvest heatmap; wash-sale calendar (61-day windows); realised ledger with Schedule D netting and carryforward; transactions. Record buys/sells, scheduled DRIPs, CSV import. |
| Harvest | Drag-to-reorder constraint hierarchy (tax alpha / tracking error / factor neutrality) with weights; TE budget (soft or hard), sector drift, turnover, min trade, cost, tax horizon; opportunistic vs full-rebalance mode. Trade list with per-trade wash explanation, blocked lots with reasons, replacement candidates with correlations, before/after exposures, TE decomposition and sectors, TE-budget frontier, six-way priority comparison, run history. Mark trades acted-on or book them as executed (paper). |
| Risk model | Two estimators: barra_lite (six styles, GICS sectors, EWMA + shrinkage) and the full equity risk model ERM (multi-descriptor styles: size, non-linear size, beta, momentum, residual volatility, value, quality, growth, liquidity, leverage; GICS industry groups; Huber-robust option; EWMA vol × correlation with Newey-West; eigenfactor and volatility-regime adjustments; Bayesian-shrunk specific risk with a structural fallback). Spec editor, fit, versioned model list with activate/compare; portfolio vs benchmark exposure bars, style radar, sector weights, TE decomposition, cumulative factor returns, holdings exposures, diagnostics. |
| Risk lab | Where risk comes from and whether the model can be trusted: total/active risk decomposition by factor group, factor and holding (marginal contributions); factor stress tests in sigma or raw units with correlated propagation and presets (market -2σ, momentum crash, flight to quality, rates +100bp, …); historical factor-return replay; parametric VaR/ES; estimator diagnostics (t-stats, R² over time, factor and style correlation heatmaps, descriptor coverage); out-of-sample bias tests. |
| Concentration | Embedded gains and position concentration workbench: per-position tax-if-liquidated (2026 brackets stacked on other income, NIIT, state), risk share and effective N; multi-year tax-aware glide-path optimiser (convex; loss offsets and carryforwards, ST→LT timing, gain budgets, step-up probability, alpha view, risk aversion) with policy comparison; Monte Carlo of after-tax wealth for hold / sell now / instalments / optimised; Black-Scholes collars with zero-cost solver and constructive-sale / straddle flags; charitable vs sell-then-donate, gifting, Section 721 exchange fund, step-up option value; one-click gain-offset trade plan paired with wash-safe losses; lock-in ratio (tax per 1% TE removed) and completion portfolios around locked names. |
| Model portfolios | Build min-TE baskets (name cap, weight cap, sector band, style tilts, exclude held / wash-blocked names), inspect members and exposures vs benchmark, set a basket as the benchmark so Harvest migrates toward it in full-rebalance mode. |
| TLH model builder | Drag-and-drop pipeline canvas: Universe → Filter → Rank → Benchmark → Construction → Tax-aware transition → Harvest → Save/export. Left-to-right order is execution order; edit parameters in the property panel; run end to end (saves a basket, runs the wash-safe harvest toward it, optional Excel). Save/load, import/export JSON, three examples, or *Ask YANG to design…* from a sentence. |
| Strategy lab | Twelve construction methods on the risk model's covariance: equal/cap weight, minimum variance, maximum diversification, risk parity, hierarchical risk parity, mean-variance with signal alphas, Black-Litterman with views, minimum CVaR, stratified direct-index sampling, factor tilts, and tax-aware transition (move toward a target under a net realised-gain budget). Save any result as a basket; walk-forward backtests with costs, point-in-time membership, equity/drawdown/weights charts and full metrics vs benchmark. |
| YANG | Streamed markdown chat with live reasoning, tool cards, stop button and cost meter. 31 tools: portfolio/model/run context, security search and price stats, wash screens, run_harvest / run_frontier with config overrides, evaluate_trade_list (hand-designed plans, every trade wash-screened, saved as reviewable runs), optimize/create/analyze baskets, build_strategy_basket / backtest_strategy for any construction method, set_benchmark, fit_risk_model / compare_models, custom style-factor modules, sandboxed code changes with approve/reject/rollback, audit log. |
| YANG Agent | Claude works unattended. Scheduled tasks (daily harvest scan, wash-window watch, weekly model health, month-end transition check, strategy leaderboard, or your own) run in the background on a 30-second scheduler, file markdown reports, notify via the tray, and queue any code changes for approval. **Ask Claude** pop-up: Ctrl+Space in the app or the global hotkey Ctrl+Alt+C from anywhere on the desktop; type a job, watch it stream, open the conversation. Headless: `python -m tlh --run-task "Daily harvest scan"` for Windows Task Scheduler, `--agent-loop` to run the scheduler without the GUI. |
| Export | Formatted workbook per run (summary, trade ticket, trades, wash explanations, blocked lots, replacements, exposures, TE, sectors, positions, optional frontier/priority sheets). |
| How-to (Help › Interactive how-to, F1) | Dockable 12-step guided tour with completion tracking, *Show me* (jumps and highlights the control), *Do it* (performs the step), and *Ask YANG* per step. Opens automatically on first launch. |
| Settings | Entities and accounts (wash scope), tax rates and carryforwards, presumed-identical ETF toggle, universe/benchmark, snapshots, demo seed/reset, AI config display. |

## Layout

```
tlh/
  config.py          settings from .env
  db/                schema.sql, sqlite connection, repositories (+ LotBook hydration)
  data/              norgate wrapper, Parquet/DuckDB snapshots, macro series, substitutes.yaml + loader
  tax/               holding period, lots + selection methods, wash-sale engine, ledger + Schedule D netting, rates, concentration.py (brackets, collars, alternatives)
  risk/              barra_lite factors + fit, ERM descriptors/estimator (descriptors.py, erm.py), analytics.py (decomposition, stress, VaR, bias tests), benchmark weights, custom/ plugins
  optim/             cvxpy harvest optimizer, frontier + priority sweeps, basket construction, strategies (12 methods), walk-forward backtester
  ai/                registry of AI-editable modules, local sandbox, tool schemas + executor, Claude co-pilot + promotion, schedule grammar
  export/            xlsxwriter workbook
  services/          AppContext + data / portfolio / risk / harvest / basket / strategy / agent services + demo seed (no Qt)
  gui/               PySide6 app, theme, table model, Plotly-in-Qt charts, workers, quick.py (pop-up, hotkey, tray), screens/
tests/               150 tests (tax rules are hand-verified against IRS Pub 550 examples)
var/                 runtime state (git-ignored): tlh.sqlite, cache/snapshots, models, runs, exports, sandbox, logs
```

See DECISIONS.md for every stack decision and the tax conventions, docs/ARCHITECTURE.md for data flow.

## Data

Norgate (prices, GICS, fundamentals, delisted history, economic series) is the only required source. Every run and
model fit records the snapshot id it consumed; snapshots are immutable Parquet folders, so "what did the model see on
date X" is answerable offline. Norgate fundamentals are current-values-only; see DECISIONS.md D3 for what that
implies for the fitted style factors.

## YANG (AI co-pilot)

Runs against the Anthropic Messages API (`claude-opus-5` by default, adaptive thinking, effort `high`). Its 31 tools read
the portfolio, model and runs; build and analyse model portfolios; run the harvest optimizer and frontier with config
overrides; evaluate hand-designed trade plans (saved as reviewable `ai_plan` runs); fit and compare risk-model variants;
invent style factors as plugin modules under `tlh/risk/custom/`; read/test/propose changes to the modules listed in
`tlh/ai/registry.py`; and run analysis scripts against a copy of the state. Every proposed change is executed in a local subprocess sandbox with the gating
tests (plus the wash-sale suite as a canary) before you see it; promotion writes a new code version, keeps the old one,
and hot-reloads. Nothing the co-pilot does can place a trade or bypass the wash-sale engine.

## API key

Settings › *YANG (AI co-pilot)* lets you paste the Anthropic key, pick the model and effort, test the key, and save. The
key is written to `.env` next to the app (git-ignored) and takes effect immediately; it is never stored in the database.

## Portable EXE (one click)

```
Build-EXE.bat                 # or: python setup.py build_exe --clean
```
Produces `dist/TLHEngine/TLHEngine.exe` (PyInstaller one-folder build; installs PyInstaller if missing). Copy the whole
`dist/TLHEngine` folder to any Windows machine that has Norgate Data Updater. On first run the app creates `var/` and
`.env` next to the exe; enter the API key in Settings. The YANG code-change sandbox needs the source checkout and is
disabled in the EXE (everything else works).

## GitHub

`.env`, `var/`, `build/`, `dist/` are git-ignored. To publish:
```
git remote add origin https://github.com/<you>/tlhEngine.git
git push -u origin main
```

## Disclaimers

Decision support for a professional operator. Not tax or investment advice. Wash-sale determinations follow the
conventions in DECISIONS.md D8; confirm edge cases with a tax advisor.
