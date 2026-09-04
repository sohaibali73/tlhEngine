# TLH Engine

Desktop tax-loss-harvesting engine for a single quant operator: lot-level tax accounting, a wash-sale compliance
engine with full test coverage, a Barra-style factor risk model, a constrained harvest optimizer with a user-adjustable
constraint hierarchy, an embedded co-pilot YANG (built on Claude) that can propose and sandbox-test changes to the model code, and
institutional Excel output. PySide6 GUI, Norgate data, SQLite + Parquet persistence.

**Nothing places orders.** Trade tickets are the terminal output.

## Quick start

New clone? Follow [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) (Python 3.12, Norgate Data Updater, `pip install -e ".[dev]"`, `.env`, first launch, health checks).

```
# 1. Norgate Data Updater must be running.
# 2. Put your key in .env (ANTHROPIC_API_KEY=...). Optional: TLH_AI_MODEL, TLH_AI_EFFORT, FRED_API_KEY.
python -m tlh --seed-demo          # seeds a demo household, pulls an S&P 500 snapshot, launches the GUI
python -m tlh                      # normal launch
python -m tlh --fit --harvest --no-gui   # headless: refresh snapshot, fit model, run harvest, print
python -m tlh --run-task "Daily harvest scan"   # run one agent task headless (Task Scheduler friendly)
python -m tlh --agent-loop                     # headless scheduler
python -m pytest                   # 231 tests; wash-sale/holding/ledger suites are the compliance gate
set PYTHONPATH=. && python scripts/smoke_tools.py     # exercise every co-pilot tool against live state (no API calls)
set PYTHONPATH=. && python scripts/smoke_gui.py       # offscreen GUI smoke
set PYTHONPATH=. && python scripts/smoke_copilot.py   # live two-turn co-pilot check (costs a few cents)
```

First launch: a splash appears in well under a second, the window follows in about 1.5 s, and the app lands on **Start here**
(advisor view: import holdings, set the client's state, find the tax savings). Data pulls (~640 symbols, 10 years, ~20 s)
and model fits run in the background. Use **View > Expert mode** (or `--expert`) for every quant workbench.

## Screens

| Screen | What it does |
|---|---|
| Start here | Three-click advisor flow: import a broker CSV/Excel (Schwab, Fidelity, IBKR, TradeStation, Vanguard, generic) or load the demo; pick state / filing status / income (every state's capital-gains rule built in); **Find my tax savings now** runs data refresh, model fit and the wash-safe harvest, then explains the result in plain English with the tax value, tickets, a savings gauge and an after-tax wealth projection. Simple/expert mode toggle. |
| TLH research | Due-diligence backtesting lab for the harvesting rules on the point-in-time S&P 500 (every member since 1999, delisted included): rolling 5- or 10-year windows starting each calendar year from 2000, monthly, whole shares, wash windows. Sweeps over account size ($10k to $1m), basket size (50 to 300), harvest trigger (0.01% to 1%), approach (pairs within sector, pairs within index, SARD twin baskets, TE optimizer with sector band and factor alignment) and concentrated starts (position size x embedded gain, tax-neutral unwind). Metrics: losses harvested per year, harvest life and half-life before ossification, realised and forecast tracking error, turnover, names held. Medians and IQR across windows, curves, heatmaps, resumable multi-core runs, markdown + Excel write-up. |
| Tactical overlay | Potomac signals (the five Potomac strategies read from their funds' NAVs via Yahoo Finance: flat NAV = risk-off, 80/5/5/5/5 target allocations, generated at the prior close and traded at the next close; also strategy CSV, manual beta, example rules, blends) set a target beta 0 to 1.5; the overlay reaches it with leveraged or inverse S&P ETFs on top of the core (no futures, no shorts, never selling stock) inside a Reg-T / house-maintenance margin policy; ticket, margin usage, market drop to a call, carry, tax avoided vs selling core; day-by-day backtest with leveraged-fund compounding. **Levered-beta model**: every S&P stock at index weight + 2x/3x ETFs + optional margin to a 1.5 beta at near-zero tracking error to 1.5x the index (model TE about 0.03%, realised leverage-layer TE about 0.3% a year), with drag, embedded financing and interest priced against each other. |
| Tax rates | All 50 states + DC on a choropleth and in a table: treatment (ordinary / none / exclusion / flat capital-gains / Washington excise), state ST/LT rates, combined federal + NIIT + state marginal rates at the client's income, plain-English explanation, one click to apply to the household. Approximate 2026 planning figures, flagged. |
| Portfolio | Lots with live valuation, ST/LT bucketing, days-to-long-term, wash-sale status + explanation per lot; positions treemap; harvest heatmap; wash-sale calendar (61-day windows); realised ledger with Schedule D netting and carryforward; transactions. Record buys/sells, scheduled DRIPs, CSV import. |
| Harvest | Drag-to-reorder constraint hierarchy (tax alpha / tracking error / factor neutrality) with weights; TE budget (soft or hard), sector drift, turnover, min trade, cost, tax horizon; opportunistic vs full-rebalance mode. Trade list with per-trade wash explanation, blocked lots with reasons, replacement candidates with correlations, before/after exposures, TE decomposition and sectors, TE-budget frontier, six-way priority comparison, run history. Mark trades acted-on or book them as executed (paper). |
| Risk model | Model library of 13 presets (ERM standard / short horizon / long horizon / robust / GARCH-dynamic / regime-conditional, hybrid ERM + statistical residual factors, Potomac calibrated covariances, tight-pair sample covariance, PCA, barra_lite with or without macro) with one-click **Fit whole library** comparison. Two fundamental estimators: barra_lite (six styles, GICS sectors, EWMA + shrinkage) and the full equity risk model ERM (multi-descriptor styles: size, non-linear size, beta, momentum, residual volatility, value, quality, growth, liquidity, leverage; GICS industry groups; Huber-robust option; EWMA vol × correlation with Newey-West; eigenfactor and volatility-regime adjustments; Bayesian-shrunk specific risk with a structural fallback). Spec editor, fit, versioned model list with activate/compare; portfolio vs benchmark exposure bars, style radar, sector weights, TE decomposition, cumulative factor returns, holdings exposures, diagnostics. |
| Risk lab | Where risk comes from and whether the model can be trusted: total/active risk decomposition by factor group, factor and holding (marginal contributions); factor stress tests in sigma or raw units with correlated propagation and presets (market -2σ, momentum crash, flight to quality, rates +100bp, …); historical factor-return replay; parametric VaR/ES; estimator diagnostics (t-stats, R² over time, factor and style correlation heatmaps, descriptor coverage); out-of-sample bias tests; **Calibration** tab re-runs the 2026 calibration study (lookback x weighting x estimator x horizon, walk-forward, seven metrics, composite score) on the live snapshot with an optional PCA arm and your holdings as a test basket, applies the winner in one click, and a substitute-pair TE study shows when to trust the sample matrix over shrinkage. |
| Concentration | Embedded gains and position concentration workbench: per-position tax-if-liquidated (2026 brackets stacked on other income, NIIT, state), risk share and effective N; multi-year tax-aware glide-path optimiser (convex; loss offsets and carryforwards, ST→LT timing, gain budgets, step-up probability, alpha view, risk aversion) with policy comparison; Monte Carlo of after-tax wealth for hold / sell now / instalments / optimised; Black-Scholes collars with zero-cost solver and constructive-sale / straddle flags; charitable vs sell-then-donate, gifting, Section 721 exchange fund, step-up option value; one-click gain-offset trade plan paired with wash-safe losses; lock-in ratio (tax per 1% TE removed) and completion portfolios around locked names. |
| Model portfolios | Build min-TE baskets (name cap, weight cap, sector band, style tilts, exclude held / wash-blocked names), inspect members and exposures vs benchmark, set a basket as the benchmark so Harvest migrates toward it in full-rebalance mode. **Sample model portfolios**: 17 one-click recipes (index trackers, integrated multi-factor, defensive equity, quality-momentum, min-variance, max-diversification, risk parity, HRP, value / low-vol / growth tilts, min-CVaR, Black-Litterman, 130/30 and 145/45 long/short tax engines). |
| TLH model builder | Drag-and-drop pipeline canvas: Universe → Filter → Rank → Benchmark → Construction → Tax-aware transition → Harvest → Save/export. Left-to-right order is execution order; edit parameters in the property panel; run end to end (saves a basket, runs the wash-safe harvest toward it, optional Excel). Save/load, import/export JSON, three examples, or *Ask YANG to design…* from a sentence. |
| Strategy lab | Eighteen construction methods (incl. `levered_beta`) on the risk model's covariance: integrated multi-factor (vs mixed sleeves), defensive equity with a covariance-implied beta cap, quality-momentum, 130/30-style long/short extension (beta-neutral, sector/style-neutral extension, disjoint long/short tranches), market-neutral overlay around existing holdings, plus equal/cap weight, minimum variance, maximum diversification, risk parity, hierarchical risk parity, mean-variance with signal alphas, Black-Litterman with views, minimum CVaR, stratified direct-index sampling, factor tilts, and tax-aware transition (move toward a target under a net realised-gain budget). Save any result as a basket; walk-forward backtests with costs, point-in-time membership, equity/drawdown/weights charts and full metrics vs benchmark. |
| YANG | Streamed markdown chat with live reasoning, tool cards, stop button and cost meter. 63 tools: onboarding (import_holdings, set_tax_setup, state_tax_rates, one_click_harvest), model library / calibration / pair studies, sample baskets, long/short economics (loss generation vs long-only, financing, tax-neutral Exchange glide), futures overlay sizing with section 1256 / 1092 flags, plus portfolio/model/run context, security search and price stats, wash screens, run_harvest / run_frontier with config overrides, evaluate_trade_list (hand-designed plans, every trade wash-screened, saved as reviewable runs), optimize/create/analyze baskets, build_strategy_basket / backtest_strategy for any construction method, set_benchmark, fit_risk_model / compare_models, custom style-factor modules, sandboxed code changes with approve/reject/rollback, audit log. |
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
  tax/               holding period, lots + selection methods, wash-sale engine, ledger + Schedule D netting, rates, state_rates.yaml/.py (51 jurisdictions), concentration.py (brackets, collars, alternatives)
  risk/              barra_lite factors + fit, ERM descriptors/estimator (descriptors.py, erm.py), statistical.py (calibrated covariance, PCA, hybrid, GARCH, regime), calibration.py (walk-forward study), model.py (library of presets), analytics.py, benchmark weights, custom/ plugins
  research/          due-diligence backtesting lab (deep store, monthly lot-level simulator, sweeps on every core, write-up)
  optim/             cvxpy harvest optimizer, frontier + priority sweeps, basket construction, strategies (18 methods incl. long/short and levered beta), basket_library (sample recipes), longshort.py (L/S economics), leverage.py (leveraged ETFs, margin policy, tactical overlay, simulator), tactical.py (signals), overlay.py (futures reference), glidepath, backtester
  ai/                registry of AI-editable modules, local sandbox, tool schemas + executor, Claude co-pilot + promotion, schedule grammar
  export/            xlsxwriter workbook
  services/          AppContext + data / portfolio / risk / harvest / basket / strategy / agent / home (one-click) / import services + demo seed (no Qt)
  gui/               PySide6 app (lazy tabs, splash, simple/expert mode), theme, table model, Plotly-in-Qt charts (Plotly.react updates), workers, quick.py (pop-up, hotkey, tray), import_dialog, screens/
tests/               220+ tests (tax rules are hand-verified against IRS Pub 550 examples)
var/                 runtime state (git-ignored): tlh.sqlite, cache/snapshots, models, runs, exports, sandbox, logs
```

See DECISIONS.md for every stack decision and the tax conventions, docs/ARCHITECTURE.md for data flow, and
[docs/WHITEPAPER.md](docs/WHITEPAPER.md) for the full write-up of the algorithms, validation and the AI-assisted operating model.

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
