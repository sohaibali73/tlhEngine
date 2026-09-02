# Architecture & Tech-Stack Decisions

Decisions the brief marked "decide and document". Each records the choice, the alternatives weighed, and why.
Change a decision here first, then in code.

## D1. GUI framework: PySide6 (Qt 6) with QtWebEngine for Plotly charts
- Alternatives: (a) FastAPI + Plotly Dash / React in a browser; (b) Tkinter; (c) PySide6.
- Chosen: PySide6. The brief rules out web-based GUIs; the operator already uses Qt/Tk for other quant
  tools; QTableView + QSortFilterProxyModel gives genuinely dense, sortable, filterable numeric tables;
  QWebEngineView renders Plotly HTML with full hover/zoom/click interactivity, so nothing is lost on charts
  versus a browser app; a chat panel is trivial in Qt. Long computations run on QThreadPool workers so the
  UI never blocks.
- Cost: click-to-drill from Plotly back into Qt needs a small QWebChannel bridge. Accepted.

## D2. Persistence: SQLite (state) + Parquet/DuckDB (market-data snapshots)
- State of record (entities, accounts, lots, transactions, runs, model versions, AI audit) lives in one
  SQLite file (var/tlh.sqlite, WAL mode). Transactional, single-file, zero-ops, trivially backed up.
- Market-data snapshots (prices, fundamentals, classifications for a universe on a pull date) are written
  as Parquet under var/cache/snapshots/<snapshot_id>/ and queried through DuckDB. Columnar and fast for
  500+ symbols x 10y, and the snapshot folder is the reproducibility unit: every run records the snapshot
  id it saw, so "what did the model see on date X" is answerable without touching Norgate.
- Rejected: DuckDB as sole store (state tables are row-oriented and need concurrent GUI writers);
  SQLAlchemy ORM (installed, but plain sqlite3 with an explicit schema.sql is easier to audit for a
  compliance-critical system).

## D3. Data sources
- Prices / reference / classification / fundamentals: norgatedata. Norgate Data Updater must be running;
  the data layer checks norgatedata.status() at startup and the GUI surfaces a hard error banner.
  GICS sector/industry from classification_at_level('GICS', ...). Fundamentals used: mktcap, qbvps,
  peexclxor, ttmpr2rev, ttmnpmgn, ttmgrosmgn, qtotd2eq, ttmepschg, revchngyr, ttmfcf, ttmniac,
  sharesoutstanding, beta.
- Limitation, documented: Norgate fundamentals are point-in-time current values (one value + as-of date),
  not a history. Fundamental style exposures (value, quality, growth) use the current fundamental scaled by
  historical price where that is meaningful (B/P, E/P, S/P) and are held constant otherwise. Price-derived
  factors (momentum, low-vol, size via shares x price) are fully historical. This introduces mild
  look-ahead in the fit of fundamental factor returns; it does not affect today's exposures or the
  covariance used for today's trade. A paid point-in-time fundamentals vendor would remove it; not worth
  the cost for a decision-support tool at this stage.
- Macro overlay: Norgate's Economic database carries the FRED-equivalent series locally (%TNX 10y yield,
  %IRX 3m bill, #US10Y-2Y slope, %COBAA minus %COAAA credit spread, $USDX dollar index). Used by default
  so no extra key is needed. fredapi is wired as an optional alternative when FRED_API_KEY is set. Macro
  factors are toggleable and default OFF.
- Substantially-identical / substitute mapping: in-house YAML (tlh/data/substitutes.yaml) with three
  relationship tiers: identical (same assetid; share classes of one issuer), presumed_identical
  (different-issuer ETFs on the same index; treated as identical by default, configurable), and
  substitute (correlated, different index or exposure; wash-safe replacement). Norgate subtype2/3 seeds the
  ETF-vs-stock logic. The YAML is one of the AI-editable artifacts.

## D4. Optimizer: cvxpy with CLARABEL (fallback OSQP / SCS), two-pass minimum-trade-size heuristic
- The core problem is a convex QP: continuous lot-sell fractions and replacement-buy dollars; quadratic
  tracking-error and factor-drift penalties; linear tax objective and budgets.
- Minimum trade size is semi-continuous (needs MIQP). No free MIQP solver ships with cvxpy here, so:
  solve the QP, zero trades below the minimum, fix them at zero, re-solve. Documented as a heuristic.
- Constraint hierarchy: the user orders {tax alpha, tracking error, factor neutrality}; the ordering maps to
  geometrically decaying objective weights plus optional hard caps. A frontier sweep varies the TE weight
  and records tax alpha vs TE for the chart.

## D5. Risk model: Barra-style cross-sectional regression
- Exposures: market (1); styles value, momentum, quality, size, low-vol, growth (cap-weighted z-scores,
  winsorised at +/-3); GICS sector dummies; optional macro block (time-series betas to macro shocks).
- Factor returns by weighted least squares (sqrt-cap weights) on daily returns over the lookback (default
  504 trading days). Factor covariance: EWMA (half-life 126 days) with Ledoit-Wolf-style shrinkage.
  Specific risk: EWMA residual variance shrunk toward the cross-sectional median.
- Assets without fundamentals or GICS (ETFs, funds) get exposures by ridge time-series regression of their
  returns on the estimated factor returns. This is how benchmark ETFs and replacement ETFs enter the same
  covariance as single stocks.
- Every fit persists as a versioned artifact (Parquet + JSON diagnostics) with a model_versions row.

## D6. Embedded AI: Anthropic Messages API, manual tool loop, local sandbox
- Model: claude-opus-5 default with adaptive thinking (display summarized), effort high, streaming.
  Configurable via .env (TLH_AI_MODEL, TLH_AI_EFFORT) and the Settings screen. claude-sonnet-5 is the
  cheaper option, claude-fable-5-1 the most capable. Authoring risk-model code warrants Opus-tier
  reasoning; the operator can downgrade explicitly.
- Loop: manual tool loop (not the beta tool runner) so the GUI can stream text, gate propose_code_change
  behind human approval, and persist every turn to SQLite.
- Code execution: local subprocess sandbox, not Anthropic's server-side code-execution tool. The code being
  authored must run against local Norgate data, the local SQLite state, and the live tlh package; none of
  that exists in a remote container. The sandbox copies the package to a temp dir, applies the proposed
  file, runs pytest (targeted tests plus the full wash-sale suite as a canary) and an optional smoke script,
  with a wall-clock timeout and captured stdout/stderr. This is the draft -> sandbox -> diff + tests ->
  approve -> promote loop.
- Promotion: approved code is written to the real module, the prior text kept in code_versions, the module
  hot-reloaded, and the change row moved to promoted. Rollback restores any prior version.
- Context: the co-pilot receives compact JSON of positions, lots, active model diagnostics, the last run
  summary, and tool-based read access to the AI-editable modules. The stable system prompt is cached with
  cache_control; the context block is trimmed to roughly 30k tokens.
- Credentials: .env at the repo root (git-ignored) holding ANTHROPIC_API_KEY; the process environment
  overrides it. The key is never written to the database or the audit log.

## D7. Excel: xlsxwriter
- Write-only, fastest, richest formatting API (conditional formats, freeze panes, number formats).
  openpyxl is reserved for reading user-provided lot imports.

## D8. Tax-rule conventions (highest-scrutiny code; see tests/test_holding.py, tests/test_washsale.py)
- Long-term iff sale date > acquisition date + 1 year (holding period starts the day after acquisition; a
  sale on the anniversary is still short-term). Feb 29 acquisitions treat Feb 28 of the next year as the
  anniversary, so Mar 1 is the first long-term day.
- Wash-sale window: 30 calendar days before and after the loss sale (61 days inclusive), per
  substantially-identical group, across every account in the same tax entity, including IRAs and Roths.
- Replacement shares are matched to loss shares in chronological order of acquisition; the disallowed loss
  is proportional when replacement quantity < loss quantity; the disallowed amount is added to the
  replacement lot's basis and the sold lot's holding period tacks on; a replacement inside an IRA disallows
  the loss permanently (Rev. Rul. 2008-5) with no basis step-up.
- Carryforward: net ST and LT separately; unused net loss carries forward retaining character after the
  ordinary-income offset ($3,000, or $1,500 married filing separately), configurable in the tax profile.
- Not legal advice. The engine flags and explains; the operator and their tax advisor decide.

## D9. Universe and benchmark
- Default fit universe: S&P 500 current constituents (Norgate watchlist) plus every held symbol plus every
  symbol in substitutes.yaml. Default benchmark: S&P 500 cap-weighted from Norgate shares outstanding x
  price, with an ETF ticker (SPY/IVV/VOO) selectable as an alternative.

## D10. Model portfolios, AI trade plans and co-pilot tool surface (added 2026-09-02)
- Baskets are first-class state (`baskets`, `basket_members`). `optim/basket.py` builds them as a min-TE QP with a
  name cap enforced by iterative pruning; when a tight cap makes the sector band infeasible the band is relaxed
  progressively and finally the last feasible solution is truncated (status says so). A basket can be the benchmark
  (`benchmark_name = "basket:<name>"`), which is how "rebalance toward a model portfolio while harvesting" works:
  full-rebalance mode may buy any wash-safe benchmark constituent.
- AI-designed trade plans go through `HarvestService.evaluate_trade_list`: every sell is lot-matched (HIFO unless a
  lot id is given) and wash-screened including against the plan's own buys; every buy is screened against realised
  and planned loss sales; TE/exposures before/after are computed; the plan is saved as an `ai_plan` run the user can
  load, export and book. Nothing is executed by the AI.
- Conversation persistence stores only replayable content: SDK-only fields (`parsed_output`, `caller`) and nulls are
  stripped, and `_repair_history` drops dangling tool_use turns (e.g. after a user cancel) so a resumed conversation
  never 400s.
- New style factors are plugin modules under `tlh/risk/custom/` registering into `STYLE_DEFINITIONS`; the folder is
  auto-imported before every fit and after every promotion. Model fits triggered by the co-pilot are never activated
  unless asked (`activate=true`); the user activates from the Risk model screen.
- Cost visibility: each turn reports tokens and an estimated cost from a per-model price table in `ai/copilot.py`.

## D11. Construction strategies and backtesting (added 2026-09-02)
- `optim/strategies.py` exposes one contract (`run_strategy(spec, inputs)`) over twelve methods. Covariance is an input
  (factor-model covariance by default, Ledoit-Wolf shrunk sample optionally), so every method is comparable and the
  co-pilot can add methods without touching the risk model. Name caps use the same iterative pruning as baskets;
  sector bands relax gracefully with the status reporting it.
- Risk parity uses the convex log-barrier form (min 0.5 w'Sw - (1/n) sum log w), names pre-selected by minimum variance
  when a cap applies. HRP follows Lopez de Prado (single linkage on correlation distance, recursive bisection with
  inverse-variance allocation). Mean-variance alphas follow Grinold (IC x vol x standardized signal) and are
  benchmark-relative by default. Black-Litterman uses the equilibrium prior from benchmark weights with delta implied
  by a market Sharpe of 0.4 and Omega from view confidence. Min-CVaR is the Rockafellar-Uryasev LP on trailing daily
  scenarios. Stratified indexing samples sector x size strata proportionally to benchmark weight and min-TE reweights.
- Tax-aware transition uses explicit sell/buy variables so the net realised gain (gain_frac . sell) is linear, with
  losses offsetting gains inside the budget; names cannot be both sold and bought in one rebalance (iterative fix of
  buy = 0 on churned names) because that would be a wash sale, not a harvest.
- The backtester is walk-forward with trailing shrunk covariance and price-based signals computed at each rebalance,
  weights drifting between rebalances and one-way turnover charged in bps. It reports its own caveats: survivorship
  (mitigated by point-in-time index membership pulled into new snapshots via `index_constituent_timeseries`; older
  snapshots have none) and fundamental look-ahead for value/quality/growth signals (Norgate fundamentals are
  current-only, D3). Results are persisted as `backtest` runs with equity, weights history and turnover artifacts.

## D12. Unattended agent and pop-up (added 2026-09-02)
- An agent job is an ordinary co-pilot conversation with an extra, uncached system block ("unattended task … finish with
  ## Report"), so it inherits every tool and every guardrail. One job runs at a time (service-level lock); runs are
  persisted in `agent_runs` with cost, tool-call count, change ids and report; tasks in `agent_tasks` with a small
  schedule grammar (manual | startup | every Nm/h | daily HH:MM | weekdays HH:MM | weekly dow HH:MM | monthly D HH:MM).
- The scheduler is a 30-second QTimer in the main window (only while the app is open) plus a headless CLI
  (`--run-task`, `--agent-loop`) for Windows Task Scheduler. No background service is installed.
- The pop-up is a frameless always-on-top Qt window; the system-wide hotkey uses Win32 RegisterHotKey through ctypes
  and a QAbstractNativeEventFilter (no extra dependency), defaults to Ctrl+Alt+C, is configurable via the
  `quick_hotkey` setting, and degrades to the in-app Ctrl+Space shortcut if the combo is taken.
- Notifications are tray balloons plus a status-bar badge (unread reports, changes awaiting approval). Agent effort
  defaults per template (low/medium) to keep unattended spend predictable; each run's estimated cost is recorded.

## D13. YANG naming, TLH model builder and interactive how-to (added 2026-09-02)
- The co-pilot's user-facing name is YANG (system prompt persona, tabs, pop-up, tray, agent). Module names stay
  `tlh/ai/copilot.py` etc.; the screens dict key "AI co-pilot" is kept internally and mapped to the tab title "YANG".
- Pipelines are a typed block list (`optim/pipeline.py` NODE_TYPES) rendered generically by the builder's property
  panel and authored as JSON by YANG (`pipeline_schema`, `save_pipeline`, `run_pipeline`). Execution order is the
  canvas x-order with a stage tie-break, validated before running; execution reuses StrategyService, BasketRepo and
  HarvestService so a pipeline produces the same artifacts (baskets, runs) as the manual screens.
- The how-to is a dock of steps with predicate-based completion (data, entity, model, runs, baskets, backtests,
  pipelines, conversations, agent tasks, exports) so it doubles as a health checklist; it opens on first launch
  (`tour_seen` setting) and from Help / F1.

## D14. Equity risk model (ERM) (added 2026-09-02)
- Second estimator selected by `RiskModelSpec.model_kind = "erm"`; it emits the same `FittedRiskModel` so every consumer
  (optimizer, baskets, strategies, analytics) is unchanged. Industry factors use the `ind:` prefix; consumers treat
  `sec:` and `ind:` alike.
- Styles are fixed-weight composites of standardized descriptors (USE4-style weights in `descriptors.py`), re-standardized
  and orthogonalised where Barra does (resvol on beta+size, liquidity on size, non-linear size on size). Descriptors whose
  inputs are missing are dropped from their composite; a style with no usable descriptors is dropped from the model
  (e.g. liquidity when a snapshot has no volume) and the fitted spec records what was actually used.
- Estimation: daily WLS with sqrt-cap weights capped at the 95th percentile (mega-cap dominance), optional Huber
  M-estimation; the cap-weighted sum of industry factor returns is constrained to zero so `market` is the market.
  Factor t-stats and R² are stored as diagnostics.
- Covariance: EWMA volatility (hl 84) x EWMA correlation (hl 504), Newey-West with 2 lags, eigenfactor risk adjustment
  (Monte Carlo, 200 sims, scale 1.2, gammas clipped to [0.8, 3]) and volatility regime adjustment (EWMA of the
  cross-sectional bias statistic, hl 42, clipped to [0.5, 2]). Specific risk: EWMA residual variance with its own VRA,
  Bayesian shrinkage toward the cap-decile mean, and a structural log-sigma regression for names with < min_obs.
- Validation is first-class: `analytics.bias_test` refits at each period start and reports bias statistics with
  1 ± sqrt(2/n) bands for the cap-weighted market, equal-weight, holdings and long-short style portfolios.
- Known limits: fundamentals remain current-only (D3); macro block is barra_lite-only for now; the eigen adjustment
  assumes Gaussian factor returns.

## D15. Embedded gains & concentration (added 2026-09-02)
- Bracket schedules are data, not code: approximate 2026 federal LTCG/ordinary brackets per filing status with NIIT
  thresholds live in `tax/concentration.py` and are editable/saved per user (`bracket_schedule` setting). Every result
  shows the assumptions. Gains stack on a user-set "other taxable income".
- The glide path is a convex program: bracket taxes enter as sums of hinge functions (`convex_pieces`, verified equal to
  the stacked bracket tax), loss offsets are variables with cumulative availability, short-term lots pay ordinary rates
  until they turn long-term, terminal holdings pay LT tax weighted by (1 - p_stepup), and a quadratic idiosyncratic-risk
  penalty (risk aversion x specific variance x remaining value^2 / wealth) trades against taxes, costs and an alpha view.
  Policies (sell now, equal instalments, hold) are evaluated on the same objective for an honest comparison.
- Monte Carlo uses beta x market + idiosyncratic lognormal paths from the active risk model, annual bracket taxes on each
  sale, terminal LT tax unless step-up. It is a comparison tool, not a forecast.
- Hedging uses Black-Scholes with user or model vol; tax flags follow practitioner heuristics (constructive-sale band 15%,
  straddle rules on short-term lots, in-the-money covered calls) and are shown as warnings, never as clearance.
- The gain-offset plan reuses `evaluate_trade_list`, so every leg is wash-screened and the result is a normal reviewable run.

## D16. Key management, packaging (added 2026-09-02)
- The API key lives only in `.env` (git-ignored). The Settings screen writes it with `update_env_file` (preserving
  comments), reloads settings and calls `Copilot.reconfigure()` on every live instance, so no restart is needed. The
  key is never written to SQLite, the audit log or exports.
- Packaging is PyInstaller one-folder via `python setup.py build_exe` (`Build-EXE.bat`). One-folder was chosen over
  one-file because QtWebEngine and its helper processes are fragile and slow to unpack in one-file mode. In a frozen
  build `REPO_ROOT` is the exe folder (so `.env` and `var/` travel with the app) and read-only resources resolve from
  `sys._MEIPASS`. The code sandbox requires the source tree and reports that clearly when frozen.
