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

## D17. Startup and chart performance (added 2026-09-03)
- Heavy solvers are imported lazily (`tlh/lazy.py`: `cp = lazy_module("cvxpy")`, `norm = lazy_object("scipy.stats", "norm")`) and
  warmed up in a background thread after the window is shown. `pydantic-settings` was replaced by a dependency-free `.env`
  loader with the same `Settings` surface. Import of `tlh.gui.app` fell from ~5 s to ~1 s.
- Screens are built on first access (`gui/lazy_tabs.ScreenRegistry`): placeholders are added to the tab bar, the real widget
  is constructed when the tab is opened or when code touches `win.screens[name]`; unbuilt screens are marked dirty and
  refresh themselves when first built. A splash appears before the heavy imports.
- `PlotlyView` loads one HTML shell per view and then pushes figure JSON through `Plotly.react`, so chart updates no longer
  re-parse plotly.min.js or hit the `setHtml` 2 MB limit. Figures requested before the shell is ready are queued.
- The Norgate status is polled every 10 s; the red banner clears itself and a stale snapshot is pulled when NDU comes up.

## D18. Model library, statistical and dynamic risk models, calibration (added 2026-09-03)
- `RiskModelSpec.model_kind` gained `statistical` (Potomac calibrated covariance: fixed window, equal/exponential weights with
  half-life = 0.35 x window, sample or Ledoit-Wolf constant-correlation shrinkage), `pca` (asymptotic PCA, Ahn-Horenstein factor
  count) and `hybrid` (ERM + PCA on its residuals). Statistical covariances are carried as eigen-factor models (`stat:k`
  factors, unit factor variance, floored specific variance) so every consumer is unchanged.
- `cov_method` (`ewma | garch | regime`) post-processes the factor covariance of fundamental models: GARCH(1,1) per factor
  (variance-targeted MLE) with EWMA correlations for a decision horizon, or a calm/stress mixture weighted by the logistic
  probability of stress from the market-vol z-score.
- `RISK_MODEL_PRESETS` (13 named models) is the library the GUI and YANG use; `preset` is recorded in the spec and
  diagnostics. Model version names default to the preset name.
- `risk/calibration.py` ports the September 2026 calibration study (Risk_Model_Calibration v1.3): non-overlapping forward
  windows anchored to the longest lookback, 20 seeded 5-name baskets against the equal-weight universe, sampled pairs, the
  seven metrics and the within-horizon mean-rank Score. Extensions: optional PCA arm, the client's holdings as an extra
  basket, and `pair_study` for tight substitute pairs. The study's conclusions are encoded as presets (126d equal
  Ledoit-Wolf for 3–6 month horizons; sample matrix for near-identical pairs).

## D19. Advisor-grade onboarding and state taxes (added 2026-09-03)
- A `Start here` screen and `services/home_service.py` implement the three-step flow (import holdings -> set taxes -> one
  click harvest) with plain-English sentences generated in `tlh/explain.py` from the engine's numbers. `ui_mode`
  (`simple` hides Risk lab, Strategy lab, Builder, Agent, Export; `expert` shows all) is persisted per user.
- `tax/state_rates.yaml` holds all 50 states + DC (approximate 2026 planning figures, flagged everywhere): treatment
  (ordinary / none / exclusion / flat capital-gains rate / Washington excise), simplified brackets, investment-income
  surtaxes and local-tax notes. `combined_marginal` stacks federal brackets, NIIT and the state rule; `set_tax_setup`
  derives the default TaxProfile and syncs the concentration bracket schedule.
- Holdings import (`services/import_service.py`) maps broker headers by alias, tolerates title blocks and cash rows, and
  records every row through the normal ledger path; missing dates default to 400 days ago and are flagged, missing costs
  fall back to the current price and are flagged.

## D20. Long/short extensions, overlay and sample library (added 2026-09-03)
- Strategies added: `multi_factor` (integrated composite vs mixed sleeves, sector-neutral scores), `defensive_equity`
  (covariance-implied beta cap), `quality_momentum`, `long_short_extension` (130/30-style) and `overlay_neutral`
  (market-neutral extension around fixed holdings). Long/short books are solved with disjoint tranches: the bottom 40%
  by composite score is short-eligible, the rest long-eligible, so a name is never long and short at once; if the
  neutrality constraints are infeasible the tranche widens (50%, 60%) and then neutrality is relaxed with the status
  saying so. Betas come from the covariance (`_betas`), never from the Barra `market` intercept column.
- `BasketRepo.save` stores signed weights when a basket has shorts (net = 1) and keeps long-only baskets normalised.
- `optim/longshort.py` reproduces the long/short TLH economics from first principles (Monte Carlo of loss generation with
  wash lockouts, financing, tax-neutral Exchange glide) and carries the published Quantinno reference profiles as data for
  comparison only. `optim/overlay.py` sizes index-futures overlays with §1256 and §1092 flags; both are decision support
  and state that shorting/futures need custodian capability and tax counsel.
- `optim/basket_library.py` is the 17-recipe sample library built through `StrategyService.build`; recipes are data so
  YANG can add them.

## D22. TLH research laboratory (added 2026-09-04)
- Purpose: decide and defend the harvesting parameters (account size, basket size, trigger, approach, concentrated
  starts) with the distribution of historical outcomes, not one backtest. Windows start on the first trading day of every
  calendar year from 2000 (5 or 10 years, monthly review); results are medians and interquartile ranges across windows.
- Data: a separate deep store (var/research/store) built from Norgate "S&P 500 Current & Past" (1,211 symbols with prices
  since 1999, 550 delisted), split-adjusted closes + cash dividends, point-in-time membership, today's GICS sectors and
  shares. Numpy arrays memory-mapped by worker processes. The benchmark return is the real S&P 500 TR index; the
  construction cap weights are a proxy (adjusted close x today's shares) and are labelled as such in every report.
  Fundamentals are not point-in-time in Norgate, so twin pairing uses price-based descriptors (size, momentum, vol,
  beta, yield) rather than price-to-book.
- Simulator (research/engine.py) is deliberately separate from the production tax code: a lean lot model with the two
  wash rules that matter for research (no loss sale of a name bought inside 30 days, no re-buy inside 30 days of a loss
  sale), whole shares, a minimum trade, a flat cost per trade, delistings sold at the last close, dividends reinvested at
  the next review. The trigger is a lot loss as a fraction of account value (the brief's 0.01% .. 1%) or of lot cost.
  A concentrated start is unwound only as far as realised losses (plus an optional gain budget) cover the gain.
- Approaches: pairs on the fly within sector / within index (trailing 252-day correlation), SARD twin baskets (pre-paired
  same-sector twins, re-paired yearly, swap back when the twin is harvested), and a TE optimizer (buys only, sector band,
  factor-alignment bands, name cap, wash exclusions). Risk model everywhere: the calibration study's 126-day equal-weight
  Ledoit-Wolf covariance (D18), kept as the house approach. The optimizer is a direct OSQP QP (about 0.1 s per month);
  cvxpy is only a fallback. Bands relax in stages (x2, x4, sector only, box only) and the stage that worked last month is
  tried first, so the bands re-tighten when they can.
- Metrics: harvested losses per year (% of start; ST/LT split and tax value), harvest life (last month with trailing-12-month
  yield above 0.2% of value) and half-life, realised TE (daily active vs the TR index) and ex-ante forecast, turnover,
  trades, wash-window blocks, names held (whole-share rounding makes small accounts fail visibly), ending embedded gain,
  months to diversify a concentrated position. Live 2010-2020 base case ($500k, 150 names, 0.25% trigger): 1.7-2.3% a year
  harvested at 2.0-2.6% TE, half-life 28-69 months by approach; $10k holds 7 names at 11.6% TE.
- Execution: ProcessPoolExecutor over (parameter set, window); results in parquet, resumable; a full MVP study (~550
  runs) takes 20-30 minutes on this machine, quick mode (every 3rd start year) under 10.

## D23. Potomac strategies as overlay signals from fund NAVs; YANG chat rendering (added 2026-09-04)
- The five Potomac strategies (Bull Bear, Focused Growth, Guardian, Income Plus, Navigrowth) each hold the same five
  tactical funds (CRDBX, CRTPX, CRTBX, CRMVX, CRTOX) at 80% core / 4 x 5% target allocations (fact sheets; subject to
  change). `optim/potomac.py` encodes them and reads each fund's risk state off its NAV: the funds go to cash and the NAV
  prints flat. state = 0 on a day the published NAV is unchanged to the cent *and* the fund's exposure times the index
  move would have moved it by more than two cents (otherwise the day is uninformative and the previous state carries);
  exposure = state x slow beta (the fund's beta over its last 60 risk-on days, re-read monthly, so a risk-on day is worth
  ~1 for the equity funds and ~0.15 for Managed Volatility). Strategy exposure = sum of allocation x fund exposure, mapped
  onto [beta_min, beta_max]. Data: Yahoo Finance adjusted NAVs (returns) and published closes (flat test) via yfinance,
  cached 12 h under var/tactical/navs; Norgate has no mutual funds.
- Timing: every non-manual signal is generated on the prior close and traded on the next close (`SignalSpec.lag_days = 1`,
  applied to potomac, csv, rules and blends); the simulator and the recommendation use the lagged series.
- Live (2020-07 to 2026-09): Bull Bear mean target beta 1.03 with 55 changes a year at a 0.05 threshold; Income Plus
  0.33. Caveat: a flat NAV is only evidence of cash when the day was informative; low-beta funds and stale prints can
  still produce false risk-off days, hence `nav_confirm_days` (default 1) for operators who prefer confirmation.
- YANG chat: markdown rendered with markdown-it (headings become bold labels, hashtags never leak, tables and code keep
  styling), each transcript item's HTML is cached so a streamed token re-renders one bubble, the flush interval is 40 ms,
  and an animated working indicator (status spinner + pulsing bubble) shows thinking / reasoning / tool / writing states.
  The system prompt asks for chat formatting (no headings, no emoji, results first).

## D21. Levered beta without futures, margin policy, tactical overlay (added 2026-09-03)
- Futures are not available on the advisor custody platforms and direct shorting is not permitted (broker notes, Aug 2026),
  so leverage comes from leveraged / inverse ETFs (`optim/leverage.INSTRUMENTS`: SSO, SPUU, UPRO, SPXL, SH, SDS, SPXU,
  SPXS and the Nasdaq-100 set) and, optionally, Reg-T margin. These symbols are always added to the fit universe so the
  risk model carries them (their beta is covariance-implied, about 2 or 3, never assumed).
- `levered_beta` strategy: stocks + leveraged ETFs + loan `m`, minimising tracking variance versus `target_beta x benchmark`
  plus a cost term (ETF expense + volatility drag `(k^2-k)/2 sigma^2` + margin interest), subject to beta = target,
  Reg-T initial margin (loan <= 50% of long value), a house maintenance requirement of 30% on stocks and 30% x |k| on
  leveraged funds with a 25% equity buffer, per-stock and per-ETF caps and a sector band on the stock sleeve. Weights are
  fractions of equity summing to 1 + loan; `margin_max = 0` gives a cash-only book. Margin report includes the market
  drop that triggers a call.
- Tracking error first (2026-09-03, second pass). The model is built against the *index* benchmark even when the house
  benchmark is a saved basket (a levered basket set as benchmark had made the model track itself), and leveraged funds
  never count as benchmark members. Default `replicate=True`: every index name is held at 1.5 x its weight (no name
  cap, no minimum weight, the stock cap is raised to clear the largest index weight), so the stock sleeve tracks exactly
  and the only tracking error comes from the leverage layer. Leveraged funds are modelled as k x the *benchmark basket*
  (not k x SPY's fitted row, whose idiosyncratic term made them look risky and pushed the solver into margin) plus each
  fund's measured tracking variance versus k x SPY. Minimum-weight pruning re-solves with the pruned names fixed at zero
  (pro-rata re-scaling had destroyed both TE and beta). The cost term now includes the fund's embedded financing
  (k-1)(rf + swap spread) so leveraged ETFs and a margin loan are compared on the same footing; `cost_weight` defaults to
  0.1 (TE first). The QP objective is `||F'(w - beta_T wb)||^2` with `S = FF'` (SOC form; CLARABEL solves the 506-name
  problem in about 0.4 s). Diagnostics carry a realised check with actual ETF histories: the "structure" numbers replace
  the stock sleeve by the index ETF (what leverage, fees and interest add: about 0.3% a year at monthly rebalancing on
  live data), the full-book numbers are labelled look-ahead because today's index weights favour past winners.
- Tactical overlay: a signal (Potomac strategy CSV, manual beta, example rules, or a blend) gives a target beta in
  [0, 1.5]; `tactical_overlay` sizes one leveraged (up) or inverse (down) ETF that moves the *total* beta to the target
  without selling core stock, picking the cheapest instrument that fits the margin policy and reporting the tax that
  selling core instead would have realised. `simulate_tactical` backtests the signal day by day with leveraged-fund
  compounding, expense ratios, margin interest and costs. Example rules are labelled as not Potomac's models; the CSV /
  blend path is the integration point for the real strategies. Signals persist as Parquet under var/tactical.
- Snapshot frames (raw parquet, close matrices, last prices) are cached in memory per snapshot path and mtime
  (`data/cache._FRAME_CACHE`); snapshots are immutable so hits are safe. This removed the 1-2 s pivot from every fit,
  harvest, KPI refresh and strategy build.
