# TLH Engine — project guide for Claude Code

Desktop tax-loss-harvesting engine for a single quant operator. Python 3.12, PySide6 GUI, Norgate data,
cvxpy optimizer, embedded co-pilot YANG (built on Claude). See DECISIONS.md for every stack decision and the tax-rule
conventions; see docs/ARCHITECTURE.md for the layer map.

## Run
```
python -m tlh                # launch GUI
python -m tlh --seed-demo    # create demo entity/accounts/lots (idempotent), then launch
python -m pytest             # full test suite (wash-sale + lot tests are the gate; never skip them)
PYTHONPATH=. python scripts/smoke_tools.py   # every co-pilot tool against live state; smoke_gui.py, smoke_copilot.py likewise
ruff check tlh tests
```
Norgate Data Updater must be running. Secrets live in .env (git-ignored): ANTHROPIC_API_KEY, optional
FRED_API_KEY, TLH_AI_MODEL, TLH_AI_EFFORT, TLH_VAR_DIR.

## Layers (tlh/)
- data/     Norgate wrapper, Parquet/DuckDB snapshot cache, macro series, substitutes.yaml + loader
- tax/      holding-period rules, lots + lot-selection methods, wash-sale engine, ledger + carryforward,
            tax profile / after-tax math
- risk/     barra_lite (factors.py, model.py incl. RISK_MODEL_PRESETS library), ERM (descriptors.py, erm.py), statistical.py (calibrated cov, PCA, hybrid, GARCH, regime),
            calibration.py (walk-forward study), analytics.py (decomposition/stress/VaR/bias), benchmark.py, custom/
- optim/    harvest optimizer (cvxpy), constraint hierarchy, frontier sweep, replacement selection, basket construction,
            strategies.py (17 construction methods incl. long/short, one run_strategy contract), basket_library.py (sample recipes),
            longshort.py (L/S TLH economics), leverage.py (leveraged/inverse ETFs, Reg-T margin policy, levered_beta, tactical overlay sizing +
            daily simulator), tactical.py (target-beta signals: manual / Potomac CSV / example rules / blends), overlay.py (futures reference),
            glidepath.py, backtest.py (walk-forward)
- research/ due-diligence backtesting lab: data.py (deep Norgate store: every S&P 500 member since 1999, memory-mapped numpy), spec.py
            (ResearchSpec / StudySpec + the parameter grids), engine.py (monthly lot-level simulator: wash windows, whole shares, pairs /
            twin-basket / optimizer reinvestment, concentrated unwind, metrics), grid.py (sweep design, ProcessPool runner, parquet results,
            summaries), report.py (markdown + xlsx write-up)
- ai/       tools.py (schemas + executor), copilot.py (stream loop, promotion), sandbox.py, registry.py
- export/   xlsxwriter workbook builder
- services/ application services that compose the layers for the GUI (no Qt imports here); home_service.py = Start-here one-click flow,
            import_service.py = broker holdings import
- gui/      PySide6 app; screens/ one module per screen; charts.py Plotly -> QWebEngineView
- db/       schema.sql, connection, repositories

## Rules
- Nothing places orders. Trade tickets are the terminal output.
- Tax and wash-sale code is highest-scrutiny: every rule change needs a test with a hand-computed expected
  value in tests/test_washsale.py or tests/test_holding.py.
- Persist with assetid, display with symbol.
- Never overwrite a promoted model or code version; add a new version row.
- GUI code never computes; it calls services. Services never import Qt.
- AI-editable modules are listed in tlh/ai/registry.py. Add there before letting the co-pilot touch a file.
- TLH model builder: tlh/optim/pipeline.py (block schema/validation), services/pipeline_service.py (execution), gui/screens/builder.py; how-to steps in gui/tour.py.
- Concentration workbench: tax/concentration.py (brackets, collars, alternatives), optim/glidepath.py (glide path + Monte Carlo), services/concentration_service.py, gui/screens/concentration.py.
- Agent tasks: tlh/services/agent_service.py (templates, scheduler logic), tlh/ai/schedule.py (grammar), gui/quick.py (pop-up/hotkey/tray).
- Adding a strategy = a function in tlh/optim/strategies.py + _DISPATCH + STRATEGIES entry + a test in tests/test_strategies.py.
- Adding a risk-model preset = an entry in RISK_MODEL_PRESETS (tlh/risk/model.py); a sample basket = a BasketRecipe in tlh/optim/basket_library.py.
- Heavy imports (cvxpy, scipy.stats) go through tlh/lazy.py; screens are built lazily (gui/lazy_tabs.py). Never import cvxpy at module top level in gui/ or services/.
- State tax figures live in tlh/tax/state_rates.yaml (approximate planning values; always labelled as such in the GUI).
- Leverage never comes from futures or short sales (custodian constraint): use INSTRUMENTS in tlh/optim/leverage.py and the MarginPolicy.
  Leveraged funds are modelled as k x the index basket (nominal leverage), not as fitted; `levered_beta` always builds against the index
  benchmark with full replication by default (TE first; see DECISIONS D21). Tactical overlay screen = services/tactical_service.py.
- Potomac strategies (optim/potomac.py): five funds CRDBX/CRTPX/CRTBX/CRMVX/CRTOX at 80/5/5/5/5; NAVs from yfinance (not Norgate), cached under
  var/tactical/navs; flat NAV = risk-off; every non-manual signal is lagged one day (prior-close signal, next-close trade). Tests: tests/test_potomac.py.
- YANG chat rendering lives in gui/screens/copilot.py (md_to_html via markdown-it; headings -> bold labels). quick.py and agent.py reuse it.
- Adding a co-pilot tool = a schema entry in tlh/ai/tools.py TOOLS plus a `t_<name>` method on ToolExecutor.
- TLH research (screen "TLH research", expert mode): services/research_service.py; the store lives in var/research/store (rebuild from the screen or
  the research_store tool); studies in var/research/studies/<name>/ (results.parquet + monthly.parquet, resumable). Adding a harvesting approach =
  a branch in engine._reinvest/_construct + APPROACHES entry + a test; adding a sweep = grid.design + StudySpec field + SWEEPS in the screen.
- Persisted conversation content must stay API-replayable: use sanitize_block, never raw model_dump().
