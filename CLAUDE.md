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
- risk/     barra_lite (factors.py, model.py), ERM (descriptors.py, erm.py), analytics.py (decomposition/stress/VaR/bias), benchmark.py, custom/
- optim/    harvest optimizer (cvxpy), constraint hierarchy, frontier sweep, replacement selection, basket construction,
            strategies.py (12 construction methods, one run_strategy contract), backtest.py (walk-forward)
- ai/       tools.py (schemas + executor), copilot.py (stream loop, promotion), sandbox.py, registry.py
- export/   xlsxwriter workbook builder
- services/ application services that compose the layers for the GUI (no Qt imports here)
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
- Adding a co-pilot tool = a schema entry in tlh/ai/tools.py TOOLS plus a `t_<name>` method on ToolExecutor.
- Persisted conversation content must stay API-replayable: use sanitize_block, never raw model_dump().
