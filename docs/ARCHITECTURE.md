# Architecture

Five decoupled layers; every layer talks to the one below through plain Python objects (DataFrames, dataclasses),
never through the GUI or the database directly.

```
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │ Presentation  gui/  PySide6 screens · Plotly in QWebEngineView · xlsxwriter export  │
 ├────────────────────────────────────────────────────────────────────────────────────┤
 │ Services      services/  AppContext · DataService · PortfolioService · RiskService  │
 │                          HarvestService · demo                                      │
 ├──────────────────────┬─────────────────────────┬───────────────────────────────────┤
 │ Risk & Optimisation  │ Portfolio & Tax         │ AI Orchestration                  │
 │ risk/ optim/         │ tax/                    │ ai/ (registry, sandbox, copilot)  │
 ├──────────────────────┴─────────────────────────┴───────────────────────────────────┤
 │ Data          data/  NorgateClient · SnapshotStore (Parquet/DuckDB) · substitutes  │
 ├────────────────────────────────────────────────────────────────────────────────────┤
 │ Persistence   db/    SQLite (state of record) · var/ (snapshots, models, runs)      │
 └────────────────────────────────────────────────────────────────────────────────────┘
```

## Data flow of one harvest run

1. `DataService.ensure_snapshot` pulls universe = index watchlist ∪ held ∪ substitutes into
   `var/cache/snapshots/<id>/{prices,securities,fundamentals,macro}.parquet` and catalogues it in `snapshots`.
2. `RiskService.fit` reads the snapshot, runs `FactorRiskModel.fit`, saves the artifact under `var/models/model_<id>`
   and inserts a `model_versions` row (active).
3. `PortfolioRepo.load_book` hydrates a `LotBook` (lots, closures, scheduled events) with the substantially-identical
   groups from `substitutes.yaml` resolved to assetids.
4. `HarvestService.build_inputs` assembles `HarvestInputs`: open lots, last prices, active model, benchmark weights,
   tax profile, acquisitions (past + scheduled), loss sales in the last 30 days, returns matrix, securities.
5. `run_harvest` screens every loss lot with `screen_proposed_sale` and every replacement candidate with
   `screen_proposed_buy`, solves the QP, applies the minimum-trade second pass, converts to shares, and re-screens the
   final trade list (raises if anything is unsafe).
6. `HarvestService._persist` writes a `runs` row + `run_trades` rows + a `var/runs/run_<id>/` folder with every
   DataFrame the GUI needs to redisplay the run without recomputation.
7. The GUI shows the result; Export builds the workbook from the persisted run.

## Wash-sale engine in two directions

* Sale time: `LotBook.record_sale` → `evaluate_loss_sale` against all acquisitions in the entity within ±30 days
  (never the lot's own shares; shares already used as replacement are consumed once). Disallowed loss is added to the
  replacement lot's basis and the holding period tacks. Tax-deferred replacement → permanent disallowance.
* Purchase time: `LotBook.record_purchase` → `_apply_retroactive_wash` looks back 30 days for loss closures in the
  same group and washes them against the new lot.
* Pre-trade: the optimizer and the Portfolio screen call the pure `screen_*` functions on deep copies, so screening
  never mutates state.

## AI change loop

```
chat → tool: read_module / test_change (sandbox) … → tool: propose_change
     → ai_changes row (diff, sandbox stdout, passed?) → GUI review → approve_and_promote
     → code_versions row (prior kept) → file written → importlib.reload chain → status=promoted → audit_log
```
Rollback writes any prior version back and records a new version row (`source=rollback`). The sandbox stages a copy
of `tlh/` + `tests/` under `var/sandbox/`, points `TLH_VAR_DIR` at an empty var dir, strips the API key from the
environment, and runs pytest with a timeout.

## Threading

All long operations run through `gui/workers.run_task` on Qt's global thread pool; SQLite connections are per-thread
(WAL mode). Screens receive results via signals and never block the UI thread.
