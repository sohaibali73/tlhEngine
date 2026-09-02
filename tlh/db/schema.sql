-- TLH Engine state-of-record schema (SQLite). Idempotent: every statement is IF NOT EXISTS.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------- tax entities & accounts
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    filing_status TEXT NOT NULL DEFAULT 'single',   -- single | mfj | mfs | hoh
    notes       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS accounts (
    id           INTEGER PRIMARY KEY,
    entity_id    INTEGER NOT NULL REFERENCES entities(id),
    name         TEXT NOT NULL,
    account_type TEXT NOT NULL DEFAULT 'taxable',   -- taxable | ira | roth | 401k | other_deferred
    broker       TEXT,
    owner        TEXT,                              -- e.g. 'self', 'spouse'
    is_active    INTEGER NOT NULL DEFAULT 1,
    notes        TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(entity_id, name)
);

-- ---------------------------------------------------------------- reference data
CREATE TABLE IF NOT EXISTS securities (
    assetid       INTEGER PRIMARY KEY,
    symbol        TEXT NOT NULL,
    name          TEXT,
    subtype1      TEXT,
    subtype2      TEXT,
    subtype3      TEXT,
    gics_sector   TEXT,
    gics_industry_group TEXT,
    gics_industry TEXT,
    gics_sub_industry TEXT,
    gics_code     TEXT,
    first_quoted  TEXT,
    last_quoted   TEXT,
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_securities_symbol ON securities(symbol);

-- ---------------------------------------------------------------- transactions & lots
CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY,
    account_id  INTEGER NOT NULL REFERENCES accounts(id),
    assetid     INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,                      -- ISO date
    tx_type     TEXT NOT NULL,                      -- BUY | SELL | DRIP | TRANSFER_IN | TRANSFER_OUT
    quantity    REAL NOT NULL,                      -- positive
    price       REAL NOT NULL,                      -- per share, pre-fees
    fees        REAL NOT NULL DEFAULT 0,
    notes       TEXT,
    source      TEXT NOT NULL DEFAULT 'manual',     -- manual | import | run:<id>
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_tx_account_asset_date ON transactions(account_id, assetid, trade_date);

CREATE TABLE IF NOT EXISTS lots (
    id                    INTEGER PRIMARY KEY,
    account_id            INTEGER NOT NULL REFERENCES accounts(id),
    assetid               INTEGER NOT NULL,
    symbol                TEXT NOT NULL,
    acquired_date         TEXT NOT NULL,            -- actual purchase date
    holding_start_date    TEXT NOT NULL,            -- may be earlier than acquired_date after wash tacking
    quantity_original     REAL NOT NULL,
    quantity_open         REAL NOT NULL,
    cost_per_share        REAL NOT NULL,            -- original cost incl. fees
    basis_adjustment      REAL NOT NULL DEFAULT 0,  -- total $ added by wash-sale disallowance
    source                TEXT NOT NULL DEFAULT 'buy',   -- buy | drip | transfer | wash_replacement
    open_tx_id            INTEGER REFERENCES transactions(id),
    is_closed             INTEGER NOT NULL DEFAULT 0,
    notes                 TEXT,
    created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_lots_account_asset ON lots(account_id, assetid, is_closed);

CREATE TABLE IF NOT EXISTS lot_closures (
    id                       INTEGER PRIMARY KEY,
    lot_id                   INTEGER NOT NULL REFERENCES lots(id),
    sell_tx_id               INTEGER NOT NULL REFERENCES transactions(id),
    sale_date                TEXT NOT NULL,
    quantity                 REAL NOT NULL,
    proceeds                 REAL NOT NULL,         -- net of fees
    cost_basis               REAL NOT NULL,         -- incl. any basis adjustment share
    realized_gain            REAL NOT NULL,         -- proceeds - cost_basis (pre-wash)
    term                     TEXT NOT NULL,         -- ST | LT
    wash_disallowed          REAL NOT NULL DEFAULT 0,
    wash_replacement_lot_id  INTEGER REFERENCES lots(id),
    wash_explanation         TEXT,
    created_at               TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_closures_date ON lot_closures(sale_date);

-- Known future purchases / DRIPs that the forward-looking wash-sale guard must see.
CREATE TABLE IF NOT EXISTS scheduled_events (
    id          INTEGER PRIMARY KEY,
    account_id  INTEGER NOT NULL REFERENCES accounts(id),
    assetid     INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    event_date  TEXT NOT NULL,
    event_type  TEXT NOT NULL,                      -- BUY | DRIP
    quantity    REAL,                               -- may be NULL for DRIP of unknown size
    est_value   REAL,
    notes       TEXT,
    is_active   INTEGER NOT NULL DEFAULT 1
);

-- ---------------------------------------------------------------- tax assumptions
CREATE TABLE IF NOT EXISTS tax_profiles (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    fed_st_rate     REAL NOT NULL,
    fed_lt_rate     REAL NOT NULL,
    state_rate      REAL NOT NULL DEFAULT 0,
    niit_rate       REAL NOT NULL DEFAULT 0.038,
    ordinary_offset REAL NOT NULL DEFAULT 3000,
    is_default      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS carryforwards (
    id          INTEGER PRIMARY KEY,
    entity_id   INTEGER NOT NULL REFERENCES entities(id),
    tax_year    INTEGER NOT NULL,
    st_amount   REAL NOT NULL DEFAULT 0,            -- positive = loss available
    lt_amount   REAL NOT NULL DEFAULT 0,
    notes       TEXT,
    UNIQUE(entity_id, tax_year)
);

-- ---------------------------------------------------------------- data snapshots & models
CREATE TABLE IF NOT EXISTS snapshots (
    id            TEXT PRIMARY KEY,                 -- e.g. 2026-09-02T1330_sp500
    as_of_date    TEXT NOT NULL,                    -- last price date contained
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    universe_name TEXT NOT NULL,
    n_symbols     INTEGER NOT NULL,
    path          TEXT NOT NULL,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    snapshot_id    TEXT REFERENCES snapshots(id),
    as_of_date     TEXT NOT NULL,
    universe_name  TEXT NOT NULL,
    lookback_days  INTEGER NOT NULL,
    factor_list    TEXT NOT NULL,                   -- JSON array
    diagnostics    TEXT NOT NULL,                   -- JSON
    artifact_path  TEXT NOT NULL,
    code_version_id INTEGER,
    is_active      INTEGER NOT NULL DEFAULT 0,
    notes          TEXT
);

-- ---------------------------------------------------------------- recommendation runs
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    run_type         TEXT NOT NULL,                 -- harvest | frontier
    as_of_date       TEXT NOT NULL,
    entity_id        INTEGER REFERENCES entities(id),
    snapshot_id      TEXT REFERENCES snapshots(id),
    model_version_id INTEGER REFERENCES model_versions(id),
    params           TEXT NOT NULL,                 -- JSON (constraint hierarchy, budgets, mode)
    summary          TEXT NOT NULL,                 -- JSON (TE before/after, harvested loss, tax alpha...)
    artifact_path    TEXT,                          -- folder with detail parquet/json
    status           TEXT NOT NULL DEFAULT 'ok',
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS run_trades (
    id                INTEGER PRIMARY KEY,
    run_id            INTEGER NOT NULL REFERENCES runs(id),
    account_id        INTEGER NOT NULL REFERENCES accounts(id),
    assetid           INTEGER NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,                -- SELL | BUY
    quantity          REAL NOT NULL,
    est_price         REAL NOT NULL,
    est_value         REAL NOT NULL,
    lot_id            INTEGER REFERENCES lots(id),
    realized_gain     REAL,                         -- negative = loss (SELL only)
    term              TEXT,                         -- ST | LT
    tax_benefit       REAL,
    wash_status       TEXT,                         -- SAFE | BLOCKED | FLAGGED
    wash_explanation  TEXT,
    replacement_for   TEXT,                         -- symbol this BUY replaces
    acted_on          INTEGER NOT NULL DEFAULT 0,
    acted_at          TEXT
);
CREATE INDEX IF NOT EXISTS ix_run_trades_run ON run_trades(run_id);

-- ---------------------------------------------------------------- AI-authored code & audit
CREATE TABLE IF NOT EXISTS code_versions (
    id                INTEGER PRIMARY KEY,
    module_path       TEXT NOT NULL,                -- repo-relative, e.g. tlh/risk/factors.py
    version_no        INTEGER NOT NULL,
    code_text         TEXT NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    source            TEXT NOT NULL,                -- human | ai | rollback
    parent_version_id INTEGER,
    change_id         INTEGER,
    is_active         INTEGER NOT NULL DEFAULT 0,
    UNIQUE(module_path, version_no)
);

CREATE TABLE IF NOT EXISTS ai_conversations (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    title       TEXT,
    model       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_messages (
    id               INTEGER PRIMARY KEY,
    conversation_id  INTEGER NOT NULL REFERENCES ai_conversations(id),
    role             TEXT NOT NULL,                 -- user | assistant
    content          TEXT NOT NULL,                 -- JSON content blocks
    usage            TEXT,                          -- JSON
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ai_changes (
    id                   INTEGER PRIMARY KEY,
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    conversation_id      INTEGER REFERENCES ai_conversations(id),
    module_path          TEXT NOT NULL,
    title                TEXT NOT NULL,
    rationale            TEXT,
    prompt_excerpt       TEXT,
    proposed_code        TEXT NOT NULL,
    diff_text            TEXT,
    sandbox_stdout       TEXT,
    sandbox_passed       INTEGER,
    sandbox_ran_at       TEXT,
    status               TEXT NOT NULL DEFAULT 'drafted',  -- drafted | tested | approved | rejected | promoted | rolled_back
    approved_by          TEXT,
    approved_at          TEXT,
    promoted_version_id  INTEGER REFERENCES code_versions(id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id       INTEGER PRIMARY KEY,
    ts       TEXT NOT NULL DEFAULT (datetime('now')),
    actor    TEXT NOT NULL,                         -- user | system | ai
    action   TEXT NOT NULL,
    target   TEXT,
    details  TEXT                                    -- JSON
);

CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL                              -- JSON
);

-- ---------------------------------------------------------------- model portfolios / baskets
CREATE TABLE IF NOT EXISTS baskets (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    description    TEXT,
    source         TEXT NOT NULL DEFAULT 'manual',   -- manual | ai | optimizer | index
    benchmark_name TEXT,
    params         TEXT,                             -- JSON (optimizer settings used)
    metrics        TEXT,                             -- JSON (TE, n names, exposures at creation)
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS basket_members (
    id         INTEGER PRIMARY KEY,
    basket_id  INTEGER NOT NULL REFERENCES baskets(id) ON DELETE CASCADE,
    symbol     TEXT NOT NULL,
    assetid    INTEGER,
    weight     REAL NOT NULL,
    UNIQUE(basket_id, symbol)
);

-- ---------------------------------------------------------------- unattended agent tasks
CREATE TABLE IF NOT EXISTS agent_tasks (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    prompt        TEXT NOT NULL,
    schedule      TEXT NOT NULL DEFAULT 'manual',   -- manual | startup | every 30m | daily 08:30 | weekdays 16:30 | weekly mon 09:00 | monthly 1 09:00
    enabled       INTEGER NOT NULL DEFAULT 1,
    notify        INTEGER NOT NULL DEFAULT 1,
    effort        TEXT,                              -- override co-pilot effort for this task (low|medium|high|xhigh|max)
    last_run_at   TEXT,
    next_run_at   TEXT,
    last_status   TEXT,
    last_summary  TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id               INTEGER PRIMARY KEY,
    task_id          INTEGER REFERENCES agent_tasks(id),
    name             TEXT NOT NULL,
    prompt           TEXT NOT NULL,
    started_at       TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at      TEXT,
    status           TEXT NOT NULL DEFAULT 'running',   -- running | done | failed | cancelled
    report           TEXT,
    error            TEXT,
    conversation_id  INTEGER REFERENCES ai_conversations(id),
    change_ids       TEXT,                              -- JSON list
    tool_calls       INTEGER,
    cost_usd         REAL,
    duration_s       REAL,
    trigger          TEXT NOT NULL DEFAULT 'manual',    -- manual | schedule | popup | cli
    is_read          INTEGER NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------- TLH model pipelines (drag-and-drop builder)
CREATE TABLE IF NOT EXISTS pipelines (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    spec        TEXT NOT NULL,                       -- JSON (Pipeline.to_json)
    description TEXT,
    source      TEXT NOT NULL DEFAULT 'builder',     -- builder | ai | example
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
