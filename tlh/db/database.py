"""SQLite connection management and tiny query helpers.

One `Database` per process. Connections are per-thread (sqlite3 objects are not shareable across the GUI
thread and worker threads), all in WAL mode with foreign keys on.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
if not SCHEMA_PATH.exists():                      # frozen build: resources under sys._MEIPASS
    import sys
    SCHEMA_PATH = Path(getattr(sys, "_MEIPASS", ".")) / "tlh" / "db" / "schema.sql"
SCHEMA_VERSION = 1


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialise()

    # ------------------------------------------------------------------ connections
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _initialise(self) -> None:
        with self._init_lock:
            sql = SCHEMA_PATH.read_text(encoding="utf-8")
            self.conn.executescript(sql)
            row = self.conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            if row["v"] is None:
                self.conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))

    # ------------------------------------------------------------------ transactions
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ helpers
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        self.conn.executemany(sql, [tuple(r) for r in rows])

    def fetchone(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, tuple(params)).fetchone()

    def fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchall()

    def insert(self, table: str, **cols: Any) -> int:
        keys = list(cols)
        sql = f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})"
        cur = self.conn.execute(sql, tuple(_adapt(v) for v in cols.values()))
        return int(cur.lastrowid)

    def update(self, table: str, where: str, where_params: Iterable[Any], **cols: Any) -> int:
        sets = ", ".join(f"{k} = ?" for k in cols)
        sql = f"UPDATE {table} SET {sets} WHERE {where}"
        cur = self.conn.execute(sql, tuple(_adapt(v) for v in cols.values()) + tuple(where_params))
        return cur.rowcount

    # settings -----------------------------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.fetchone("SELECT value FROM settings WHERE key = ?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, default=str)),
        )

    # audit --------------------------------------------------------------------------------
    def audit(self, actor: str, action: str, target: str | None = None, **details: Any) -> None:
        self.insert("audit_log", actor=actor, action=action, target=target,
                    details=json.dumps(details, default=str) if details else None)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def _adapt(v: Any) -> Any:
    """Convert Python values to SQLite-storable ones."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, dict | list):
        return json.dumps(v, default=str)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]
