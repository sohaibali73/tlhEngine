"""Versioned market-data snapshots (Parquet on disk, DuckDB for queries, SQLite row for the catalogue).

A snapshot is the reproducibility unit: universe + prices + reference + fundamentals + macro as pulled at one
moment. Runs and model fits record the snapshot id they consumed, so any past result can be re-derived
without touching Norgate.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from ..db.database import Database
from .norgate import NorgateClient

log = logging.getLogger(__name__)


@dataclass
class Snapshot:
    id: str
    as_of_date: date
    universe_name: str
    path: Path
    n_symbols: int
    created_at: str = ""
    notes: str | None = None

    # lazy loaders -----------------------------------------------------------------------
    def _read(self, name: str) -> pd.DataFrame:
        p = self.path / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def prices(self) -> pd.DataFrame:
        return self._read("prices")

    def securities(self) -> pd.DataFrame:
        return self._read("securities")

    def fundamentals(self) -> pd.DataFrame:
        return self._read("fundamentals")

    def macro(self) -> pd.DataFrame:
        df = self._read("macro")
        if not df.empty and "date" in df.columns:
            df = df.set_index("date")
        return df

    def membership(self) -> pd.DataFrame:
        """Wide date x symbol boolean matrix of index membership (empty if the snapshot has none)."""
        df = self._read("membership")
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot_table(index="date", columns="symbol", values="member", aggfunc="last").sort_index()
        wide.index = pd.to_datetime(wide.index)
        return wide.ffill().fillna(0).astype(bool)

    def manifest(self) -> dict:
        p = self.path / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def close_matrix(self, field: str = "close") -> pd.DataFrame:
        """Wide date x symbol matrix straight from Parquet via DuckDB (fast pivot)."""
        p = (self.path / "prices.parquet").as_posix()
        con = duckdb.connect()
        try:
            df = con.execute(
                f"SELECT date, symbol, {field} AS v FROM read_parquet('{p}') ORDER BY date"
            ).df()
        finally:
            con.close()
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot(index="date", columns="symbol", values="v").sort_index()
        wide.index = pd.to_datetime(wide.index)
        return wide

    def last_prices(self) -> pd.Series:
        p = (self.path / "prices.parquet").as_posix()
        con = duckdb.connect()
        try:
            df = con.execute(
                f"""SELECT symbol, arg_max(unadj_close, date) AS px FROM read_parquet('{p}')
                    WHERE unadj_close IS NOT NULL GROUP BY symbol"""
            ).df()
        finally:
            con.close()
        return df.set_index("symbol")["px"].astype(float) if not df.empty else pd.Series(dtype=float)

    def symbols(self) -> list[str]:
        sec = self.securities()
        return sorted(sec["symbol"].tolist()) if not sec.empty else []


class SnapshotStore:
    def __init__(self, db: Database, root: Path, client: NorgateClient | None = None):
        self.db = db
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = client or NorgateClient()

    # ------------------------------------------------------------------ catalogue
    def list(self) -> list[Snapshot]:
        rows = self.db.fetchall("SELECT * FROM snapshots ORDER BY created_at DESC")
        return [self._from_row(r) for r in rows]

    def get(self, snapshot_id: str) -> Snapshot | None:
        r = self.db.fetchone("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,))
        return self._from_row(r) if r else None

    def latest(self, universe_name: str | None = None) -> Snapshot | None:
        if universe_name:
            r = self.db.fetchone("SELECT * FROM snapshots WHERE universe_name = ? ORDER BY created_at DESC LIMIT 1",
                                 (universe_name,))
        else:
            r = self.db.fetchone("SELECT * FROM snapshots ORDER BY created_at DESC LIMIT 1")
        return self._from_row(r) if r else None

    def _from_row(self, r) -> Snapshot:
        return Snapshot(id=r["id"], as_of_date=date.fromisoformat(r["as_of_date"]), universe_name=r["universe_name"],
                        path=Path(r["path"]), n_symbols=int(r["n_symbols"]), created_at=r["created_at"],
                        notes=r["notes"])

    # ------------------------------------------------------------------ creation
    def create(self, universe_name: str, symbols: list[str], start: str | date, notes: str | None = None,
               progress=None) -> Snapshot:
        """Pull everything for `symbols` and persist as a new snapshot. `progress(msg)` is optional."""
        self.client.require()
        symbols = sorted(set(symbols))
        say = progress or (lambda m: log.info(m))
        stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        sid = f"{stamp}_{_slug(universe_name)}"
        while self.get(sid) is not None or (self.root / sid).exists():      # never collide, even for concurrent pulls
            import uuid
            sid = f"{stamp}_{_slug(universe_name)}_{uuid.uuid4().hex[:4]}"
        path = self.root / sid
        path.mkdir(parents=True, exist_ok=True)

        say(f"Pulling prices for {len(symbols)} symbols from {start}...")
        prices = self.client.price_panel(symbols, start)
        if prices.empty:
            raise RuntimeError("no price data returned; check symbols and NDU")
        prices.to_parquet(path / "prices.parquet", index=False)
        as_of = pd.Timestamp(prices["date"].max()).date()

        say("Pulling reference data & GICS classification...")
        sec = self.client.securities_meta(symbols)
        sec.to_parquet(path / "securities.parquet", index=False)

        say("Pulling fundamentals...")
        fund = self.client.fundamentals_table(symbols)
        fund.to_parquet(path / "fundamentals.parquet", index=False)

        say("Pulling point-in-time index membership...")
        try:
            mem = self.client.index_constituents(symbols, universe_name, start)
            if not mem.empty:
                mem["date"] = pd.to_datetime(mem["date"]).dt.normalize()
                mem.to_parquet(path / "membership.parquet", index=False)
        except Exception as e:  # Platinum-only feature or non-index universe name
            log.warning("index membership unavailable for %s: %s", universe_name, e)

        say("Pulling macro series...")
        macro = self.client.macro_panel(start)
        macro.index.name = "date"
        macro.reset_index().to_parquet(path / "macro.parquet", index=False)

        manifest = {
            "id": sid, "universe_name": universe_name, "requested_symbols": symbols,
            "returned_symbols": sorted(prices["symbol"].unique().tolist()),
            "start": str(start), "as_of_date": as_of.isoformat(), "created_at": datetime.now().isoformat(),
            "n_price_rows": int(len(prices)), "notes": notes,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        self.db.insert("snapshots", id=sid, as_of_date=as_of.isoformat(), universe_name=universe_name,
                       n_symbols=len(manifest["returned_symbols"]), path=str(path), notes=notes)
        self.db.audit("system", "snapshot.create", sid, universe=universe_name, n=len(symbols))
        say(f"Snapshot {sid} ready (as of {as_of}).")
        return self.get(sid)  # type: ignore[return-value]

    def delete(self, snapshot_id: str) -> None:
        snap = self.get(snapshot_id)
        if snap is None:
            return
        import shutil
        shutil.rmtree(snap.path, ignore_errors=True)
        self.db.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
        self.db.audit("user", "snapshot.delete", snapshot_id)


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")[:40]
