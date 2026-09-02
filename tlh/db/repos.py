"""Repositories: typed access to the SQLite state plus LotBook hydration/persistence.

The tax engine (`tax/ledger.LotBook`) is pure Python; this module is the only place that translates between
it and the database. Every mutation happens inside one SQLite transaction and writes an audit row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from ..tax.ledger import Closure, LotBook
from ..tax.lots import Lot, LotMethod
from ..tax.rates import TaxProfile
from ..tax.washsale import Acquisition, SubstantiallyIdentical
from .database import Database, rows_to_dicts


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


# ============================================================================ entities / accounts
@dataclass
class Account:
    id: int
    entity_id: int
    name: str
    account_type: str
    broker: str | None = None
    owner: str | None = None
    is_active: bool = True

    @property
    def is_taxable(self) -> bool:
        return self.account_type == "taxable"


class EntityRepo:
    def __init__(self, db: Database):
        self.db = db

    def list(self) -> list[dict]:
        return rows_to_dicts(self.db.fetchall("SELECT * FROM entities ORDER BY id"))

    def get_or_create(self, name: str, filing_status: str = "single") -> int:
        r = self.db.fetchone("SELECT id FROM entities WHERE name = ?", (name,))
        if r:
            return int(r["id"])
        eid = self.db.insert("entities", name=name, filing_status=filing_status)
        self.db.audit("user", "entity.create", str(eid), name=name)
        return eid

    def accounts(self, entity_id: int | None = None, active_only: bool = True) -> list[Account]:
        sql = "SELECT * FROM accounts WHERE 1=1"
        params: list[Any] = []
        if entity_id is not None:
            sql += " AND entity_id = ?"
            params.append(entity_id)
        if active_only:
            sql += " AND is_active = 1"
        rows = self.db.fetchall(sql + " ORDER BY id", params)
        return [Account(r["id"], r["entity_id"], r["name"], r["account_type"], r["broker"], r["owner"],
                        bool(r["is_active"])) for r in rows]

    def account(self, account_id: int) -> Account:
        r = self.db.fetchone("SELECT * FROM accounts WHERE id = ?", (account_id,))
        if not r:
            raise KeyError(f"account {account_id}")
        return Account(r["id"], r["entity_id"], r["name"], r["account_type"], r["broker"], r["owner"],
                       bool(r["is_active"]))

    def get_or_create_account(self, entity_id: int, name: str, account_type: str = "taxable",
                              broker: str | None = None, owner: str | None = None) -> int:
        r = self.db.fetchone("SELECT id FROM accounts WHERE entity_id = ? AND name = ?", (entity_id, name))
        if r:
            return int(r["id"])
        aid = self.db.insert("accounts", entity_id=entity_id, name=name, account_type=account_type,
                             broker=broker, owner=owner)
        self.db.audit("user", "account.create", str(aid), name=name, type=account_type)
        return aid


# ============================================================================ securities
class SecurityRepo:
    def __init__(self, db: Database):
        self.db = db

    def upsert(self, meta: dict) -> None:
        cols = ["assetid", "symbol", "name", "subtype1", "subtype2", "subtype3", "gics_sector",
                "gics_industry_group", "gics_industry", "gics_sub_industry", "gics_code", "first_quoted", "last_quoted"]
        vals = [meta.get(c) for c in cols]
        sets = ", ".join(f"{c} = excluded.{c}" for c in cols[1:])
        self.db.execute(
            f"INSERT INTO securities ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            f"ON CONFLICT(assetid) DO UPDATE SET {sets}, updated_at = datetime('now')", vals)

    def upsert_frame(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        with self.db.transaction():
            for rec in df.to_dict("records"):
                self.upsert({k: (None if pd.isna(v) else v) if not isinstance(v, str) else v for k, v in rec.items()})

    def by_symbol(self, symbol: str) -> dict | None:
        r = self.db.fetchone("SELECT * FROM securities WHERE symbol = ?", (symbol,))
        return dict(r) if r else None

    def by_assetid(self, assetid: int) -> dict | None:
        r = self.db.fetchone("SELECT * FROM securities WHERE assetid = ?", (assetid,))
        return dict(r) if r else None

    def resolve(self, symbol: str) -> int | None:
        r = self.by_symbol(symbol)
        return int(r["assetid"]) if r else None

    def all(self) -> pd.DataFrame:
        return pd.DataFrame(rows_to_dicts(self.db.fetchall("SELECT * FROM securities")))


# ============================================================================ tax profiles / carryforwards
class TaxRepo:
    def __init__(self, db: Database):
        self.db = db

    def profiles(self) -> list[TaxProfile]:
        rows = self.db.fetchall("SELECT * FROM tax_profiles ORDER BY is_default DESC, id")
        return [self._to_profile(r) for r in rows]

    def default_profile(self) -> TaxProfile:
        r = self.db.fetchone("SELECT * FROM tax_profiles ORDER BY is_default DESC, id LIMIT 1")
        if r is None:
            pid = self.save(TaxProfile(name="default"), make_default=True)
            r = self.db.fetchone("SELECT * FROM tax_profiles WHERE id = ?", (pid,))
        return self._to_profile(r)

    def _to_profile(self, r) -> TaxProfile:
        return TaxProfile(name=r["name"], fed_st_rate=r["fed_st_rate"], fed_lt_rate=r["fed_lt_rate"],
                          state_rate=r["state_rate"], niit_rate=r["niit_rate"], ordinary_offset=r["ordinary_offset"],
                          id=r["id"])

    def save(self, p: TaxProfile, make_default: bool = False) -> int:
        with self.db.transaction():
            if make_default:
                self.db.execute("UPDATE tax_profiles SET is_default = 0")
            r = self.db.fetchone("SELECT id FROM tax_profiles WHERE name = ?", (p.name,))
            if r:
                self.db.update("tax_profiles", "id = ?", (r["id"],), fed_st_rate=p.fed_st_rate, fed_lt_rate=p.fed_lt_rate,
                               state_rate=p.state_rate, niit_rate=p.niit_rate, ordinary_offset=p.ordinary_offset,
                               is_default=int(make_default))
                pid = int(r["id"])
            else:
                pid = self.db.insert("tax_profiles", name=p.name, fed_st_rate=p.fed_st_rate, fed_lt_rate=p.fed_lt_rate,
                                     state_rate=p.state_rate, niit_rate=p.niit_rate, ordinary_offset=p.ordinary_offset,
                                     is_default=int(make_default))
            self.db.audit("user", "tax_profile.save", p.name)
        return pid

    def carryforward(self, entity_id: int, tax_year: int) -> tuple[float, float]:
        r = self.db.fetchone("SELECT st_amount, lt_amount FROM carryforwards WHERE entity_id = ? AND tax_year = ?",
                             (entity_id, tax_year))
        return (float(r["st_amount"]), float(r["lt_amount"])) if r else (0.0, 0.0)

    def set_carryforward(self, entity_id: int, tax_year: int, st: float, lt: float, notes: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO carryforwards(entity_id, tax_year, st_amount, lt_amount, notes) VALUES (?,?,?,?,?) "
            "ON CONFLICT(entity_id, tax_year) DO UPDATE SET st_amount=excluded.st_amount, lt_amount=excluded.lt_amount, "
            "notes=excluded.notes", (entity_id, tax_year, st, lt, notes))
        self.db.audit("user", "carryforward.set", f"{entity_id}:{tax_year}", st=st, lt=lt)


# ============================================================================ portfolio (lots, transactions)
class PortfolioRepo:
    def __init__(self, db: Database):
        self.db = db
        self.entities = EntityRepo(db)

    # ------------------------------------------------------------------ hydration
    def load_book(self, entity_id: int, groups: SubstantiallyIdentical | None = None,
                  include_closed: bool = True) -> LotBook:
        book = LotBook(groups=groups or SubstantiallyIdentical())
        accts = {a.id: a for a in self.entities.accounts(entity_id, active_only=False)}
        if not accts:
            return book
        ph = ",".join("?" * len(accts))
        sql = f"SELECT * FROM lots WHERE account_id IN ({ph})"
        if not include_closed:
            sql += " AND is_closed = 0"
        rows = self.db.fetchall(sql + " ORDER BY holding_start_date, id", list(accts))
        lots_by_id: dict[int, Lot] = {}
        for r in rows:
            lot = Lot(id=r["id"], account_id=r["account_id"], assetid=r["assetid"], symbol=r["symbol"],
                      acquired_date=_d(r["acquired_date"]), holding_start_date=_d(r["holding_start_date"]),
                      quantity_original=r["quantity_original"], quantity_open=r["quantity_open"],
                      cost_per_share=r["cost_per_share"], basis_adjustment=r["basis_adjustment"], source=r["source"],
                      account_type=accts[r["account_id"]].account_type, entity_id=entity_id,
                      is_closed=bool(r["is_closed"]), notes=r["notes"])
            lots_by_id[lot.id] = lot
            book.lots.append(lot)
        # closures (needed for retroactive wash + reporting)
        crow = self.db.fetchall(
            f"SELECT c.* FROM lot_closures c JOIN lots l ON l.id = c.lot_id WHERE l.account_id IN ({ph}) "
            f"ORDER BY c.sale_date, c.id", list(accts))
        used: dict[int, float] = {}
        for r in crow:
            lot = lots_by_id.get(r["lot_id"])
            if lot is None:
                continue
            c = Closure(lot=lot, sale_date=_d(r["sale_date"]), quantity=r["quantity"], proceeds=r["proceeds"],
                        cost_basis=r["cost_basis"], term=r["term"], wash_disallowed=r["wash_disallowed"],
                        wash_replacement_lot=lots_by_id.get(r["wash_replacement_lot_id"]),
                        wash_explanation=r["wash_explanation"] or "", id=r["id"], sell_tx_id=r["sell_tx_id"])
            if c.wash_disallowed and c.realized_gain < 0:
                c.wash_matched_quantity = c.quantity * min(c.wash_disallowed / -c.realized_gain, 1.0)
                if c.wash_replacement_lot is not None:
                    used[c.wash_replacement_lot.id] = used.get(c.wash_replacement_lot.id, 0.0) + c.wash_matched_quantity
            book.closures.append(c)
        for lid, q in used.items():
            lots_by_id[lid].extra["used_as_replacement"] = q
        # scheduled events
        srows = self.db.fetchall(
            f"SELECT * FROM scheduled_events WHERE is_active = 1 AND account_id IN ({ph})", list(accts))
        for r in srows:
            book.scheduled.append(Acquisition(
                assetid=r["assetid"], symbol=r["symbol"], account_id=r["account_id"],
                account_type=accts[r["account_id"]].account_type, acquired_date=_d(r["event_date"]),
                quantity=r["quantity"] or 0.0, kind=f"scheduled_{(r['event_type'] or 'buy').lower()}"))
        return book

    # ------------------------------------------------------------------ persistence helpers
    def _persist_lot(self, lot: Lot, open_tx_id: int | None = None) -> None:
        if lot.id is None or lot.id < 0:
            lot.id = self.db.insert(
                "lots", account_id=lot.account_id, assetid=lot.assetid, symbol=lot.symbol,
                acquired_date=lot.acquired_date, holding_start_date=lot.holding_start_date,
                quantity_original=lot.quantity_original, quantity_open=lot.quantity_open,
                cost_per_share=lot.cost_per_share, basis_adjustment=lot.basis_adjustment, source=lot.source,
                open_tx_id=open_tx_id, is_closed=int(lot.is_closed), notes=lot.notes)
        else:
            self.db.update("lots", "id = ?", (lot.id,), holding_start_date=lot.holding_start_date,
                           quantity_open=lot.quantity_open, basis_adjustment=lot.basis_adjustment,
                           is_closed=int(lot.is_closed))

    def _persist_closure(self, c: Closure) -> None:
        repl_id = c.wash_replacement_lot.id if c.wash_replacement_lot is not None else None
        if c.id is None:
            c.id = self.db.insert(
                "lot_closures", lot_id=c.lot.id, sell_tx_id=c.sell_tx_id, sale_date=c.sale_date, quantity=c.quantity,
                proceeds=c.proceeds, cost_basis=c.cost_basis, realized_gain=c.realized_gain, term=c.term,
                wash_disallowed=c.wash_disallowed, wash_replacement_lot_id=repl_id, wash_explanation=c.wash_explanation)
        else:
            self.db.update("lot_closures", "id = ?", (c.id,), wash_disallowed=c.wash_disallowed,
                           wash_replacement_lot_id=repl_id, wash_explanation=c.wash_explanation)

    # ------------------------------------------------------------------ mutations
    def record_purchase(self, account_id: int, symbol: str, assetid: int, trade_date: date, quantity: float,
                        price: float, fees: float = 0.0, source: str = "buy", notes: str | None = None,
                        groups: SubstantiallyIdentical | None = None) -> Lot:
        acct = self.entities.account(account_id)
        book = self.load_book(acct.entity_id, groups)
        with self.db.transaction():
            tx_id = self.db.insert("transactions", account_id=account_id, assetid=assetid, symbol=symbol,
                                   trade_date=trade_date, tx_type="DRIP" if source == "drip" else "BUY",
                                   quantity=quantity, price=price, fees=fees, notes=notes)
            lot, touched = book.record_purchase(account_id, assetid, symbol, trade_date, quantity, price, fees,
                                                account_type=acct.account_type, source=source, entity_id=acct.entity_id)
            self._persist_lot(lot, open_tx_id=tx_id)
            for c in touched:
                self._persist_closure(c)
                self._persist_lot(c.lot)
            self.db.audit("user", "tx.buy", symbol, account_id=account_id, qty=quantity, price=price,
                          wash_touched=[c.id for c in touched])
        return lot

    def record_sale(self, account_id: int, symbol: str, assetid: int, sale_date: date, quantity: float,
                    price: float, fees: float = 0.0, method: LotMethod = LotMethod.HIFO,
                    specific_ids: list[int] | None = None, notes: str | None = None,
                    groups: SubstantiallyIdentical | None = None, source: str = "manual") -> list[Closure]:
        acct = self.entities.account(account_id)
        book = self.load_book(acct.entity_id, groups)
        with self.db.transaction():
            tx_id = self.db.insert("transactions", account_id=account_id, assetid=assetid, symbol=symbol,
                                   trade_date=sale_date, tx_type="SELL", quantity=quantity, price=price, fees=fees,
                                   notes=notes, source=source)
            closures = book.record_sale(account_id, assetid, sale_date, quantity, price, fees, method, specific_ids)
            for c in closures:
                c.sell_tx_id = tx_id
                self._persist_closure(c)
                self._persist_lot(c.lot)
                if c.wash_replacement_lot is not None:
                    self._persist_lot(c.wash_replacement_lot)
            self.db.audit("user", "tx.sell", symbol, account_id=account_id, qty=quantity, price=price, method=method.value,
                          wash=[(c.id, c.wash_disallowed) for c in closures if c.wash_disallowed])
        return closures

    def add_scheduled_event(self, account_id: int, symbol: str, assetid: int, event_date: date, event_type: str,
                            quantity: float | None, est_value: float | None = None, notes: str | None = None) -> int:
        sid = self.db.insert("scheduled_events", account_id=account_id, assetid=assetid, symbol=symbol,
                             event_date=event_date, event_type=event_type.upper(), quantity=quantity,
                             est_value=est_value, notes=notes)
        self.db.audit("user", "scheduled.add", symbol, account_id=account_id, date=str(event_date), type=event_type)
        return sid

    def scheduled_events(self, entity_id: int) -> pd.DataFrame:
        rows = self.db.fetchall(
            "SELECT s.*, a.name AS account_name FROM scheduled_events s JOIN accounts a ON a.id = s.account_id "
            "WHERE s.is_active = 1 AND a.entity_id = ? ORDER BY s.event_date", (entity_id,))
        return pd.DataFrame(rows_to_dicts(rows))

    # ------------------------------------------------------------------ views
    def lots_frame(self, entity_id: int, open_only: bool = True) -> pd.DataFrame:
        rows = self.db.fetchall(
            "SELECT l.*, a.name AS account_name, a.account_type FROM lots l JOIN accounts a ON a.id = l.account_id "
            "WHERE a.entity_id = ?" + (" AND l.is_closed = 0" if open_only else "") + " ORDER BY l.symbol, l.holding_start_date",
            (entity_id,))
        return pd.DataFrame(rows_to_dicts(rows))

    def transactions_frame(self, entity_id: int) -> pd.DataFrame:
        rows = self.db.fetchall(
            "SELECT t.*, a.name AS account_name FROM transactions t JOIN accounts a ON a.id = t.account_id "
            "WHERE a.entity_id = ? ORDER BY t.trade_date DESC, t.id DESC", (entity_id,))
        return pd.DataFrame(rows_to_dicts(rows))

    def closures_frame(self, entity_id: int) -> pd.DataFrame:
        rows = self.db.fetchall(
            "SELECT c.*, l.symbol, l.account_id, a.name AS account_name FROM lot_closures c "
            "JOIN lots l ON l.id = c.lot_id JOIN accounts a ON a.id = l.account_id WHERE a.entity_id = ? "
            "ORDER BY c.sale_date DESC, c.id DESC", (entity_id,))
        return pd.DataFrame(rows_to_dicts(rows))

    def held_symbols(self, entity_id: int) -> list[str]:
        rows = self.db.fetchall(
            "SELECT DISTINCT l.symbol FROM lots l JOIN accounts a ON a.id = l.account_id "
            "WHERE a.entity_id = ? AND l.is_closed = 0", (entity_id,))
        return sorted(r["symbol"] for r in rows)


# ============================================================================ runs / models / AI
class RunRepo:
    def __init__(self, db: Database):
        self.db = db

    def create(self, run_type: str, as_of: date, entity_id: int | None, snapshot_id: str | None,
               model_version_id: int | None, params: dict, summary: dict, artifact_path: str | None = None,
               notes: str | None = None) -> int:
        rid = self.db.insert("runs", run_type=run_type, as_of_date=as_of, entity_id=entity_id, snapshot_id=snapshot_id,
                             model_version_id=model_version_id, params=json.dumps(params, default=str),
                             summary=json.dumps(summary, default=str), artifact_path=artifact_path, notes=notes)
        self.db.audit("system", "run.create", str(rid), run_type=run_type)
        return rid

    def add_trades(self, run_id: int, trades: list[dict]) -> None:
        with self.db.transaction():
            for t in trades:
                self.db.insert("run_trades", run_id=run_id, **t)

    def list(self, limit: int = 100) -> pd.DataFrame:
        rows = self.db.fetchall("SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ?", (limit,))
        df = pd.DataFrame(rows_to_dicts(rows))
        if not df.empty:
            df["params"] = df["params"].apply(json.loads)
            df["summary"] = df["summary"].apply(json.loads)
        return df

    def get(self, run_id: int) -> dict | None:
        r = self.db.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not r:
            return None
        d = dict(r)
        d["params"] = json.loads(d["params"])
        d["summary"] = json.loads(d["summary"])
        return d

    def trades(self, run_id: int) -> pd.DataFrame:
        rows = self.db.fetchall(
            "SELECT t.*, a.name AS account_name FROM run_trades t JOIN accounts a ON a.id = t.account_id "
            "WHERE run_id = ? ORDER BY side DESC, est_value DESC", (run_id,))
        return pd.DataFrame(rows_to_dicts(rows))

    def mark_acted(self, trade_ids: list[int], acted: bool = True) -> None:
        with self.db.transaction():
            for tid in trade_ids:
                self.db.update("run_trades", "id = ?", (tid,), acted_on=int(acted),
                               acted_at=pd.Timestamp.now().isoformat() if acted else None)
        self.db.audit("user", "run.mark_acted", None, ids=trade_ids, acted=acted)


class ModelRepo:
    def __init__(self, db: Database):
        self.db = db

    def create(self, name: str, snapshot_id: str | None, as_of: date, universe_name: str, lookback_days: int,
               factor_list: list[str], diagnostics: dict, artifact_path: str, code_version_id: int | None = None,
               notes: str | None = None, make_active: bool = True) -> int:
        with self.db.transaction():
            if make_active:
                self.db.execute("UPDATE model_versions SET is_active = 0")
            mid = self.db.insert("model_versions", name=name, snapshot_id=snapshot_id, as_of_date=as_of,
                                 universe_name=universe_name, lookback_days=lookback_days,
                                 factor_list=json.dumps(factor_list), diagnostics=json.dumps(diagnostics, default=str),
                                 artifact_path=artifact_path, code_version_id=code_version_id, is_active=int(make_active),
                                 notes=notes)
            self.db.audit("system", "model.create", str(mid), name=name)
        return mid

    def list(self) -> pd.DataFrame:
        rows = self.db.fetchall("SELECT * FROM model_versions ORDER BY created_at DESC, id DESC")
        df = pd.DataFrame(rows_to_dicts(rows))
        if not df.empty:
            df["factor_list"] = df["factor_list"].apply(json.loads)
            df["diagnostics"] = df["diagnostics"].apply(json.loads)
        return df

    def get(self, model_id: int) -> dict | None:
        r = self.db.fetchone("SELECT * FROM model_versions WHERE id = ?", (model_id,))
        if not r:
            return None
        d = dict(r)
        d["factor_list"] = json.loads(d["factor_list"])
        d["diagnostics"] = json.loads(d["diagnostics"])
        return d

    def active(self) -> dict | None:
        r = self.db.fetchone("SELECT id FROM model_versions WHERE is_active = 1 ORDER BY id DESC LIMIT 1")
        return self.get(int(r["id"])) if r else None

    def set_active(self, model_id: int) -> None:
        with self.db.transaction():
            self.db.execute("UPDATE model_versions SET is_active = 0")
            self.db.update("model_versions", "id = ?", (model_id,), is_active=1)
        self.db.audit("user", "model.activate", str(model_id))


class CodeRepo:
    """Versioned text of AI-editable modules + the AI change queue."""

    def __init__(self, db: Database):
        self.db = db

    def latest_version(self, module_path: str) -> dict | None:
        r = self.db.fetchone("SELECT * FROM code_versions WHERE module_path = ? ORDER BY version_no DESC LIMIT 1",
                             (module_path,))
        return dict(r) if r else None

    def versions(self, module_path: str | None = None) -> pd.DataFrame:
        if module_path:
            rows = self.db.fetchall("SELECT id, module_path, version_no, created_at, source, parent_version_id, change_id, "
                                    "is_active, length(code_text) AS n_chars FROM code_versions WHERE module_path = ? "
                                    "ORDER BY version_no DESC", (module_path,))
        else:
            rows = self.db.fetchall("SELECT id, module_path, version_no, created_at, source, parent_version_id, change_id, "
                                    "is_active, length(code_text) AS n_chars FROM code_versions ORDER BY module_path, version_no DESC")
        return pd.DataFrame(rows_to_dicts(rows))

    def get_version(self, version_id: int) -> dict | None:
        r = self.db.fetchone("SELECT * FROM code_versions WHERE id = ?", (version_id,))
        return dict(r) if r else None

    def add_version(self, module_path: str, code_text: str, source: str, change_id: int | None = None,
                    make_active: bool = True) -> int:
        with self.db.transaction():
            prev = self.latest_version(module_path)
            vno = (prev["version_no"] + 1) if prev else 1
            if make_active:
                self.db.execute("UPDATE code_versions SET is_active = 0 WHERE module_path = ?", (module_path,))
            vid = self.db.insert("code_versions", module_path=module_path, version_no=vno, code_text=code_text,
                                 source=source, parent_version_id=prev["id"] if prev else None, change_id=change_id,
                                 is_active=int(make_active))
            self.db.audit("system" if source != "ai" else "ai", "code.version", module_path, version=vno, source=source)
        return vid

    # change queue -------------------------------------------------------------------------
    def create_change(self, module_path: str, title: str, proposed_code: str, rationale: str | None,
                      diff_text: str | None, conversation_id: int | None, prompt_excerpt: str | None = None) -> int:
        cid = self.db.insert("ai_changes", module_path=module_path, title=title, proposed_code=proposed_code,
                             rationale=rationale, diff_text=diff_text, conversation_id=conversation_id,
                             prompt_excerpt=prompt_excerpt, status="drafted")
        self.db.audit("ai", "change.draft", str(cid), module=module_path, title=title)
        return cid

    def set_sandbox_result(self, change_id: int, stdout: str, passed: bool) -> None:
        self.db.update("ai_changes", "id = ?", (change_id,), sandbox_stdout=stdout[-200_000:], sandbox_passed=int(passed),
                       sandbox_ran_at=pd.Timestamp.now().isoformat(), status="tested")
        self.db.audit("system", "change.sandbox", str(change_id), passed=passed)

    def set_status(self, change_id: int, status: str, approved_by: str | None = None,
                   promoted_version_id: int | None = None) -> None:
        cols: dict[str, Any] = {"status": status}
        if approved_by:
            cols["approved_by"] = approved_by
            cols["approved_at"] = pd.Timestamp.now().isoformat()
        if promoted_version_id is not None:
            cols["promoted_version_id"] = promoted_version_id
        self.db.update("ai_changes", "id = ?", (change_id,), **cols)
        self.db.audit("user" if status in {"approved", "rejected"} else "system", f"change.{status}", str(change_id))

    def change(self, change_id: int) -> dict | None:
        r = self.db.fetchone("SELECT * FROM ai_changes WHERE id = ?", (change_id,))
        return dict(r) if r else None

    def changes(self, status: str | None = None, limit: int = 200) -> pd.DataFrame:
        if status:
            rows = self.db.fetchall("SELECT * FROM ai_changes WHERE status = ? ORDER BY id DESC LIMIT ?", (status, limit))
        else:
            rows = self.db.fetchall("SELECT * FROM ai_changes ORDER BY id DESC LIMIT ?", (limit,))
        return pd.DataFrame(rows_to_dicts(rows))


class ConversationRepo:
    def __init__(self, db: Database):
        self.db = db

    def create(self, model: str, title: str | None = None) -> int:
        return self.db.insert("ai_conversations", model=model, title=title)

    def add_message(self, conversation_id: int, role: str, content: Any, usage: dict | None = None) -> int:
        return self.db.insert("ai_messages", conversation_id=conversation_id, role=role,
                              content=json.dumps(content, default=str), usage=json.dumps(usage, default=str) if usage else None)

    def messages(self, conversation_id: int) -> list[dict]:
        rows = self.db.fetchall("SELECT * FROM ai_messages WHERE conversation_id = ? ORDER BY id", (conversation_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["content"] = json.loads(d["content"])
            d["usage"] = json.loads(d["usage"]) if d["usage"] else None
            out.append(d)
        return out

    def list(self, limit: int = 50) -> pd.DataFrame:
        rows = self.db.fetchall("SELECT c.*, (SELECT COUNT(*) FROM ai_messages m WHERE m.conversation_id = c.id) AS n "
                                "FROM ai_conversations c ORDER BY c.id DESC LIMIT ?", (limit,))
        return pd.DataFrame(rows_to_dicts(rows))


class BasketRepo:
    """Named model portfolios (symbol -> weight)."""

    def __init__(self, db: Database):
        self.db = db

    def list(self) -> pd.DataFrame:
        rows = self.db.fetchall(
            "SELECT b.*, (SELECT COUNT(*) FROM basket_members m WHERE m.basket_id = b.id) AS n_names FROM baskets b ORDER BY b.updated_at DESC")
        df = pd.DataFrame(rows_to_dicts(rows))
        if not df.empty:
            df["params"] = df["params"].apply(lambda s: json.loads(s) if s else {})
            df["metrics"] = df["metrics"].apply(lambda s: json.loads(s) if s else {})
        return df

    def get(self, name: str) -> dict | None:
        r = self.db.fetchone("SELECT * FROM baskets WHERE name = ?", (name,))
        if not r:
            return None
        d = dict(r)
        d["params"] = json.loads(d["params"]) if d["params"] else {}
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
        d["weights"] = self.weights(int(d["id"]))
        return d

    def weights(self, basket_id: int) -> pd.Series:
        rows = self.db.fetchall("SELECT symbol, weight FROM basket_members WHERE basket_id = ? ORDER BY weight DESC", (basket_id,))
        return pd.Series({r["symbol"]: float(r["weight"]) for r in rows}, dtype=float)

    def save(self, name: str, weights: pd.Series, description: str | None = None, source: str = "manual",
             benchmark_name: str | None = None, params: dict | None = None, metrics: dict | None = None,
             resolve=None) -> int:
        w = weights[weights > 0].astype(float)
        if w.empty:
            raise ValueError("basket has no positive weights")
        w = w / w.sum()
        with self.db.transaction():
            r = self.db.fetchone("SELECT id FROM baskets WHERE name = ?", (name,))
            if r:
                bid = int(r["id"])
                self.db.update("baskets", "id = ?", (bid,), description=description, source=source, benchmark_name=benchmark_name,
                               params=json.dumps(params or {}, default=str), metrics=json.dumps(metrics or {}, default=str),
                               updated_at=pd.Timestamp.now().isoformat())
                self.db.execute("DELETE FROM basket_members WHERE basket_id = ?", (bid,))
            else:
                bid = self.db.insert("baskets", name=name, description=description, source=source, benchmark_name=benchmark_name,
                                     params=json.dumps(params or {}, default=str), metrics=json.dumps(metrics or {}, default=str))
            for sym, wt in w.items():
                aid = resolve(sym) if resolve else None
                self.db.insert("basket_members", basket_id=bid, symbol=str(sym), assetid=aid, weight=float(wt))
            self.db.audit("ai" if source == "ai" else "user", "basket.save", name, n=int(len(w)), source=source)
        return bid

    def delete(self, name: str) -> None:
        r = self.db.fetchone("SELECT id FROM baskets WHERE name = ?", (name,))
        if r:
            with self.db.transaction():
                self.db.execute("DELETE FROM basket_members WHERE basket_id = ?", (r["id"],))
                self.db.execute("DELETE FROM baskets WHERE id = ?", (r["id"],))
            self.db.audit("user", "basket.delete", name)


class AgentRepo:
    """Scheduled / ad-hoc unattended co-pilot tasks and their runs."""

    def __init__(self, db: Database):
        self.db = db

    # tasks ----------------------------------------------------------------------------------
    def tasks(self, enabled_only: bool = False) -> pd.DataFrame:
        sql = "SELECT * FROM agent_tasks" + (" WHERE enabled = 1" if enabled_only else "") + " ORDER BY name"
        return pd.DataFrame(rows_to_dicts(self.db.fetchall(sql)))

    def task(self, task_id: int) -> dict | None:
        r = self.db.fetchone("SELECT * FROM agent_tasks WHERE id = ?", (task_id,))
        return dict(r) if r else None

    def task_by_name(self, name: str) -> dict | None:
        r = self.db.fetchone("SELECT * FROM agent_tasks WHERE name = ?", (name,))
        return dict(r) if r else None

    def upsert_task(self, name: str, prompt: str, schedule: str = "manual", enabled: bool = True, notify: bool = True,
                    effort: str | None = None, next_run_at: str | None = None) -> int:
        r = self.db.fetchone("SELECT id FROM agent_tasks WHERE name = ?", (name,))
        if r:
            self.db.update("agent_tasks", "id = ?", (r["id"],), prompt=prompt, schedule=schedule, enabled=int(enabled), notify=int(notify),
                           effort=effort, next_run_at=next_run_at, updated_at=pd.Timestamp.now().isoformat())
            tid = int(r["id"])
        else:
            tid = self.db.insert("agent_tasks", name=name, prompt=prompt, schedule=schedule, enabled=int(enabled), notify=int(notify),
                                 effort=effort, next_run_at=next_run_at)
        self.db.audit("user", "agent_task.save", name, schedule=schedule, enabled=enabled)
        return tid

    def set_task(self, task_id: int, **cols: Any) -> None:
        cols["updated_at"] = pd.Timestamp.now().isoformat()
        self.db.update("agent_tasks", "id = ?", (task_id,), **cols)

    def delete_task(self, task_id: int) -> None:
        with self.db.transaction():
            self.db.execute("UPDATE agent_runs SET task_id = NULL WHERE task_id = ?", (task_id,))
            self.db.execute("DELETE FROM agent_tasks WHERE id = ?", (task_id,))
        self.db.audit("user", "agent_task.delete", str(task_id))

    # runs -----------------------------------------------------------------------------------
    def start_run(self, task_id: int | None, name: str, prompt: str, trigger: str, conversation_id: int | None) -> int:
        return self.db.insert("agent_runs", task_id=task_id, name=name, prompt=prompt, trigger=trigger, conversation_id=conversation_id)

    def finish_run(self, run_id: int, status: str, report: str | None = None, error: str | None = None,
                   change_ids: list[int] | None = None, tool_calls: int | None = None, cost_usd: float | None = None,
                   duration_s: float | None = None) -> None:
        self.db.update("agent_runs", "id = ?", (run_id,), status=status, report=report, error=error,
                       change_ids=json.dumps(change_ids or []), tool_calls=tool_calls, cost_usd=cost_usd, duration_s=duration_s,
                       finished_at=pd.Timestamp.now().isoformat())

    def runs(self, limit: int = 100, task_id: int | None = None) -> pd.DataFrame:
        if task_id is not None:
            rows = self.db.fetchall("SELECT * FROM agent_runs WHERE task_id = ? ORDER BY id DESC LIMIT ?", (task_id, limit))
        else:
            rows = self.db.fetchall("SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (limit,))
        df = pd.DataFrame(rows_to_dicts(rows))
        if not df.empty:
            df["change_ids"] = df["change_ids"].apply(lambda s: json.loads(s) if s else [])
        return df

    def run(self, run_id: int) -> dict | None:
        r = self.db.fetchone("SELECT * FROM agent_runs WHERE id = ?", (run_id,))
        if not r:
            return None
        d = dict(r)
        d["change_ids"] = json.loads(d["change_ids"]) if d["change_ids"] else []
        return d

    def unread_count(self) -> int:
        r = self.db.fetchone("SELECT COUNT(*) AS n FROM agent_runs WHERE is_read = 0 AND status IN ('done','failed')")
        return int(r["n"]) if r else 0

    def mark_read(self, run_ids: list[int] | None = None) -> None:
        if run_ids:
            for rid in run_ids:
                self.db.update("agent_runs", "id = ?", (rid,), is_read=1)
        else:
            self.db.execute("UPDATE agent_runs SET is_read = 1")


class PipelineRepo:
    def __init__(self, db: Database):
        self.db = db

    def list(self) -> pd.DataFrame:
        return pd.DataFrame(rows_to_dicts(self.db.fetchall("SELECT id, name, description, source, created_at, updated_at FROM pipelines ORDER BY updated_at DESC")))

    def get(self, name: str) -> dict | None:
        r = self.db.fetchone("SELECT * FROM pipelines WHERE name = ?", (name,))
        return dict(r) if r else None

    def save(self, name: str, spec_json: str, description: str | None = None, source: str = "builder") -> int:
        r = self.db.fetchone("SELECT id FROM pipelines WHERE name = ?", (name,))
        if r:
            self.db.update("pipelines", "id = ?", (r["id"],), spec=spec_json, description=description, source=source,
                           updated_at=pd.Timestamp.now().isoformat())
            pid = int(r["id"])
        else:
            pid = self.db.insert("pipelines", name=name, spec=spec_json, description=description, source=source)
        self.db.audit("ai" if source == "ai" else "user", "pipeline.save", name)
        return pid

    def delete(self, name: str) -> None:
        self.db.execute("DELETE FROM pipelines WHERE name = ?", (name,))
        self.db.audit("user", "pipeline.delete", name)
