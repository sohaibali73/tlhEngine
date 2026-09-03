"""Holdings import from broker CSV / Excel exports.

Any file with a symbol, a quantity and (ideally) a cost and an acquisition date can be imported. Column names are
auto-mapped from a dictionary of broker aliases (Schwab, Fidelity, IBKR, TradeStation, Vanguard, generic). Each row
becomes a purchase recorded through the normal ledger path, so wash-sale groups and holding periods are computed the
same way as for hand-entered lots. Rows with missing dates or costs are imported with documented defaults and flagged.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .context import AppContext

log = logging.getLogger(__name__)

ALIASES: dict[str, list[str]] = {
    "symbol": ["symbol", "ticker", "security", "sym", "instrument", "stock symbol"],
    "quantity": ["quantity", "qty", "shares", "units", "quantity held", "position"],
    "cost_per_share": ["cost/share", "cost per share", "unit cost", "average cost", "avg cost", "cost basis per share",
                       "average cost basis", "purchase price", "price paid", "cost price", "open price", "costprice"],
    "cost_basis": ["cost basis", "cost basis total", "total cost", "basis", "costbasis", "book cost", "cost"],
    "acquired": ["date acquired", "acquired", "acquisition date", "purchase date", "open date", "opendate", "trade date",
                 "dateacquired", "date opened", "holding date", "date"],
    "account": ["account", "account name", "account number", "acct", "account #", "account_id"],
    "price": ["price", "last price", "current price", "mark", "market price", "last"],
}
DEFAULT_LOOKBACK_DAYS = 400          # missing acquisition date: assume long-term, flagged


@dataclass
class ImportPlan:
    frame: pd.DataFrame                      # normalised rows: account, symbol, quantity, cost_per_share, acquired, flags
    mapping: dict[str, str]                  # canonical -> source column
    warnings: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9/#]+", " ", str(s).strip().lower()).strip()


def guess_mapping(columns: list[str]) -> dict[str, str]:
    cols = {_norm(c): c for c in columns}
    out: dict[str, str] = {}
    for canon, names in ALIASES.items():
        for n in names:
            if n in cols and cols[n] not in out.values():
                out[canon] = cols[n]
                break
        if canon not in out:                       # fuzzy: alias contained in column name
            for nc, orig in cols.items():
                if orig in out.values():
                    continue
                if any(n in nc for n in names) and canon != "price" or (canon == "price" and nc in ("price", "last price")):
                    out[canon] = orig
                    break
    return out


def read_file(path: str | Path) -> pd.DataFrame:
    """Read a broker export. Title blocks and blank lines above the header are skipped: the header is the first line
    with at least three delimited cells (delimiter chosen by counting , ; tab | on that line)."""
    import io

    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(p, header=None)
        hdr = 0
        for i in range(min(len(raw), 30)):
            if raw.iloc[i].notna().sum() >= 3:
                hdr = i
                break
        df = raw.iloc[hdr + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    else:
        text = p.read_text(encoding="utf-8-sig", errors="replace")
        lines = text.splitlines()
        hdr, delim = 0, ","
        for i, line in enumerate(lines[:40]):
            counts = {d: line.count(d) for d in (",", "\t", ";", "|")}
            d, c = max(counts.items(), key=lambda kv: kv[1])
            if c >= 2:
                hdr, delim = i, d
                break
        body = "\n".join(lines[hdr:])
        df = pd.read_csv(io.StringIO(body), sep=delim, engine="python", on_bad_lines="skip", dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    df = df.loc[:, [c for c in df.columns if c and c.lower() != "nan" and not c.startswith("Unnamed")]]
    return df.reset_index(drop=True)


def _num(v) -> float | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).replace("$", "").replace(",", "").replace("%", "").strip()
    if s in ("", "--", "n/a", "N/A", "-"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def _date(v) -> date | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).strip()
    if not s or s.lower() in ("--", "n/a", "various", "multiple"):
        return None
    try:
        return pd.Timestamp(s).date()
    except (ValueError, TypeError):
        return None


def plan_import(path: str | Path, mapping: dict[str, str] | None = None, default_account: str = "Imported account") -> ImportPlan:
    df = read_file(path)
    mapping = mapping or guess_mapping(list(df.columns))
    warnings: list[str] = []
    if "symbol" not in mapping or "quantity" not in mapping:
        raise ValueError(f"could not find symbol/quantity columns in {list(df.columns)[:12]}; map them manually")
    rows = []
    today = date.today()
    for _, r in df.iterrows():
        sym = str(r[mapping["symbol"]]).strip().upper()
        if not sym or sym in ("NAN", "CASH", "TOTAL", "ACCOUNT TOTAL", "PENDING ACTIVITY") or sym.startswith("CASH"):
            continue
        sym = sym.replace(" ", "")
        qty = _num(r[mapping["quantity"]])
        if qty is None or qty <= 0:
            continue
        flags = []
        cps = _num(r[mapping["cost_per_share"]]) if "cost_per_share" in mapping else None
        if cps is None and "cost_basis" in mapping:
            cb = _num(r[mapping["cost_basis"]])
            cps = cb / qty if cb else None
        if cps is None and "price" in mapping:
            cps = _num(r[mapping["price"]])
            if cps is not None:
                flags.append("cost missing: current price used (zero unrealised P&L)")
        if cps is None:
            flags.append("cost missing: filled from snapshot price at import")
        acq = _date(r[mapping["acquired"]]) if "acquired" in mapping else None
        if acq is None:
            acq = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
            flags.append(f"acquisition date missing: assumed {DEFAULT_LOOKBACK_DAYS} days ago (long-term)")
        acct = str(r[mapping["account"]]).strip() if "account" in mapping and str(r[mapping["account"]]).strip() not in ("", "nan") else default_account
        rows.append({"account": acct, "symbol": sym, "quantity": float(qty), "cost_per_share": cps, "acquired": acq, "flags": "; ".join(flags)})
    frame = pd.DataFrame(rows, columns=["account", "symbol", "quantity", "cost_per_share", "acquired", "flags"])
    if frame.empty:
        warnings.append("no holdings rows recognised")
    n_flag = int((frame["flags"] != "").sum()) if not frame.empty else 0
    if n_flag:
        warnings.append(f"{n_flag} row(s) needed a default (see flags column)")
    return ImportPlan(frame=frame, mapping=mapping, warnings=warnings, source=str(path))


class ImportService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    def execute(self, entity_id: int, plan: ImportPlan, account_type: str = "taxable", broker: str | None = None,
                progress=None) -> dict:
        """Record every planned row as a purchase. Creates accounts named in the file under `entity_id`."""
        from .data_service import DataService
        from .portfolio_service import PortfolioService

        say = progress or (lambda m: None)
        ps = PortfolioService(self.ctx)
        snap = DataService(self.ctx).latest_snapshot()
        px = snap.last_prices() if snap is not None else pd.Series(dtype=float)
        accounts: dict[str, int] = {}
        done, skipped = 0, []
        for i, r in plan.frame.iterrows():
            name = r["account"]
            if name not in accounts:
                accounts[name] = self.ctx.entities.get_or_create_account(entity_id, name, account_type, broker=broker or "import")
            cps = r["cost_per_share"]
            if cps is None or cps != cps:
                cps = float(px.get(r["symbol"], float("nan")))
                if cps != cps:
                    skipped.append(f"{r['symbol']}: no cost and no price available")
                    continue
            try:
                ps.buy(accounts[name], r["symbol"], r["acquired"], float(r["quantity"]), float(cps), notes=f"import:{Path(plan.source).name}", source="import")
                done += 1
            except Exception as e:  # unknown symbol etc.
                skipped.append(f"{r['symbol']}: {e}")
            if (i + 1) % 10 == 0:
                say(f"Imported {done} of {plan.n_rows} lots…")
        self.ctx.db.audit("user", "holdings.import", str(entity_id), rows=done, source=plan.source)
        return {"imported": done, "skipped": skipped, "accounts": list(accounts), "warnings": plan.warnings}


def template_csv() -> str:
    return ("Account,Symbol,Quantity,Cost Per Share,Date Acquired\n"
            "Brokerage,AAPL,100,148.25,2023-03-15\n"
            "Brokerage,MSFT,40,310.10,2024-01-08\n"
            "IRA,VTI,120,205.00,2022-06-01\n")
