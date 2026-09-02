"""Dialogs: record trade, scheduled event, add account/entity, lot import."""
from __future__ import annotations

from datetime import date

import pandas as pd
from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from ..tax.lots import LotMethod


def _date_edit(d: date | None = None) -> QDateEdit:
    e = QDateEdit()
    e.setCalendarPopup(True)
    e.setDisplayFormat("yyyy-MM-dd")
    d = d or date.today()
    e.setDate(QDate(d.year, d.month, d.day))
    return e


def _to_date(e: QDateEdit) -> date:
    q = e.date()
    return date(q.year(), q.month(), q.day())


class TradeDialog(QDialog):
    def __init__(self, accounts, side: str = "BUY", symbol: str = "", lots: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Record {side.title()}")
        self.side = side
        form = QFormLayout(self)
        self.account = QComboBox()
        for a in accounts:
            self.account.addItem(f"{a.name} ({a.account_type})", a.id)
        self.symbol = QLineEdit(symbol)
        self.date = _date_edit()
        self.qty = QDoubleSpinBox()
        self.qty.setRange(0.0001, 1e9)
        self.qty.setDecimals(4)
        self.qty.setValue(100)
        self.price = QDoubleSpinBox()
        self.price.setRange(0.0001, 1e7)
        self.price.setDecimals(4)
        self.price.setPrefix("$ ")
        self.fees = QDoubleSpinBox()
        self.fees.setRange(0, 1e6)
        self.fees.setPrefix("$ ")
        self.notes = QLineEdit()
        form.addRow("Account", self.account)
        form.addRow("Symbol", self.symbol)
        form.addRow("Trade date", self.date)
        form.addRow("Quantity", self.qty)
        form.addRow("Price / share", self.price)
        form.addRow("Fees", self.fees)
        if side == "SELL":
            self.method = QComboBox()
            for m in LotMethod:
                self.method.addItem(m.value, m)
            self.method.setCurrentText("HIFO")
            form.addRow("Lot method", self.method)
            self.specific = QLineEdit()
            self.specific.setPlaceholderText("lot ids, comma-separated (SPECIFIC only)")
            form.addRow("Specific lots", self.specific)
        else:
            self.source = QComboBox()
            self.source.addItems(["buy", "drip", "transfer"])
            form.addRow("Source", self.source)
        form.addRow("Notes", self.notes)
        note = QLabel("Recording a trade updates lots, realised P&L and wash-sale state. Nothing is sent to a broker.")
        note.setProperty("muted", True)
        note.setWordWrap(True)
        form.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        out = {"account_id": self.account.currentData(), "symbol": self.symbol.text().strip().upper(),
               "trade_date": _to_date(self.date), "quantity": self.qty.value(), "price": self.price.value(),
               "fees": self.fees.value(), "notes": self.notes.text() or None}
        if self.side == "SELL":
            out["method"] = self.method.currentData()
            ids = [int(x) for x in self.specific.text().replace(" ", "").split(",") if x]
            out["specific_ids"] = ids or None
        else:
            out["source"] = self.source.currentText()
        return out


class ScheduledEventDialog(QDialog):
    def __init__(self, accounts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add scheduled purchase / DRIP")
        form = QFormLayout(self)
        self.account = QComboBox()
        for a in accounts:
            self.account.addItem(f"{a.name} ({a.account_type})", a.id)
        self.symbol = QLineEdit()
        self.date = _date_edit()
        self.kind = QComboBox()
        self.kind.addItems(["DRIP", "BUY"])
        self.qty = QDoubleSpinBox()
        self.qty.setRange(0, 1e9)
        self.qty.setDecimals(4)
        self.notes = QLineEdit()
        form.addRow("Account", self.account)
        form.addRow("Symbol", self.symbol)
        form.addRow("Event date", self.date)
        form.addRow("Type", self.kind)
        form.addRow("Quantity (est.)", self.qty)
        form.addRow("Notes", self.notes)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        return {"account_id": self.account.currentData(), "symbol": self.symbol.text().strip().upper(),
                "event_date": _to_date(self.date), "event_type": self.kind.currentText(), "quantity": self.qty.value() or None,
                "notes": self.notes.text() or None}


class AccountDialog(QDialog):
    def __init__(self, entities: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add account")
        form = QFormLayout(self)
        self.entity = QComboBox()
        for e in entities:
            self.entity.addItem(e["name"], e["id"])
        self.name = QLineEdit()
        self.type = QComboBox()
        self.type.addItems(["taxable", "ira", "roth", "401k", "other_deferred"])
        self.broker = QLineEdit()
        self.owner = QLineEdit("self")
        form.addRow("Tax entity", self.entity)
        form.addRow("Account name", self.name)
        form.addRow("Type", self.type)
        form.addRow("Broker", self.broker)
        form.addRow("Owner", self.owner)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        return {"entity_id": self.entity.currentData(), "name": self.name.text().strip(), "account_type": self.type.currentText(),
                "broker": self.broker.text() or None, "owner": self.owner.text() or None}


class EntityDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add tax entity (household)")
        form = QFormLayout(self)
        self.name = QLineEdit()
        self.filing = QComboBox()
        self.filing.addItems(["single", "mfj", "mfs", "hoh"])
        form.addRow("Name", self.name)
        form.addRow("Filing status", self.filing)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        return {"name": self.name.text().strip(), "filing_status": self.filing.currentText()}


def import_lots_csv(parent, portfolio_service, accounts) -> int:
    """CSV columns: account, symbol, date, quantity, price[, fees]. Returns number of lots created."""
    path, _ = QFileDialog.getOpenFileName(parent, "Import lots CSV", "", "CSV (*.csv)")
    if not path:
        return 0
    df = pd.read_csv(path)
    need = {"account", "symbol", "date", "quantity", "price"}
    if not need <= set(c.lower() for c in df.columns):
        QMessageBox.warning(parent, "Import", f"CSV must have columns {sorted(need)}")
        return 0
    df.columns = [c.lower() for c in df.columns]
    by_name = {a.name: a.id for a in accounts}
    n = 0
    errors = []
    for _, r in df.iterrows():
        try:
            aid = by_name[r["account"]]
            portfolio_service.buy(aid, str(r["symbol"]), pd.Timestamp(r["date"]).date(), float(r["quantity"]), float(r["price"]),
                                  float(r.get("fees", 0.0) or 0.0), notes="import")
            n += 1
        except Exception as e:
            errors.append(f"{r.get('symbol')}: {e}")
    if errors:
        QMessageBox.warning(parent, "Import", f"Imported {n}; {len(errors)} errors:\n" + "\n".join(errors[:15]))
    return n


class InfoDialog(QDialog):
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 320)
        lay = QVBoxLayout(self)
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(lbl.textInteractionFlags() | 1)
        lay.addWidget(lbl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)
