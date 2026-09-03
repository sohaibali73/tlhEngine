"""Import holdings from a broker CSV / Excel export: pick file -> auto-map columns -> preview -> import."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from ..services.import_service import ALIASES, ImportPlan, ImportService, plan_import, read_file, template_csv
from .widgets import FrameTable, button
from .workers import run_task


class ImportDialog(QDialog):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.setWindowTitle("Import holdings")
        self.resize(980, 640)
        self._plan: ImportPlan | None = None
        self._columns: list[str] = []
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("Broker export (.csv, .xlsx)")
        row.addWidget(self.path, 1)
        row.addWidget(button("Browse…", self._browse))
        row.addWidget(button("Save a CSV template", self._template))
        lay.addLayout(row)
        g = QGroupBox("Column mapping (auto-detected; adjust if needed)")
        f = QFormLayout(g)
        self.maps: dict[str, QComboBox] = {}
        for canon in ("symbol", "quantity", "cost_per_share", "cost_basis", "acquired", "account", "price"):
            cb = QComboBox()
            cb.currentIndexChanged.connect(self._remap)
            self.maps[canon] = cb
            f.addRow(canon.replace("_", " "), cb)
        lay.addWidget(g)
        ent = QHBoxLayout()
        ent.addWidget(QLabel("Tax entity"))
        self.entity = QComboBox()
        for e in self.ctx.entities.list():
            self.entity.addItem(e["name"], e["id"])
        ent.addWidget(self.entity, 1)
        ent.addWidget(QLabel("Default account name"))
        self.acct = QLineEdit("Imported brokerage")
        ent.addWidget(self.acct, 1)
        ent.addWidget(QLabel("Account type"))
        self.acct_type = QComboBox()
        self.acct_type.addItems(["taxable", "ira", "roth", "401k", "other_deferred"])
        ent.addWidget(self.acct_type)
        lay.addLayout(ent)
        self.preview = FrameTable(["account", "symbol", "quantity", "cost_per_share", "acquired", "flags"], filter_box=False)
        lay.addWidget(self.preview, 1)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setProperty("muted", True)
        lay.addWidget(self.info)
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.ok_btn = button("Import these lots", self._go, primary=True)
        bb.addButton(self.ok_btn, QDialogButtonBox.AcceptRole)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if self.entity.count() == 0:
            self.entity.addItem("New household", None)

    def _browse(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Broker export", "", "Holdings (*.csv *.xlsx *.xls *.txt);;All files (*)")
        if p:
            self.path.setText(p)
            self._load(p)

    def _template(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Save template", "holdings_template.csv", "CSV (*.csv)")
        if p:
            Path(p).write_text(template_csv(), encoding="utf-8")
            self.info.setText(f"Template saved to {p}. Fill it in and import.")

    def _load(self, p: str) -> None:
        try:
            df = read_file(p)
        except Exception as e:
            QMessageBox.critical(self, "Cannot read file", str(e))
            return
        self._columns = list(df.columns)
        from ..services.import_service import guess_mapping
        guess = guess_mapping(self._columns)
        for canon, cb in self.maps.items():
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("— none —", "")
            for c in self._columns:
                cb.addItem(c, c)
            if canon in guess:
                cb.setCurrentIndex(cb.findData(guess[canon]))
            cb.blockSignals(False)
        self._remap()

    def _mapping(self) -> dict[str, str]:
        return {k: cb.currentData() for k, cb in self.maps.items() if cb.currentData()}

    def _remap(self) -> None:
        if not self.path.text() or not self._columns:
            return
        try:
            self._plan = plan_import(self.path.text(), self._mapping(), self.acct.text().strip() or "Imported brokerage")
        except Exception as e:
            self.info.setText(str(e))
            self._plan = None
            return
        self.preview.set_frame(self._plan.frame)
        w = "; ".join(self._plan.warnings)
        self.info.setText(f"{self._plan.n_rows} lots recognised" + (f" · {w}" if w else "") + ". Aliases understood: "
                          + ", ".join(sorted({a for v in ALIASES.values() for a in v[:2]})))

    def _go(self) -> None:
        if self._plan is None or self._plan.frame.empty:
            QMessageBox.information(self, "Nothing to import", "Pick a file and check the column mapping.")
            return
        eid = self.entity.currentData()
        if eid is None:
            eid = self.ctx.entities.get_or_create("My Household", "mfj")
            self.ctx.current_entity_id = eid
        self.ok_btn.setEnabled(False)
        svc = ImportService(self.ctx)
        run_task(svc.execute, eid, self._plan, self.acct_type.currentText(), on_done=self._done, on_error=self._fail, on_progress=self.info.setText)

    def _done(self, out: dict) -> None:
        self.ok_btn.setEnabled(True)
        msg = f"Imported {out['imported']} lots into {', '.join(out['accounts'])}."
        if out["skipped"]:
            msg += f"\nSkipped {len(out['skipped'])}: " + "; ".join(out["skipped"][:8])
        QMessageBox.information(self, "Import complete", msg)
        self.accept()

    def _fail(self, msg: str) -> None:
        self.ok_btn.setEnabled(True)
        QMessageBox.critical(self, "Import failed", msg[:2000])
