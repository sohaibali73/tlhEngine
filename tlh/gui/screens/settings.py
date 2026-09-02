"""Settings: entities/accounts, tax profile, carryforwards, wash-sale conventions, data snapshots, demo, AI config."""
from __future__ import annotations

from datetime import date

import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...services.demo import reset_demo, seed_demo
from ...tax.rates import TaxProfile
from ..dialogs import AccountDialog, EntityDialog
from ..widgets import FrameTable, button, hbox, header
from ..workers import run_task


class SettingsScreen(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self._build()

    def _build(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        w = QWidget()
        scroll.setWidget(w)
        root = QVBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(header("Settings"))
        row = QHBoxLayout()
        root.addLayout(row)
        left = QVBoxLayout()
        right = QVBoxLayout()
        row.addLayout(left, 1)
        row.addLayout(right, 1)

        # ---- entities & accounts
        g = QGroupBox("Tax entities & accounts (wash-sale scope = all accounts in an entity)")
        gl = QVBoxLayout(g)
        self.accounts = FrameTable(["id", "entity_id", "name", "account_type", "broker", "owner"], filter_box=False)
        gl.addWidget(self.accounts)
        gl.addWidget(hbox(button("Add entity", self._add_entity), button("Add account", self._add_account), None))
        left.addWidget(g)

        # ---- tax profile
        g = QGroupBox("Tax assumptions (default profile)")
        f = QFormLayout(g)
        self.fed_st = self._pct()
        self.fed_lt = self._pct()
        self.state = self._pct()
        self.niit = self._pct()
        self.filing = QComboBox()
        self.filing.addItems(["single", "mfj", "mfs", "hoh"])
        self.offset = QDoubleSpinBox()
        self.offset.setRange(0, 1e6)
        self.offset.setPrefix("$ ")
        f.addRow("Federal short-term", self.fed_st)
        f.addRow("Federal long-term", self.fed_lt)
        f.addRow("State", self.state)
        f.addRow("NIIT", self.niit)
        f.addRow("Filing status", self.filing)
        f.addRow("Ordinary-income offset", self.offset)
        self.cf_year = QSpinBox()
        self.cf_year.setRange(2000, 2100)
        self.cf_year.setValue(date.today().year - 1)
        self.cf_st = QDoubleSpinBox()
        self.cf_st.setRange(0, 1e9)
        self.cf_st.setPrefix("$ ")
        self.cf_lt = QDoubleSpinBox()
        self.cf_lt.setRange(0, 1e9)
        self.cf_lt.setPrefix("$ ")
        f.addRow("Carryforward from year", self.cf_year)
        f.addRow("  short-term loss", self.cf_st)
        f.addRow("  long-term loss", self.cf_lt)
        f.addRow(hbox(button("Save tax settings", self._save_tax, primary=True), None))
        left.addWidget(g)

        # ---- wash-sale / data conventions
        g = QGroupBox("Wash-sale & data conventions")
        f = QFormLayout(g)
        self.presumed = QCheckBox("Treat same-index ETFs from different issuers as substantially identical (conservative)")
        self.universe = QLineEdit()
        self.benchmark = QLineEdit()
        f.addRow(self.presumed)
        f.addRow("Fit universe (Norgate watchlist)", self.universe)
        f.addRow("Benchmark (watchlist or ETF)", self.benchmark)
        f.addRow(hbox(button("Save conventions", self._save_conv), None))
        right.addWidget(g)

        # ---- data snapshots
        g = QGroupBox("Data snapshots (reproducibility units)")
        gl = QVBoxLayout(g)
        self.snaps = FrameTable(["id", "as_of_date", "created_at", "universe_name", "n_symbols", "notes"], filter_box=False)
        self.snaps.row_selected.connect(lambda r: setattr(self, "_sel_snap", r["id"]))
        gl.addWidget(self.snaps)
        gl.addWidget(hbox(button("Refresh data now (new snapshot)", self.app.refresh_data), button("Delete selected", self._del_snap, danger=True), None))
        right.addWidget(g)

        # ---- AI (YANG)
        g = QGroupBox("YANG (AI co-pilot) — API key & model")
        f = QFormLayout(g)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("sk-ant-…  (stored only in the local .env file, never in the database)")
        self.api_status = QLabel("")
        self.api_status.setProperty("muted", True)
        self.ai_model = QComboBox()
        self.ai_model.setEditable(True)
        self.ai_model.addItems(["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"])
        self.ai_effort = QComboBox()
        self.ai_effort.addItems(["low", "medium", "high", "xhigh", "max"])
        f.addRow("Anthropic API key", self.api_key)
        f.addRow("", self.api_status)
        f.addRow("Model", self.ai_model)
        f.addRow("Effort", self.ai_effort)
        f.addRow(hbox(button("Save to .env & activate", self._save_api, primary=True), button("Test key", self._test_api), None))
        note = QLabel("Saved to .env next to the app (git-ignored). Takes effect immediately for YANG, the pop-up and the agent; no restart needed.")
        note.setProperty("muted", True)
        note.setWordWrap(True)
        f.addRow(note)
        right.addWidget(g)

        # ---- demo
        g = QGroupBox("Demo data")
        gl = QVBoxLayout(g)
        lbl = QLabel("Seeds a 'Demo Household' with three accounts (brokerage, spouse, IRA), ~45 real lots priced from Norgate history, "
                     "a recent loss sale and a scheduled DRIP. Idempotent.")
        lbl.setWordWrap(True)
        lbl.setProperty("muted", True)
        gl.addWidget(lbl)
        gl.addWidget(hbox(button("Seed demo household", self._seed), button("Reset demo household", self._reset, danger=True), None))
        right.addWidget(g)
        left.addStretch(1)
        right.addStretch(1)

    def _pct(self) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0, 100)
        s.setDecimals(2)
        s.setSuffix(" %")
        return s

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        accts = self.ctx.entities.accounts(None, active_only=False)
        self.accounts.set_frame(pd.DataFrame([a.__dict__ for a in accts]))
        p = self.ctx.tax.default_profile()
        self.fed_st.setValue(p.fed_st_rate * 100)
        self.fed_lt.setValue(p.fed_lt_rate * 100)
        self.state.setValue(p.state_rate * 100)
        self.niit.setValue(p.niit_rate * 100)
        self.offset.setValue(p.ordinary_offset)
        ents = self.ctx.entities.list()
        eid = self.ctx.current_entity_id
        if eid is not None:
            e = next((x for x in ents if x["id"] == eid), None)
            if e:
                self.filing.setCurrentText(e["filing_status"])
            st, lt = self.ctx.tax.carryforward(eid, self.cf_year.value())
            self.cf_st.setValue(st)
            self.cf_lt.setValue(lt)
        self.presumed.setChecked(self.ctx.treat_presumed_identical)
        self.universe.setText(self.ctx.get("universe_name", self.ctx.settings.default_universe))
        self.benchmark.setText(self.ctx.get("benchmark_name", self.ctx.settings.default_benchmark))
        self.snaps.set_frame(pd.DataFrame([s.__dict__ | {"path": str(s.path)} for s in self.ctx.store.list()]))
        self._refresh_api()

    # ------------------------------------------------------------------ YANG settings
    def _refresh_api(self) -> None:
        s = self.ctx.settings
        self.api_status.setText(("key set (…" + s.anthropic_api_key[-6:] + ")") if s.has_anthropic_key else "no key saved")
        self.ai_model.setCurrentText(s.ai_model)
        self.ai_effort.setCurrentText(s.ai_effort)

    def _save_api(self) -> None:
        from ...config import update_env_file
        updates = {"TLH_AI_MODEL": self.ai_model.currentText().strip(), "TLH_AI_EFFORT": self.ai_effort.currentText().strip()}
        key = self.api_key.text().strip()
        if key:
            updates["ANTHROPIC_API_KEY"] = key
        try:
            path = update_env_file(updates)
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self.api_key.clear()
        self.app.reload_ai_settings()
        self._refresh_api()
        self.app.status(f"Saved to {path.name}; YANG reconfigured ({self.ctx.settings.ai_model}, effort {self.ctx.settings.ai_effort}).")

    def _test_api(self) -> None:
        key = self.api_key.text().strip() or self.ctx.settings.anthropic_api_key
        if not key:
            QMessageBox.information(self, "Test key", "Enter a key first.")
            return
        model = self.ai_model.currentText().strip()

        def work():
            import anthropic
            c = anthropic.Anthropic(api_key=key)
            m = c.models.retrieve(model)
            return f"OK: key accepted, model {m.id} available"

        run_task(work, on_done=lambda msg: (self.api_status.setText(msg), self.app.status(msg)),
                 on_error=lambda msg: (self.api_status.setText("key rejected: " + msg.splitlines()[0][:120]), self.app.status("API key test failed.")), wants_progress=False)

    # ------------------------------------------------------------------ actions
    def _add_entity(self) -> None:
        d = EntityDialog(self)
        if d.exec():
            v = d.values()
            if v["name"]:
                eid = self.ctx.entities.get_or_create(v["name"], v["filing_status"])
                self.ctx.current_entity_id = eid
                self.app.reload_entities()
                self.refresh()

    def _add_account(self) -> None:
        ents = self.ctx.entities.list()
        if not ents:
            QMessageBox.information(self, "No entity", "Add a tax entity first.")
            return
        d = AccountDialog(ents, self)
        if d.exec():
            v = d.values()
            if v["name"]:
                self.ctx.entities.get_or_create_account(v["entity_id"], v["name"], v["account_type"], v["broker"], v["owner"])
                self.refresh()
                self.app.data_changed()

    def _save_tax(self) -> None:
        p = TaxProfile(name="default", fed_st_rate=self.fed_st.value() / 100, fed_lt_rate=self.fed_lt.value() / 100,
                       state_rate=self.state.value() / 100, niit_rate=self.niit.value() / 100, ordinary_offset=self.offset.value(),
                       filing_status=self.filing.currentText())
        self.ctx.tax.save(p, make_default=True)
        eid = self.ctx.current_entity_id
        if eid is not None:
            self.ctx.db.update("entities", "id = ?", (eid,), filing_status=self.filing.currentText())
            self.ctx.tax.set_carryforward(eid, self.cf_year.value(), self.cf_st.value(), self.cf_lt.value())
        self.app.status("Tax settings saved.")
        self.app.data_changed()

    def _save_conv(self) -> None:
        self.ctx.set("treat_presumed_identical_as_identical", self.presumed.isChecked())
        self.ctx.set("universe_name", self.universe.text().strip())
        self.ctx.set("benchmark_name", self.benchmark.text().strip())
        self.ctx.reload_substitutes()
        self.app.status("Conventions saved. Wash-sale groups reloaded.")
        self.app.data_changed()

    def _del_snap(self) -> None:
        sid = getattr(self, "_sel_snap", None)
        if sid and QMessageBox.question(self, "Delete snapshot", f"Delete {sid} and its Parquet files? Runs referencing it keep their saved results.") == QMessageBox.Yes:
            self.ctx.store.delete(sid)
            self.refresh()

    def _seed(self) -> None:
        self.app.status("Seeding demo household…")
        run_task(seed_demo, self.ctx, on_done=lambda eid: (self.app.reload_entities(), self.app.data_changed(), self.refresh(), self.app.status("Demo seeded.")),
                 on_error=self.app.error, on_progress=self.app.status)

    def _reset(self) -> None:
        if QMessageBox.question(self, "Reset demo", "Delete the Demo Household and all its lots/transactions?") == QMessageBox.Yes:
            reset_demo(self.ctx)
            self.app.reload_entities()
            self.app.data_changed()
            self.refresh()
