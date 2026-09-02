"""Export center: formatted Excel workbook per harvest run, plus CSV dumps."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ...export.excel import export_run_workbook
from ..widgets import FrameTable, button, hbox, header
from ..workers import run_task


class ExportScreen(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(header("Export", "Institutional-format workbook per run: summary, trade ticket, wash-sale explanations, before/after risk, positions."))
        g = QGroupBox("Harvest run workbook")
        f = QFormLayout(g)
        self.run = QComboBox()
        self.inc_pos = QCheckBox("Include current positions sheet")
        self.inc_pos.setChecked(True)
        self.inc_front = QCheckBox("Include latest frontier & priority comparison (if computed this session)")
        self.inc_front.setChecked(True)
        self.title = QLineEdit("Tax-Loss Harvest Recommendation")
        self.path = QLineEdit(str(self.ctx.settings.exports_dir))
        f.addRow("Run", self.run)
        f.addRow("Title", self.title)
        f.addRow("Output folder", hbox(self.path, button("…", self._pick)))
        f.addRow(self.inc_pos)
        f.addRow(self.inc_front)
        f.addRow(hbox(button("Export workbook", self.export, primary=True), button("Open folder", self._open), None))
        root.addWidget(g)
        note = QLabel("Branding: this workbook uses a neutral dark-header style. If output is destined for internal Potomac review, "
                      "the existing Potomac Excel conventions can be applied instead (see DECISIONS.md D7).")
        note.setProperty("muted", True)
        note.setWordWrap(True)
        root.addWidget(note)
        root.addWidget(header("Recent exports"))
        self.recent = FrameTable(["file", "modified", "size_kb"])
        self.recent.row_activated.connect(lambda r: os.startfile(str(Path(self.path.text()) / r["file"])))
        root.addWidget(self.recent, 1)

    def refresh(self) -> None:
        df = self.ctx.runs.list(limit=100)
        self.run.clear()
        for _, r in df[df["run_type"] == "harvest"].iterrows() if not df.empty else []:
            s = r["summary"]
            self.run.addItem(f"#{r['id']} · {r['created_at'][:16]} · loss ${s.get('harvested_loss', 0):,.0f} · TE {s.get('te_after', 0):.2%}", int(r["id"]))
        folder = Path(self.path.text())
        rows = []
        if folder.exists():
            for p in sorted(folder.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
                st = p.stat()
                rows.append({"file": p.name, "modified": pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"), "size_kb": round(st.st_size / 1024, 1)})
        self.recent.set_frame(pd.DataFrame(rows))

    def _pick(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.path.text())
        if d:
            self.path.setText(d)
            self.refresh()

    def _open(self) -> None:
        p = Path(self.path.text())
        p.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(p)])

    def export(self) -> None:
        rid = self.run.currentData()
        if rid is None:
            self.app.status("No harvest run to export.")
            return
        out = Path(self.path.text()) / f"tlh_run_{rid:04d}_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx"
        self.app.status("Building workbook…")

        def work():
            run = self.app.harvest_service.load_run(rid)
            pos = self.app.portfolio_service.lots_view(self.ctx.current_entity_id) if (self.inc_pos.isChecked() and self.ctx.current_entity_id) else None
            hs = self.app.screens.get("Harvest")
            fr = getattr(hs, "frontier_df", None) if self.inc_front.isChecked() else None
            pr = getattr(hs, "prio_table_df", None) if self.inc_front.isChecked() else None
            return export_run_workbook(out, run, positions=pos, frontier=fr, priority_table=pr, title=self.title.text())

        run_task(work, on_done=lambda p: (self.app.status(f"Exported {p}"), self.refresh(), self.ctx.db.audit("user", "export.xlsx", str(p))),
                 on_error=self.app.error, wants_progress=False)
