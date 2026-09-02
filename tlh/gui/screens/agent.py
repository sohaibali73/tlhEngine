"""Agent: scheduled and ad-hoc unattended co-pilot jobs, their reports, and the scheduler status."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ...ai import schedule as sched
from ...services.agent_service import TEMPLATES
from .. import theme
from ..widgets import FrameTable, KpiCard, button, hbox, header, vbox

TASK_COLS = ["id", "name", "schedule_desc", "enabled", "next_run_at", "last_status", "last_run_at", "effort"]
RUN_COLS = ["id", "started_at", "name", "status", "trigger", "cost_usd", "duration_s", "tool_calls", "is_read"]
SCHEDULE_EXAMPLES = ["manual", "startup", "every 30m", "every 2h", "daily 08:30", "weekdays 16:30", "weekly mon 09:00", "monthly 1 09:00"]


class TaskDialog(QDialog):
    def __init__(self, task: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent task")
        self.resize(720, 460)
        f = QFormLayout(self)
        self.template = QComboBox()
        self.template.addItem("— start from a template —")
        for t in TEMPLATES:
            self.template.addItem(t["name"], t)
        self.template.currentIndexChanged.connect(self._fill)
        self.name = QLineEdit(task["name"] if task else "")
        self.prompt = QPlainTextEdit(task["prompt"] if task else "")
        self.prompt.setMinimumHeight(200)
        self.schedule = QComboBox()
        self.schedule.setEditable(True)
        self.schedule.addItems(SCHEDULE_EXAMPLES)
        self.schedule.setCurrentText(task["schedule"] if task else "manual")
        self.sched_desc = QLabel("")
        self.sched_desc.setProperty("muted", True)
        self.schedule.currentTextChanged.connect(lambda s: self.sched_desc.setText(sched.describe(s)))
        self.sched_desc.setText(sched.describe(self.schedule.currentText()))
        self.effort = QComboBox()
        self.effort.addItems(["(default)", "low", "medium", "high", "xhigh"])
        if task and task.get("effort"):
            self.effort.setCurrentText(task["effort"])
        self.enabled = QCheckBox("enabled (runs on schedule)")
        self.enabled.setChecked(bool(task["enabled"]) if task else True)
        self.notify = QCheckBox("tray notification when done")
        self.notify.setChecked(bool(task["notify"]) if task else True)
        if not task:
            f.addRow("Template", self.template)
        f.addRow("Name", self.name)
        f.addRow("Instructions", self.prompt)
        f.addRow("Schedule", self.schedule)
        f.addRow("", self.sched_desc)
        f.addRow("Effort", self.effort)
        f.addRow(self.enabled)
        f.addRow(self.notify)
        note = QLabel("The task runs as an unattended co-pilot conversation with the full tool set. It cannot trade; code changes it proposes wait for your approval.")
        note.setWordWrap(True)
        note.setProperty("muted", True)
        f.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _fill(self, i: int) -> None:
        t = self.template.currentData()
        if t:
            self.name.setText(t["name"])
            self.prompt.setPlainText(t["prompt"])
            self.schedule.setCurrentText(t["schedule"])
            self.effort.setCurrentText(t.get("effort") or "(default)")

    def _accept(self) -> None:
        try:
            sched.parse(self.schedule.currentText())
        except ValueError as e:
            QMessageBox.warning(self, "Schedule", str(e))
            return
        if not self.name.text().strip() or not self.prompt.toPlainText().strip():
            QMessageBox.warning(self, "Task", "Name and instructions are required.")
            return
        self.accept()

    def values(self) -> dict:
        eff = self.effort.currentText()
        return {"name": self.name.text().strip(), "prompt": self.prompt.toPlainText().strip(), "schedule": self.schedule.currentText().strip(),
                "enabled": self.enabled.isChecked(), "notify": self.notify.isChecked(), "effort": None if eff.startswith("(") else eff}


class AgentScreen(QWidget):
    state_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.svc = app.agent_service
        self._sel_task: dict | None = None
        self._sel_run: dict | None = None
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        top.addWidget(header("YANG Agent", "YANG runs jobs unattended: on a schedule, at startup, or on demand from the pop-up (Ctrl+Space / global hotkey). "
                                      "Reports land here; anything needing approval lands in YANG › Proposed changes."))
        top.addStretch(1)
        top.addWidget(button("Ask YANG now…", self.app.show_quick, primary=True))
        root.addLayout(top)
        k = QHBoxLayout()
        self.k_sched = KpiCard("Scheduler")
        self.k_next = KpiCard("Next due")
        self.k_unread = KpiCard("Unread reports")
        self.k_pending = KpiCard("Changes awaiting approval")
        self.k_cost = KpiCard("Agent spend (all runs)")
        for c in (self.k_sched, self.k_next, self.k_unread, self.k_pending, self.k_cost):
            k.addWidget(c)
        root.addLayout(k)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        # tasks
        self.tasks = FrameTable(TASK_COLS, filter_box=False)
        self.tasks.row_selected.connect(lambda r: setattr(self, "_sel_task", r))
        self.tasks.row_activated.connect(lambda r: self._edit())
        tbtn = hbox(button("New task", self._new), button("Install templates", self._templates, tooltip="Adds the five standard tasks (disabled) if missing"),
                    button("Edit", self._edit), button("Enable / disable", self._toggle), button("Run now", self._run_now, success=True),
                    button("Delete", self._delete, danger=True), None)
        split.addWidget(vbox(header("Tasks"), self.tasks, tbtn))
        # runs + report
        self.runs = FrameTable(RUN_COLS)
        self.runs.row_selected.connect(self._show_run)
        self.report = QTextBrowser()
        self.report.document().setDefaultStyleSheet(f"body{{color:{theme.TEXT}}} td,th{{border:1px solid {theme.BORDER};padding:2px 6px}} pre{{background:{theme.BG3}}}")
        rbtn = hbox(button("Open conversation", self._open_conv), button("Review proposed changes in YANG", lambda: self.app.goto("AI co-pilot")),
                    button("Mark all read", self._mark_read), None)
        rsplit = QSplitter(Qt.Vertical)
        rsplit.addWidget(self.runs)
        rsplit.addWidget(self.report)
        rsplit.setSizes([260, 420])
        split.addWidget(vbox(header("Runs & reports"), rsplit, rbtn))
        split.setSizes([620, 800])

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        df = self.svc.tasks()
        self.tasks.set_frame(df)
        runs = self.svc.runs()
        self.runs.set_frame(runs)
        running = self.svc.running_run_id
        self.k_sched.set("running" if running else "idle", f"run #{running}" if running else "checks every 30 s while the app is open", theme.AMBER if running else None)
        nxt = df[df["enabled"] == 1]["next_run_at"].dropna() if not df.empty else pd.Series(dtype=object)
        if len(nxt):
            n = sorted(nxt)[0]
            name = df[df["next_run_at"] == n]["name"].iloc[0]
            self.k_next.set(str(n)[:16], name)
        else:
            self.k_next.set("—", "no scheduled tasks enabled")
        unread = self.svc.unread()
        self.k_unread.set(str(unread), "", theme.ACCENT2 if unread else None)
        pend = self.svc.pending_changes()
        self.k_pending.set(str(pend), "in AI co-pilot › Proposed changes", theme.AMBER if pend else None)
        spend = float(runs["cost_usd"].fillna(0).sum()) if not runs.empty and "cost_usd" in runs else 0.0
        self.k_cost.set(f"${spend:.2f}", f"{len(runs)} runs")

    # ------------------------------------------------------------------ tasks
    def _new(self) -> None:
        d = TaskDialog(None, self)
        if d.exec():
            v = d.values()
            try:
                self.svc.save_task(**v)
            except Exception as e:
                QMessageBox.warning(self, "Task", str(e))
                return
            self.refresh()

    def _edit(self) -> None:
        if not self._sel_task:
            return
        t = self.ctx.agent.task(int(self._sel_task["id"]))
        d = TaskDialog(t, self)
        if d.exec():
            v = d.values()
            if v["name"] != t["name"]:
                self.ctx.agent.set_task(int(t["id"]), name=v["name"])
            self.svc.save_task(**v)
            self.refresh()

    def _templates(self) -> None:
        n = self.svc.install_templates(enabled=False)
        self.app.status(f"Installed {n} template task(s) (disabled). Enable the ones you want.")
        self.refresh()

    def _toggle(self) -> None:
        if not self._sel_task:
            return
        self.svc.set_enabled(int(self._sel_task["id"]), not bool(self._sel_task["enabled"]))
        self.refresh()

    def _delete(self) -> None:
        if self._sel_task and QMessageBox.question(self, "Delete task", f"Delete '{self._sel_task['name']}'? Its past runs are kept.") == QMessageBox.Yes:
            self.svc.delete_task(int(self._sel_task["id"]))
            self._sel_task = None
            self.refresh()

    def _run_now(self) -> None:
        if not self._sel_task:
            return
        if self.svc.running_run_id is not None:
            QMessageBox.information(self, "Busy", "An agent job is already running.")
            return
        self.app.run_agent_task(int(self._sel_task["id"]), trigger="manual")
        self.refresh()

    # ------------------------------------------------------------------ runs
    def _show_run(self, row: dict) -> None:
        self._sel_run = row
        r = self.svc.run_detail(int(row["id"]))
        if not r:
            return
        from .copilot import md_to_html
        body = r.get("report") or r.get("error") or ("running…" if r["status"] == "running" else "(no report)")
        head = f"<div style='color:{theme.MUTED};font-size:11px'>{r['name']} · {r['status']} · {r['trigger']} · started {r['started_at']} · " \
               f"${(r.get('cost_usd') or 0):.2f} · {r.get('tool_calls') or 0} tool calls · changes {r.get('change_ids') or []}</div>"
        self.report.setHtml(f"<body>{head}{md_to_html(body)}</body>")
        if not r.get("is_read") and r["status"] != "running":
            self.svc.mark_read([int(row["id"])])
            self.state_changed.emit()

    def _open_conv(self) -> None:
        if not self._sel_run:
            return
        r = self.svc.run_detail(int(self._sel_run["id"]))
        if r and r.get("conversation_id"):
            self.app.goto("AI co-pilot")
            self.app.screens["AI co-pilot"].select_conversation(int(r["conversation_id"]))

    def _mark_read(self) -> None:
        self.svc.mark_read()
        self.refresh()
        self.state_changed.emit()

    def show_live(self, text: str) -> None:
        """Stream a running job's text into the report pane."""
        from .copilot import md_to_html
        self.report.setHtml(f"<body><div style='color:{theme.AMBER}'>running…</div>{md_to_html(text)}</body>")
        sb = self.report.verticalScrollBar()
        sb.setValue(sb.maximum())
