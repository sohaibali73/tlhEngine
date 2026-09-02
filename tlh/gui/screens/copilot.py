"""AI co-pilot panel: streamed markdown chat with live reasoning and tool cards, cancel, cost meter; review/approve
proposed code changes; version history with rollback; audit and activity logs."""
from __future__ import annotations

import html
import json
import time

import pandas as pd
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ...ai.copilot import ChatCallbacks, Copilot
from .. import theme
from ..widgets import FrameTable, TextPanel, button, hbox, header, vbox
from ..workers import run_task

CHANGE_COLS = ["id", "created_at", "module_path", "title", "status", "sandbox_passed", "approved_by"]
VER_COLS = ["id", "module_path", "version_no", "created_at", "source", "is_active", "change_id", "n_chars"]
AUDIT_COLS = ["ts", "actor", "action", "target", "details"]
ACT_COLS = ["time", "tool", "args", "ok", "seconds", "result"]

SUGGESTIONS = [
    "Explain today's harvest recommendation and why each blocked lot is blocked.",
    "Which factors drive our tracking error, and what wash-safe trades would cut it most?",
    "Build a 35-name low-vol, quality-tilted basket from the S&P 500 that excludes my current holdings, then compare its TE to SPY.",
    "Create a model portfolio 'Core 50' (min TE to S&P 500, 50 names, 2% sector band), set it as the benchmark, and run a full-rebalance harvest toward it.",
    "Design a harvest plan by hand: sell every short-term loss lot over $500 and replace each with the most correlated wash-safe peer; evaluate it.",
    "Invent a 'dividend yield' style factor as a custom module, test it, propose it, then fit a model variant including it and compare with the active model.",
    "Fit a 3-year-lookback model variant with macro factors on and compare factor vols and my TE against the active model.",
    "Run the TE frontier and tell me where the tax-alpha-per-unit-TE flattens.",
    "Build risk-parity, HRP and minimum-variance baskets of 30 names each, backtest all three quarterly since 2022 vs SPY, and rank them by information ratio (state the caveats).",
    "Use Black-Litterman with a view that semiconductors outperform software by 4% (confidence 0.6) to build a 40-name basket; compare its exposures to the S&P 500.",
    "Run a tax-aware transition from my current holdings to basket 'Core 40' with a 0.5% net realised-gain budget and 40% turnover cap; then evaluate the implied trades.",
]


class _Streamer(QObject):
    text = Signal(str)
    thinking = Signal(str)
    tool_start = Signal(str, str, str)
    tool_end = Signal(str, str, str, bool, float)
    status = Signal(str)
    done = Signal(object)
    failed = Signal(str)


class DiffHighlighter(QSyntaxHighlighter):
    def highlightBlock(self, text: str) -> None:
        fmt = QTextCharFormat()
        if text.startswith("+") and not text.startswith("+++"):
            fmt.setForeground(QColor(theme.GREEN))
        elif text.startswith("-") and not text.startswith("---"):
            fmt.setForeground(QColor(theme.RED))
        elif text.startswith("@@"):
            fmt.setForeground(QColor(theme.ACCENT2))
        else:
            return
        self.setFormat(0, len(text), fmt)


def md_to_html(text: str) -> str:
    doc = QTextDocument()
    doc.setMarkdown(text)
    body = doc.toHtml()
    # keep only the body content so our own container styling applies
    i, j = body.find("<body"), body.rfind("</body>")
    if i >= 0 and j > i:
        body = body[body.find(">", i) + 1: j]
    return body


class Transcript(QTextBrowser):
    """Renders a list of items (user / assistant / thinking / tool) as HTML; throttled re-render while streaming."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.items: list[dict] = []
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._flush)
        self._timer.start()
        self.document().setDefaultStyleSheet(
            f"body{{color:{theme.TEXT};font-family:'Segoe UI';font-size:12px}} table{{border-collapse:collapse}} "
            f"td,th{{border:1px solid {theme.BORDER};padding:2px 6px}} code{{background:{theme.BG3};color:{theme.ACCENT2}}} "
            f"pre{{background:{theme.BG3};padding:6px;border:1px solid {theme.BORDER}}} h1,h2,h3{{color:{theme.TEXT}}}")

    def clear_items(self) -> None:
        self.items = []
        self._dirty = True

    def add(self, item: dict) -> dict:
        self.items.append(item)
        self._dirty = True
        return item

    def touch(self) -> None:
        self._dirty = True

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        parts = []
        for it in self.items:
            k = it["kind"]
            if k == "user":
                parts.append(f"<div style='margin:10px 0 4px 0;padding:6px 10px;background:{theme.BG3};border-left:3px solid {theme.ACCENT2}'>"
                             f"<b style='color:{theme.ACCENT2}'>You</b><br>{html.escape(it['text']).replace(chr(10), '<br>')}</div>")
            elif k == "thinking":
                t = it["text"].strip()
                if t:
                    parts.append(f"<div style='margin:4px 0;color:{theme.MUTED};font-size:11px;border-left:2px dotted {theme.BORDER};padding-left:8px'>"
                                 f"<i>reasoning</i> · {html.escape(t[-700:]).replace(chr(10), ' ')}</div>")
            elif k == "tool":
                st = it.get("status", "running")
                col = theme.AMBER if st == "running" else (theme.GREEN if it.get("ok") else theme.RED)
                icon = "⏳" if st == "running" else ("✓" if it.get("ok") else "✗")
                secs = f" · {it['seconds']:.1f}s" if it.get("seconds") is not None else ""
                args = html.escape(json.dumps(it.get("args", {}), default=str)[:220])
                res = html.escape((it.get("result") or "")[:300]).replace(chr(10), " ")
                parts.append(f"<div style='margin:4px 0;padding:4px 8px;background:{theme.BG2};border:1px solid {theme.BORDER};font-size:11px'>"
                             f"<span style='color:{col}'>{icon}</span> <b>{html.escape(it['name'])}</b> <span style='color:{theme.MUTED}'>{args}{secs}</span>"
                             + (f"<br><span style='color:{theme.MUTED}'>{res}</span>" if res and st != "running" else "") + "</div>")
            elif k == "assistant":
                parts.append(f"<div style='margin:6px 0 10px 0'><b style='color:{theme.GREEN}'>YANG</b>{md_to_html(it['text'])}</div>")
            elif k == "error":
                parts.append(f"<div style='margin:6px 0;color:{theme.RED}'>{html.escape(it['text'])}</div>")
        sb = self.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 40
        self.setHtml("<body>" + "".join(parts) + "</body>")
        if at_bottom:
            self.moveCursor(QTextCursor.MoveOperation.End)
            sb.setValue(sb.maximum())


class CopilotScreen(QWidget):
    code_promoted = Signal(str)
    state_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.copilot = Copilot(self.ctx)
        self.conversation_id: int | None = None
        self._sel_change: int | None = None
        self._sel_version: dict | None = None
        self._cur_assistant: dict | None = None
        self._cur_thinking: dict | None = None
        self._tool_items: dict[str, dict] = {}
        self._activity: list[dict] = []
        self._session_usage: dict[str, int] = {}
        self._session_cost = 0.0
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        top.addWidget(header("YANG — your quant co-pilot", f"Built on {self.copilot.model} · effort {self.copilot.effort}. Builds baskets, plans and runs harvests, fits model variants, "
                                            "designs TLH model pipelines, and proposes code changes that you approve (sandboxed, versioned, reversible)."))
        top.addStretch(1)
        self.conv = QComboBox()
        self.conv.setMinimumWidth(260)
        self.conv.currentIndexChanged.connect(self._switch_conv)
        top.addWidget(QLabel("Conversation"))
        top.addWidget(self.conv)
        top.addWidget(button("New", self.new_conversation))
        root.addLayout(top)
        if not self.copilot.available:
            b = QLabel("ANTHROPIC_API_KEY is not set. Add it to .env at the repo root and restart to enable the co-pilot.")
            b.setProperty("banner", "warn")
            root.addWidget(b)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        chat = QWidget()
        cl = QVBoxLayout(chat)
        cl.setContentsMargins(0, 0, 4, 0)
        self.transcript = Transcript()
        cl.addWidget(self.transcript, 1)
        self.sugg = QComboBox()
        self.sugg.addItem("Suggestions…")
        self.sugg.addItems(SUGGESTIONS)
        self.sugg.currentIndexChanged.connect(lambda i: (self.input.setPlainText(self.sugg.currentText()) if i > 0 else None))
        cl.addWidget(self.sugg)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Ask YANG: portfolio questions, build a basket, design a harvest plan or a TLH model pipeline, fit a model variant, request a code change…  (Ctrl+Enter to send)")
        self.input.setMaximumHeight(90)
        self.input.installEventFilter(self)
        cl.addWidget(self.input)
        self.status_lbl = QLabel("")
        self.status_lbl.setProperty("muted", True)
        self.cost_lbl = QLabel("")
        self.cost_lbl.setProperty("muted", True)
        self.send_btn = button("Send", self.send, primary=True)
        self.stop_btn = button("Stop", self.stop, danger=True)
        self.stop_btn.setEnabled(False)
        cl.addWidget(hbox(self.status_lbl, None, self.cost_lbl, self.stop_btn, self.send_btn))
        split.addWidget(chat)

        self.tabs = QTabWidget()
        self.changes = FrameTable(CHANGE_COLS)
        self.changes.row_selected.connect(self._change_selected)
        self.diff = QPlainTextEdit()
        self.diff.setReadOnly(True)
        self.diff.setFont(QFont("Consolas", 9))
        DiffHighlighter(self.diff.document())
        self.sandbox_out = QPlainTextEdit()
        self.sandbox_out.setReadOnly(True)
        self.sandbox_out.setFont(QFont("Consolas", 9))
        self.rationale = TextPanel("Rationale")
        acts = hbox(button("Approve & promote", self._approve, success=True), button("Reject", self._reject, danger=True),
                    button("Re-run sandbox", self._retest), None)
        detail = QTabWidget()
        detail.addTab(self.diff, "Diff")
        detail.addTab(self.sandbox_out, "Sandbox tests")
        detail.addTab(self.rationale, "Rationale")
        csplit = QSplitter(Qt.Vertical)
        csplit.addWidget(self.changes)
        csplit.addWidget(detail)
        csplit.setSizes([220, 500])
        self.tabs.addTab(vbox(csplit, acts), "Proposed changes")
        self.activity = FrameTable(ACT_COLS)
        self.tabs.addTab(self.activity, "Tool activity")
        self.versions = FrameTable(VER_COLS)
        self.versions.row_selected.connect(self._show_version)
        self.version_code = QPlainTextEdit()
        self.version_code.setReadOnly(True)
        self.version_code.setFont(QFont("Consolas", 9))
        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(self.versions)
        vsplit.addWidget(self.version_code)
        self.tabs.addTab(vbox(vsplit, hbox(button("Roll back module to selected version", self._rollback, danger=True), None)), "Code versions")
        self.audit = FrameTable(AUDIT_COLS)
        self.tabs.addTab(self.audit, "Audit log")
        split.addWidget(self.tabs)
        split.setSizes([760, 640])

    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type() == QEvent.KeyPress and ev.key() in (Qt.Key_Return, Qt.Key_Enter) and ev.modifiers() & Qt.ControlModifier:
            self.send()
            return True
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------------ conversations
    def refresh(self) -> None:
        df = self.ctx.conversations.list()
        cur = self.conversation_id
        self.conv.blockSignals(True)
        self.conv.clear()
        for _, r in df.iterrows():
            self.conv.addItem(f"#{r['id']} {r['title'] or r['created_at']} ({r['n']})", int(r["id"]))
        if cur is not None:
            i = self.conv.findData(cur)
            if i >= 0:
                self.conv.setCurrentIndex(i)
        self.conv.blockSignals(False)
        if self.conversation_id is None and self.conv.count():
            self.conversation_id = self.conv.currentData()
            self._render_history()
        self.changes.set_frame(self.ctx.code.changes())
        self.versions.set_frame(self.ctx.code.versions())
        self.audit.set_frame(pd.DataFrame([dict(r) for r in self.ctx.db.fetchall("SELECT * FROM audit_log ORDER BY id DESC LIMIT 500")]))

    def new_conversation(self) -> None:
        self.conversation_id = self.copilot.new_conversation()
        self.transcript.clear_items()
        self.refresh()

    def select_conversation(self, cid: int) -> None:
        self.refresh()
        i = self.conv.findData(cid)
        if i >= 0:
            self.conv.blockSignals(True)
            self.conv.setCurrentIndex(i)
            self.conv.blockSignals(False)
        self.conversation_id = cid
        self._render_history()

    def _switch_conv(self, i: int) -> None:
        cid = self.conv.currentData()
        if cid is not None and cid != self.conversation_id:
            self.conversation_id = cid
            self._render_history()

    def _render_history(self) -> None:
        self.transcript.clear_items()
        if self.conversation_id is None:
            return
        for m in self.ctx.conversations.messages(self.conversation_id):
            c = m["content"]
            if m["role"] == "user":
                if isinstance(c, str):
                    self.transcript.add({"kind": "user", "text": c})
                continue
            if isinstance(c, list):
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text" and b.get("text"):
                        self.transcript.add({"kind": "assistant", "text": b["text"]})
                    elif b.get("type") == "tool_use":
                        self.transcript.add({"kind": "tool", "name": b.get("name", ""), "args": b.get("input", {}), "status": "done", "ok": True})

    # ------------------------------------------------------------------ chat
    def send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        if not self.copilot.available:
            QMessageBox.warning(self, "No API key", "Add ANTHROPIC_API_KEY to .env and restart.")
            return
        if self.conversation_id is None:
            self.conversation_id = self.copilot.new_conversation(title=text[:60])
        self.input.clear()
        self.transcript.add({"kind": "user", "text": text})
        self._cur_assistant = None
        self._cur_thinking = None
        self._tool_items = {}
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._t0 = time.time()
        self._set_status("thinking…")
        st = _Streamer()
        st.text.connect(self._on_text)
        st.thinking.connect(self._on_thinking)
        st.tool_start.connect(self._on_tool_start)
        st.tool_end.connect(self._on_tool_end)
        st.status.connect(self._set_status)
        st.done.connect(self._turn_done)
        st.failed.connect(self._turn_failed)
        self._streamer = st
        cb = ChatCallbacks(on_text=st.text.emit, on_thinking=st.thinking.emit,
                           on_tool_start=lambda i, n, a: st.tool_start.emit(i, n, json.dumps(a, default=str)),
                           on_tool_end=lambda i, n, r, ok, s: st.tool_end.emit(i, n, r, ok, s), on_status=st.status.emit)
        run_task(lambda: self.copilot.chat(self.conversation_id, text, cb), on_done=st.done.emit, on_error=st.failed.emit, wants_progress=False)

    def stop(self) -> None:
        self.copilot.cancel()
        self._set_status("stopping…")

    def _set_status(self, s: str) -> None:
        self.status_lbl.setText(f"● {s}   ({time.time() - getattr(self, '_t0', time.time()):.0f}s)")

    def _on_text(self, t: str) -> None:
        if self._cur_assistant is None or self._cur_assistant.get("closed"):
            self._cur_assistant = self.transcript.add({"kind": "assistant", "text": ""})
        self._cur_assistant["text"] += t
        self.transcript.touch()
        self._set_status("writing…")

    def _on_thinking(self, t: str) -> None:
        if not t:
            return
        if self._cur_thinking is None or (self._cur_assistant is not None and not self._cur_assistant.get("closed")):
            self._cur_thinking = self.transcript.add({"kind": "thinking", "text": ""})
            if self._cur_assistant is not None:
                self._cur_assistant["closed"] = True
        self._cur_thinking["text"] += t
        self.transcript.touch()
        self._set_status("reasoning…")

    def _on_tool_start(self, tid: str, name: str, args_json: str) -> None:
        if self._cur_assistant is not None:
            self._cur_assistant["closed"] = True
        self._cur_thinking = None
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            args = {"raw": args_json}
        self._tool_items[tid] = self.transcript.add({"kind": "tool", "name": name, "args": args, "status": "running"})
        self._set_status(f"running {name}…")

    def _on_tool_end(self, tid: str, name: str, result: str, ok: bool, secs: float) -> None:
        it = self._tool_items.get(tid)
        if it is not None:
            it.update({"status": "done", "ok": ok, "seconds": secs, "result": result})
            self.transcript.touch()
        self._activity.append({"time": time.strftime("%H:%M:%S"), "tool": name, "args": json.dumps(it.get("args", {}) if it else {}, default=str)[:200],
                               "ok": ok, "seconds": round(secs, 2), "result": result[:200].replace("\n", " ")})
        self.activity.set_frame(pd.DataFrame(self._activity[::-1]))
        self._set_status("thinking…")

    def _turn_done(self, res) -> None:
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self._cur_assistant is not None:
            self._cur_assistant["closed"] = True
        for k, v in res.usage.items():
            self._session_usage[k] = self._session_usage.get(k, 0) + v
        self._session_cost += res.cost_usd
        u = res.usage
        self.status_lbl.setText(f"done ({res.stop_reason}) in {res.duration_s:.0f}s · {len(res.tool_calls)} tool calls")
        self.cost_lbl.setText(f"turn: {u.get('input_tokens', 0):,} in / {u.get('cache_read_input_tokens', 0):,} cached / {u.get('output_tokens', 0):,} out "
                              f"≈ ${res.cost_usd:.2f} · session ≈ ${self._session_cost:.2f}")
        self.refresh()
        self.state_changed.emit()
        if res.change_ids:
            self.tabs.setCurrentIndex(0)
            QMessageBox.information(self, "Change proposed", f"YANG proposed change(s) #{', #'.join(map(str, res.change_ids))}. "
                                                             "Review the diff and sandbox results, then approve or reject.")

    def _turn_failed(self, msg: str) -> None:
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("failed")
        self.transcript.add({"kind": "error", "text": msg.splitlines()[0][:400]})
        self.app.error(msg)

    # ------------------------------------------------------------------ changes
    def _change_selected(self, row: dict) -> None:
        self._sel_change = int(row["id"])
        ch = self.ctx.code.change(self._sel_change)
        self.diff.setPlainText(ch.get("diff_text") or "(no diff)")
        self.sandbox_out.setPlainText(ch.get("sandbox_stdout") or "(sandbox not run)")
        self.rationale.set_text(f"{ch['title']}\n\n{ch.get('rationale') or ''}\n\nStatus: {ch['status']} · sandbox passed: {bool(ch.get('sandbox_passed'))}")

    def _approve(self) -> None:
        if self._sel_change is None:
            return
        ch = self.ctx.code.change(self._sel_change)
        force = False
        if not ch.get("sandbox_passed"):
            if QMessageBox.question(self, "Tests failing", "Sandbox tests did not pass. Promote anyway? (Not recommended.)") != QMessageBox.Yes:
                return
            force = True
        if QMessageBox.question(self, "Promote", f"Write change #{self._sel_change} to {ch['module_path']} and hot-reload? The previous version stays available for rollback.") != QMessageBox.Yes:
            return
        try:
            vid = self.copilot.approve_and_promote(self._sel_change, approved_by="user", force=force)
        except Exception as e:
            self.app.error(str(e))
            return
        from ...ai.registry import AI_EDITABLE
        m = AI_EDITABLE.get(ch["module_path"])
        self.app.status(f"Promoted change #{self._sel_change} as version #{vid}. {m.reload_hint if m else ''}")
        self.refresh()
        self.code_promoted.emit(ch["module_path"])

    def _reject(self) -> None:
        if self._sel_change is None:
            return
        self.copilot.reject(self._sel_change)
        self.refresh()

    def _retest(self) -> None:
        if self._sel_change is None:
            return
        self.app.status("Re-running sandbox…")
        cid = self._sel_change
        run_task(self.copilot.retest, cid, on_done=lambda ok: (self.app.status(f"Sandbox {'passed' if ok else 'failed'}."), self.refresh(), self._change_selected({"id": cid})),
                 on_error=self.app.error, wants_progress=False)

    # ------------------------------------------------------------------ versions
    def _show_version(self, row: dict) -> None:
        self._sel_version = row
        v = self.ctx.code.get_version(int(row["id"]))
        self.version_code.setPlainText(v["code_text"] if v else "")

    def _rollback(self) -> None:
        if not self._sel_version:
            return
        v = self._sel_version
        if QMessageBox.question(self, "Roll back", f"Restore {v['module_path']} to version {v['version_no']}? A new version row is created; nothing is lost.") != QMessageBox.Yes:
            return
        try:
            self.copilot.rollback(v["module_path"], int(v["id"]))
        except Exception as e:
            self.app.error(str(e))
            return
        self.refresh()
        self.code_promoted.emit(v["module_path"])
