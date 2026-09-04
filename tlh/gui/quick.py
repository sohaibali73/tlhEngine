"""'Ask YANG' pop-up, system-wide hotkey (Windows) and tray icon.

The pop-up is a Spotlight-style frameless window: type a job, press Enter, YANG does it with the full tool set
as an ad-hoc agent run, streams progress, and files the report. Summoned by Ctrl+Space inside the app or by the
global hotkey (default Ctrl+Alt+C) from anywhere on the desktop, or from the tray icon.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSystemTrayIcon,
    QTextBrowser,
    QVBoxLayout,
)

from . import theme
from .widgets import button
from .workers import run_task

log = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
MODS = {"ctrl": 0x0002, "alt": 0x0001, "shift": 0x0004, "win": 0x0008}
HOTKEY_ID = 0x7A1


def _vk(key: str) -> int:
    k = key.strip().lower()
    if len(k) == 1:
        return ord(k.upper())
    if k == "space":
        return 0x20
    if k.startswith("f") and k[1:].isdigit():
        return 0x70 + int(k[1:]) - 1
    raise ValueError(f"unsupported key '{key}'")


class GlobalHotkey(QAbstractNativeEventFilter):
    """Windows RegisterHotKey + native event filter. Silently inactive on other platforms or if the combo is taken."""

    def __init__(self, callback: Callable[[], None], combo: str = "ctrl+alt+c"):
        super().__init__()
        self.callback = callback
        self.combo = combo
        self.active = False
        if sys.platform != "win32":
            return
        parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
        mods = sum(MODS[p] for p in parts[:-1] if p in MODS) | 0x4000      # MOD_NOREPEAT
        try:
            vk = _vk(parts[-1])
            self.active = bool(ctypes.windll.user32.RegisterHotKey(None, HOTKEY_ID, mods, vk))
        except Exception as e:  # noqa: BLE001
            log.warning("global hotkey %s not registered: %s", combo, e)
        if not self.active:
            log.warning("global hotkey %s unavailable (already in use?)", combo)

    def nativeEventFilter(self, eventType, message):
        if self.active and eventType in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            try:
                msg = ctypes.wintypes.MSG.from_address(int(message))
                if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                    QTimer.singleShot(0, self.callback)
            except Exception:  # noqa: BLE001
                pass
        return False, 0

    def unregister(self) -> None:
        if self.active and sys.platform == "win32":
            ctypes.windll.user32.UnregisterHotKey(None, HOTKEY_ID)
            self.active = False


class _Bus(QObject):
    text = Signal(str)
    status = Signal(str)
    done = Signal(object)
    failed = Signal(str)


class QuickAsk(QDialog):
    """Frameless always-on-top job launcher."""
    run_finished = Signal(object)          # AgentRunResult

    def __init__(self, main_window):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.main = main_window
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setObjectName("quickAsk")
        self.setStyleSheet(f"#quickAsk {{ background: {theme.BG2}; border: 1px solid {theme.ACCENT}; border-radius: 8px; }}")
        self.setMinimumWidth(760)
        self._text = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        top = QHBoxLayout()
        title = QLabel("Ask YANG to do something")
        title.setProperty("kpiLabel", True)
        top.addWidget(title)
        top.addStretch(1)
        self.effort = QComboBox()
        self.effort.addItems(["low", "medium", "high"])
        self.effort.setCurrentText("medium")
        self.effort.setToolTip("Reasoning effort for this job")
        top.addWidget(QLabel("effort"))
        top.addWidget(self.effort)
        lay.addLayout(top)
        self.input = QLineEdit()
        self.input.setPlaceholderText("e.g. run today's harvest and tell me if anything is worth doing · build a 30-name HRP basket · explain lot 36 …  (Enter runs, Esc closes)")
        f = QFont("Segoe UI", 13)
        self.input.setFont(f)
        self.input.setMinimumHeight(40)
        self.input.returnPressed.connect(self.run)
        lay.addWidget(self.input)
        self.status = QLabel("")
        self.status.setProperty("muted", True)
        lay.addWidget(self.status)
        self.out = QTextBrowser()
        self.out.setMinimumHeight(260)
        self.out.hide()
        self.out.document().setDefaultStyleSheet(f"body{{color:{theme.TEXT}}} td,th{{border:1px solid {theme.BORDER};padding:2px 6px}}")
        lay.addWidget(self.out)
        row = QHBoxLayout()
        self.open_btn = button("Open in YANG", self._open, tooltip="Show the full conversation in the YANG tab")
        self.open_btn.hide()
        self.stop_btn = button("Stop", self._stop, danger=True)
        self.stop_btn.hide()
        row.addWidget(self.open_btn)
        row.addWidget(self.stop_btn)
        row.addStretch(1)
        row.addWidget(button("Close", self.hide))
        lay.addLayout(row)
        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._flush)
        self._dirty = False
        self._cid: int | None = None

    # ------------------------------------------------------------------ show / hide
    def popup(self) -> None:
        screen = QApplication.screenAt(self.main.frameGeometry().center()) or QApplication.primaryScreen()
        g = screen.availableGeometry()
        self.adjustSize()
        self.move(g.center().x() - self.width() // 2, g.top() + g.height() // 4)
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()
        self.input.selectAll()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        prompt = self.input.text().strip()
        if not prompt:
            return
        svc = self.main.agent_service
        if not svc.copilot.available:
            self.status.setText("ANTHROPIC_API_KEY is not set (.env).")
            return
        if svc.running_run_id is not None:
            self.status.setText("An agent job is already running; queued jobs are not supported yet, please wait.")
            return
        self._text = ""
        self.out.clear()
        self.out.show()
        self.open_btn.hide()
        self.stop_btn.show()
        self.input.setEnabled(False)
        self.status.setText("● starting…")
        self._timer.start()
        bus = _Bus()
        bus.text.connect(self._on_text)
        bus.status.connect(lambda s: self.status.setText(f"● {s}"))
        bus.done.connect(self._done)
        bus.failed.connect(self._failed)
        self._bus = bus
        eff = self.effort.currentText()
        run_task(lambda: svc.run_adhoc(prompt, trigger="popup", effort=eff, on_text=bus.text.emit, on_status=bus.status.emit),
                 on_done=lambda r: bus.done.emit(r), on_error=lambda m: bus.failed.emit(m), wants_progress=False)

    def _stop(self) -> None:
        self.main.agent_service.copilot.cancel()
        self.status.setText("● stopping…")

    def _on_text(self, t: str) -> None:
        self._text += t
        self._dirty = True

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        from .screens.copilot import md_to_html
        self.out.setHtml("<body>" + md_to_html(self._text) + "</body>")
        sb = self.out.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _done(self, res) -> None:
        self._timer.stop()
        self._flush()
        self.input.setEnabled(True)
        self.stop_btn.hide()
        self.open_btn.show()
        run = self.main.agent_service.run_detail(res.run_id)
        self._cid = run.get("conversation_id") if run else None
        cost = f"${res.turn.cost_usd:.2f}" if res.turn else ""
        extra = f" · {len(res.change_ids)} change(s) awaiting approval" if res.change_ids else ""
        self.status.setText(f"{res.status} · run #{res.run_id} {cost}{extra}")
        if res.error:
            self.out.setHtml(f"<body style='color:{theme.RED}'>{res.error}</body>")
        self.run_finished.emit(res)

    def _failed(self, msg: str) -> None:
        self._timer.stop()
        self.input.setEnabled(True)
        self.stop_btn.hide()
        self.status.setText("failed")
        self.out.setHtml(f"<body style='color:{theme.RED}'>{msg.splitlines()[0]}</body>")

    def _open(self) -> None:
        self.hide()
        self.main.show()
        self.main.raise_()
        self.main.activateWindow()
        self.main.goto("AI co-pilot")
        if self._cid is not None:
            self.main.screens["AI co-pilot"].select_conversation(self._cid)


def make_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(theme.ACCENT))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 14, 14)
    p.setPen(QColor("white"))
    p.setFont(QFont("Segoe UI", 30, QFont.Bold))
    p.drawText(pm.rect(), Qt.AlignCenter, "T")
    p.end()
    return QIcon(pm)


def make_tray(main_window, hotkey_label: str) -> QSystemTrayIcon | None:
    if not QSystemTrayIcon.isSystemTrayAvailable():
        return None
    tray = QSystemTrayIcon(make_icon(), main_window)
    tray.setToolTip("TLH Engine")
    menu = QMenu()
    a1 = QAction(f"Ask YANG…  ({hotkey_label})", menu)
    a1.triggered.connect(main_window.show_quick)
    a2 = QAction("Show TLH Engine", menu)
    a2.triggered.connect(main_window.show_main)
    a3 = QAction("Run due agent tasks now", menu)
    a3.triggered.connect(main_window.agent_tick)
    a4 = QAction("Quit", menu)
    a4.triggered.connect(QApplication.instance().quit)
    for a in (a1, a2, a3):
        menu.addAction(a)
    menu.addSeparator()
    menu.addAction(a4)
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: main_window.show_main() if reason == QSystemTrayIcon.DoubleClick else None)
    tray.show()
    return tray
