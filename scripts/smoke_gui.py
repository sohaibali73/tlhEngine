"""Offscreen GUI smoke: builds every screen, runs a harvest, exercises the copilot panel rendering (no API call).
Run: set QT_QPA_PLATFORM=offscreen && python scripts/smoke_gui.py"""
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --no-sandbox")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: E402, F401
from PySide6.QtWidgets import QApplication  # noqa: E402

QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
app = QApplication(sys.argv[:1])
from tlh.gui.app import MainWindow  # noqa: E402
from tlh.gui.theme import apply_theme  # noqa: E402

apply_theme(app)
errors = []
win = MainWindow()
win.error = lambda msg: errors.append(msg.splitlines()[0])
win.show()


def pump(secs):
    t = time.time()
    while time.time() - t < secs:
        app.processEvents()
        time.sleep(0.02)


pump(6)
print("tabs:", [win.tabs.tabText(i) for i in range(win.tabs.count())])
b = win.screens["Model portfolios"]
print("baskets listed:", b.table.model.rowCount())
if b.table.model.rowCount():
    b.table.view.selectRow(0)
    pump(3)
    print("basket KPIs:", b.k_name.val.text(), b.k_te.val.text(), b.k_n.val.text(), "members:", b.members.model.rowCount())
h = win.screens["Harvest"]
print("harvest bench options:", [h.bench.itemText(i) for i in range(h.bench.count())][-3:])
h.run()
pump(8)
print("harvest:", h.k_loss.val.text(), "|", h.k_te.val.text(), "| runs:", h.runs.model.rowCount())
c = win.screens["AI co-pilot"]
c.transcript.add({"kind": "user", "text": "test"})
c.transcript.add({"kind": "thinking", "text": "considering the portfolio"})
c.transcript.add({"kind": "tool", "name": "get_run", "args": {"run_id": 1}, "status": "done", "ok": True, "seconds": 0.4, "result": "{...}"})
c.transcript.add({"kind": "assistant", "text": "**Bold** and a table:\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"})
pump(0.5)
print("copilot transcript chars:", len(c.transcript.toPlainText()), "| available:", c.copilot.available, "| convs:", c.conv.count())
c._render_history()
pump(0.5)
print("history items:", len(c.transcript.items))
# ---- flagship screens
home = win.screens["Home"]
pump(3)
print("home KPIs:", home.k_value.val.text(), "|", home.k_loss.val.text(), "|", home.k_tax.val.text(), "|", home.k_state.val.text())
i = home.state.findData("CA")
home.state.setCurrentIndex(i)
home.filing.setCurrentText("mfj")
home.income.setValue(400000)
home._save_tax()
pump(0.5)
print("home tax sentence:", home.step2.text()[:140])
tr = win.screens["Tax rates"]
pump(3)
print("tax rates rows:", tr.table.model.rowCount(), "| ST/LT:", tr.k_st.val.text(), tr.k_lt.val.text())
rm = win.screens["Risk model"]
print("presets in library combo:", rm.preset.count() - 1)
rm.preset.setCurrentIndex(rm.preset.findData("Potomac Calibrated · 126d equal Ledoit-Wolf"))
print("spec kind after preset:", rm.kind.currentText(), rm.stat_lookback.value(), rm.stat_estimator.currentText())
rl = win.screens["Risk lab"]
print("risk lab tabs:", [rl.tabs.tabText(i) for i in range(rl.tabs.count())])
tac = win.screens["Tactical overlay"]
pump(2)
print("tactical signals:", tac.signals.model.rowCount(), "| instruments:", tac.inst.model.rowCount(), "| target today:", tac.k_target.val.text())
tac.use_override.setChecked(True)
tac.override.setValue(1.5)
tac.recommend()
pump(4)
print("tactical ticket:", tac.k_ticket.val.text(), "|", tac.k_margin.val.text(), "|", tac.k_cost.val.text())
print("ui mode:", win.ui_mode, "| visible tabs:", [win.tabs.tabText(i) for i in range(win.tabs.count()) if win.tabs.isTabVisible(i)])
win.set_ui_mode("simple")
print("simple-mode tabs:", [win.tabs.tabText(i) for i in range(win.tabs.count()) if win.tabs.isTabVisible(i)])
win.set_ui_mode("expert")
for name in win.screens.names():
    scr = win.screens[name]
    tabs = getattr(scr, "tabs", None)
    if tabs is not None:
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            app.processEvents()
pump(1)
print("GUI errors:", errors if errors else "none")
win.close()
app.quit()
pump(0.5)
