"""Main window: toolbar (entity, data refresh, status), tabbed screens, status bar."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..services.agent_service import AgentService
from ..services.context import AppContext
from ..services.data_service import DataService
from ..services.harvest_service import HarvestService
from ..services.portfolio_service import PortfolioService
from ..services.risk_service import RiskService
from .quick import GlobalHotkey, QuickAsk, make_tray
from .screens.agent import AgentScreen
from .screens.baskets import BasketsScreen
from .screens.builder import BuilderScreen
from .screens.concentration import ConcentrationScreen
from .screens.copilot import CopilotScreen
from .screens.export import ExportScreen
from .screens.harvest import HarvestScreen
from .screens.portfolio import PortfolioScreen
from .screens.risk import RiskScreen
from .screens.risk_lab import RiskLabScreen
from .screens.settings import SettingsScreen
from .screens.strategy import StrategyScreen
from .tour import TourDock
from .widgets import Banner
from .workers import run_task

log = logging.getLogger(__name__)

TAB_TITLES = {"AI co-pilot": "YANG", "Agent": "YANG Agent"}


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext | None = None):
        super().__init__()
        self.ctx = ctx or AppContext()
        self.data_service = DataService(self.ctx)
        self.portfolio_service = PortfolioService(self.ctx)
        self.risk_service = RiskService(self.ctx)
        self.harvest_service = HarvestService(self.ctx)
        self.agent_service = AgentService(self.ctx)
        self._agent_busy = False
        self.setWindowTitle("TLH Engine — Tax-Loss Harvesting")
        self.resize(1600, 960)
        self._build()
        QTimer.singleShot(50, self._startup)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addWidget(QLabel("  Tax entity "))
        self.entity = QComboBox()
        self.entity.setMinimumWidth(220)
        self.entity.currentIndexChanged.connect(self._entity_changed)
        tb.addWidget(self.entity)
        tb.addSeparator()
        act = QAction("Refresh data", self)
        act.setToolTip("Pull a new Norgate snapshot for the universe + holdings + substitutes")
        act.triggered.connect(self.refresh_data)
        tb.addAction(act)
        act2 = QAction("Reload", self)
        act2.triggered.connect(self.data_changed)
        tb.addAction(act2)
        tb.addSeparator()
        self.norgate_lbl = QLabel("  Norgate: …  ")
        self.snap_lbl = QLabel("")
        self.model_lbl = QLabel("")
        for w in (self.norgate_lbl, self.snap_lbl, self.model_lbl):
            w.setProperty("muted", True)
            tb.addWidget(w)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.banner = Banner()
        lay.addWidget(self.banner)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.screens = {
            "Portfolio": PortfolioScreen(self), "Harvest": HarvestScreen(self), "Risk model": RiskScreen(self), "Risk lab": RiskLabScreen(self), "Concentration": ConcentrationScreen(self),
            "Model portfolios": BasketsScreen(self), "Strategy lab": StrategyScreen(self), "TLH model builder": BuilderScreen(self),
            "AI co-pilot": CopilotScreen(self), "Agent": AgentScreen(self), "Export": ExportScreen(self),
            "Settings": SettingsScreen(self),
        }
        for name, s in self.screens.items():
            self.tabs.addTab(s, TAB_TITLES.get(name, name))
        self.screens["Portfolio"].data_changed.connect(self.data_changed)
        self.screens["Harvest"].data_changed.connect(self.data_changed)
        self.screens["Risk model"].model_changed.connect(self.data_changed)
        self.screens["Model portfolios"].data_changed.connect(self.data_changed)
        self.screens["Strategy lab"].data_changed.connect(self.data_changed)
        self.screens["TLH model builder"].data_changed.connect(self.data_changed)
        self.screens["Concentration"].data_changed.connect(self.data_changed)
        self.screens["AI co-pilot"].code_promoted.connect(lambda p: self.status(f"Promoted {p}; re-fit / re-run to use it."))
        self.screens["AI co-pilot"].state_changed.connect(self._refresh_except_copilot)
        self.screens["Agent"].state_changed.connect(self._badge)
        self.badge = QLabel("")
        self.badge.setProperty("muted", True)
        self.badge.setCursor(Qt.PointingHandCursor)
        self.badge.mousePressEvent = lambda ev: self.goto("Agent")
        self.statusBar().addPermanentWidget(self.badge)
        self.statusBar().showMessage("Ready")
        # ---- pop-up, hotkey, tray, scheduler
        self.quick = QuickAsk(self)
        self.quick.run_finished.connect(self._agent_finished)
        QShortcut(QKeySequence("Ctrl+Space"), self, activated=self.show_quick)
        combo = str(self.ctx.get("quick_hotkey", "ctrl+alt+c"))
        self.hotkey = GlobalHotkey(self.show_quick, combo)
        QApplication.instance().installNativeEventFilter(self.hotkey)
        self.tray = make_tray(self, combo if self.hotkey.active else "Ctrl+Space in app")
        self._sched_timer = QTimer(self)
        self._sched_timer.setInterval(30_000)
        self._sched_timer.timeout.connect(self.agent_tick)
        self._sched_timer.start()
        QTimer.singleShot(20_000, self._startup_tasks)
        # ---- interactive how-to
        self.tour = TourDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tour)
        self.tour.hide()
        help_menu = self.menuBar().addMenu("Help")
        a_tour = QAction("Interactive how-to", self)
        a_tour.setShortcut("F1")
        a_tour.triggered.connect(self.show_tour)
        help_menu.addAction(a_tour)
        a_yang = QAction("Ask YANG…  (Ctrl+Space)", self)
        a_yang.triggered.connect(self.show_quick)
        help_menu.addAction(a_yang)
        a_about = QAction("About TLH Engine / YANG", self)
        a_about.triggered.connect(lambda: QMessageBox.information(self, "TLH Engine", "Tax-Loss Harvesting Engine with YANG, the embedded quant co-pilot (built on Claude). "
                                                                                       "Decision support only: nothing places orders. See DECISIONS.md for conventions."))
        help_menu.addAction(a_about)

    # ------------------------------------------------------------------ startup
    def _startup(self) -> None:
        self.reload_entities()
        ok = self.data_service.norgate_ok()
        self.norgate_lbl.setText("  Norgate: ● running  " if ok else "  Norgate: ○ NDU not running  ")
        self.norgate_lbl.setStyleSheet(f"color: {'#22C55E' if ok else '#EF4444'}")
        if not ok:
            self.banner.show_msg("Norgate Data Updater (NDU) is not running. Prices, snapshots and model fits are unavailable until it is started. "
                                 "Saved snapshots, runs and lots remain viewable.", "error")
        self._labels()
        self.data_changed()
        if not self.ctx.get("tour_seen", False):
            self.show_tour()
        snap = self.data_service.latest_snapshot()
        if ok and (snap is None or not self.data_service.snapshot_is_current(snap, self.ctx.current_entity_id)):
            self.status("Snapshot is stale or missing; pulling fresh data…")
            self.refresh_data()

    def _labels(self) -> None:
        snap = self.data_service.latest_snapshot()
        self.snap_lbl.setText(f"  snapshot {snap.id} (as of {snap.as_of_date})  " if snap else "  no snapshot  ")
        act = self.ctx.models.active()
        self.model_lbl.setText(f"  model #{act['id']} ({act['as_of_date']})  " if act else "  no risk model  ")

    def reload_entities(self) -> None:
        self.entity.blockSignals(True)
        self.entity.clear()
        for e in self.ctx.entities.list():
            self.entity.addItem(e["name"], e["id"])
        cur = self.ctx.current_entity_id
        if cur is not None:
            i = self.entity.findData(cur)
            if i >= 0:
                self.entity.setCurrentIndex(i)
        self.entity.blockSignals(False)

    def _entity_changed(self, i: int) -> None:
        eid = self.entity.currentData()
        if eid is not None and eid != self.ctx.current_entity_id:
            self.ctx.current_entity_id = eid
            self.data_changed()

    # ------------------------------------------------------------------ cross-screen
    def data_changed(self) -> None:
        self._labels()
        self._badge()
        if getattr(self, "tour", None) is not None and self.tour.isVisible():
            self.tour.refresh()
        for s in self.screens.values():
            try:
                s.refresh()
            except Exception as e:  # never let one screen kill the refresh
                log.exception("refresh failed for %s", type(s).__name__)
                self.status(f"{type(s).__name__} refresh failed: {e}")

    def _refresh_except_copilot(self) -> None:
        """After a co-pilot turn (which may have created baskets, runs, models) refresh the other screens."""
        self._labels()
        for name, s in self.screens.items():
            if name == "AI co-pilot":
                continue
            try:
                s.refresh()
            except Exception as e:
                log.exception("refresh failed for %s", type(s).__name__)
                self.status(f"{type(s).__name__} refresh failed: {e}")

    def refresh_data(self) -> None:
        if getattr(self, "_pulling", False):
            self.status("A data refresh is already running; please wait.")
            return
        if not self.data_service.norgate_ok():
            QMessageBox.warning(self, "Norgate", "NDU is not running.")
            return
        self._pulling = True
        self.status("Pulling data snapshot…")

        def done(s):
            self._pulling = False
            self.status(f"Snapshot {s.id} ready.")
            self.banner.clear_msg()
            self._auto_fit()

        def fail(msg):
            self._pulling = False
            self.error(msg)

        run_task(self.data_service.ensure_snapshot, self.ctx.current_entity_id, True, on_done=done, on_error=fail, on_progress=self.status)

    def _auto_fit(self) -> None:
        """After a fresh snapshot: refit the risk model automatically if none exists yet."""
        if self.ctx.models.active() is None:
            self.screens["Risk model"].fit()
        else:
            self.data_changed()

    # ------------------------------------------------------------------ agent / pop-up
    def show_quick(self) -> None:
        self.quick.popup()

    def reload_ai_settings(self) -> None:
        """Re-read .env and reconfigure every Copilot instance (YANG tab, pop-up/agent)."""
        from ..config import get_settings
        self.ctx.settings = get_settings(reload=True)
        for cp in {self.screens["AI co-pilot"].copilot, self.agent_service.copilot}:
            cp.reconfigure()
        self.screens["AI co-pilot"].refresh()

    def show_tour(self) -> None:
        self.tour.show()
        self.tour.raise_()
        self.tour.refresh()

    def show_main(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _badge(self) -> None:
        try:
            unread = self.agent_service.unread()
            pend = self.agent_service.pending_changes()
        except Exception:
            return
        parts = []
        if unread:
            parts.append(f"{unread} agent report{'s' if unread != 1 else ''} unread")
        if pend:
            parts.append(f"{pend} change{'s' if pend != 1 else ''} awaiting approval")
        self.badge.setText(("  ●  " + " · ".join(parts) + "  ") if parts else "")
        self.badge.setStyleSheet("color: #F59E0B;" if parts else "")

    def agent_tick(self) -> None:
        """Scheduler heartbeat: run the first due task if nothing is running."""
        if self._agent_busy or self.agent_service.running_run_id is not None:
            return
        try:
            due = self.agent_service.due_tasks()
        except Exception as e:
            log.warning("agent scheduler: %s", e)
            return
        if due:
            self.run_agent_task(int(due[0]["id"]), trigger="schedule")

    def _startup_tasks(self) -> None:
        for t in self.agent_service.startup_tasks():
            if not self._agent_busy:
                self.run_agent_task(int(t["id"]), trigger="startup")

    def run_agent_task(self, task_id: int, trigger: str = "manual") -> None:
        if self._agent_busy:
            return
        self._agent_busy = True
        agent_screen = self.screens["Agent"]
        buf = {"t": ""}

        def on_text(t: str) -> None:
            buf["t"] += t

        self.status(f"Agent task #{task_id} running ({trigger})…")
        live = QTimer(self)
        live.setInterval(500)
        live.timeout.connect(lambda: agent_screen.show_live(buf["t"]) if buf["t"] else None)
        live.start()

        def done(res):
            live.stop()
            self._agent_busy = False
            self._agent_finished(res)

        def fail(msg):
            live.stop()
            self._agent_busy = False
            self.status(f"Agent task failed: {msg.splitlines()[0][:120]}")
            self.screens["Agent"].refresh()

        run_task(self.agent_service.run_task, task_id, trigger, on_text, on_done=done, on_error=fail, wants_progress=False)

    def _agent_finished(self, res) -> None:
        name = (self.agent_service.run_detail(res.run_id) or {}).get("name", "agent task")
        cost = f" (${res.turn.cost_usd:.2f})" if getattr(res, "turn", None) else ""
        msg = f"{name}: {res.status}{cost}" + (f" · {len(res.change_ids)} change(s) to approve" if res.change_ids else "")
        self.status(msg)
        if getattr(self, "tray", None):
            self.tray.showMessage("TLH Engine · YANG finished", msg + ("\n" + (res.report or "")[:180] if res.report else ""))
        self._badge()
        self._refresh_except_copilot()
        self.screens["AI co-pilot"].refresh()

    def closeEvent(self, ev) -> None:
        try:
            self.hotkey.unregister()
            if getattr(self, "tray", None):
                self.tray.hide()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(ev)

    def goto(self, name: str) -> None:
        self.tabs.setCurrentWidget(self.screens[name])

    def status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def error(self, msg: str) -> None:
        log.error(msg)
        self.statusBar().showMessage(msg.splitlines()[0][:200])
        QMessageBox.critical(self, "Error", msg[:4000])
