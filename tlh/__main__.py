"""Entry point: `python -m tlh [--seed-demo] [--fit] [--harvest] [--no-gui]`."""
from __future__ import annotations

import argparse
import logging
import sys
import warnings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tlh", description="Tax-Loss Harvesting Engine")
    p.add_argument("--seed-demo", action="store_true", help="seed the demo household (idempotent)")
    p.add_argument("--fit", action="store_true", help="fit a risk model on the latest snapshot before launching")
    p.add_argument("--harvest", action="store_true", help="run a harvest for the current entity and print the summary")
    p.add_argument("--no-gui", action="store_true", help="do the requested actions and exit without the GUI")
    p.add_argument("--run-task", metavar="NAME", help="run one agent task headless (for Windows Task Scheduler) and exit")
    p.add_argument("--agent-loop", action="store_true", help="run the agent scheduler headless until Ctrl+C")
    p.add_argument("--expert", action="store_true", help="start in expert mode (all tabs) regardless of the saved UI mode")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("norgatedata").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.ERROR)   # Norgate's parallel pulls spam "pool is full"
    warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

    from .config import get_settings
    settings = get_settings()
    settings.ensure_dirs()
    fh = logging.FileHandler(settings.logs_dir / "tlh.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    if a.seed_demo or a.fit or a.harvest:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        from .services.context import AppContext
        ctx = AppContext(settings)
        if a.seed_demo:
            from .services.demo import seed_demo
            seed_demo(ctx, progress=print)
        if a.fit:
            from .services.data_service import DataService
            from .services.risk_service import RiskService
            snap = DataService(ctx).ensure_snapshot(ctx.current_entity_id, progress=print)
            mid, m = RiskService(ctx).fit(snap, progress=print)
            print(f"model #{mid}: {m.diagnostics}")
        if a.harvest:
            from .services.harvest_service import HarvestService
            rid, res = HarvestService(ctx).run(ctx.current_entity_id)
            print(f"run #{rid}: {res.summary}")
            print(res.trades.to_string())
        ctx.close()
    if a.run_task or a.agent_loop:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        from .services.agent_service import AgentService
        from .services.context import AppContext
        ctx = AppContext(settings)
        svc = AgentService(ctx)
        if a.run_task:
            t = ctx.agent.task_by_name(a.run_task)
            if not t:
                print(f"no agent task named {a.run_task!r}; known: {ctx.agent.tasks()['name'].tolist() if not ctx.agent.tasks().empty else []}")
                return 2
            res = svc.run_task(int(t["id"]), trigger="cli", on_status=lambda m: print(f"  [{m}]"))
            print(f"run #{res.run_id} {res.status}\n{res.report or res.error}")
            return 0 if res.status == "done" else 1
        import time as _time
        print("agent loop running; Ctrl+C to stop")
        try:
            while True:
                for t in svc.due_tasks():
                    print(f"running due task: {t['name']}")
                    res = svc.run_task(int(t["id"]), trigger="schedule")
                    print(f"  -> #{res.run_id} {res.status}")
                _time.sleep(30)
        except KeyboardInterrupt:
            return 0
    if a.no_gui:
        return 0

    # ---- GUI: show a splash immediately, then import the heavy modules behind it ---------------------
    from PySide6.QtCore import Qt
    from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401  (must import before QApplication)
    from PySide6.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("TLH Engine")
    from .gui.splash import make_splash
    splash = make_splash()
    splash.show()
    splash.showMessage("Loading engine…")
    app.processEvents()
    from .gui.app import MainWindow
    from .gui.theme import apply_theme
    apply_theme(app)
    splash.showMessage("Opening workspace…")
    app.processEvents()
    win = MainWindow(expert=a.expert)
    win.show()
    splash.finish(win)
    from .lazy import warm_up
    warm_up()                       # solver imports in the background so the first optimisation is instant
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
