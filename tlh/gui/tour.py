"""Interactive how-to: a dockable guided tour with completion tracking, "Show me" highlighting, "Do it" actions,
and an "Ask YANG" shortcut per step."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .widgets import button


@dataclass
class Step:
    key: str
    title: str
    body: str
    tab: str | None
    target: Callable | None = None       # win -> QWidget to highlight
    action: Callable | None = None       # win -> None, performs the step
    done: Callable | None = None         # win -> bool
    ask: str = ""                        # question to hand to YANG


def _steps() -> list[Step]:
    return [
        Step("start", "0 · Start here (three clicks)", """
**Start here** is the advisor view: **1** load holdings (import a broker CSV/Excel or use the demo), **2** pick the client's state,
filing status and income (every state's capital-gains rule is built in), **3** press **Find my tax savings now**.

You get plain-English sentences, the tax value in dollars, the tickets, and a wealth chart. Switch to **expert mode** (View menu)
for the quant workbenches; the rest of this how-to walks through them.""",
             "Home", lambda w: w.screens["Home"].go_btn, None, lambda w: w.ctx.get("tax_state", "") != "",
             "Explain in plain English what this app does for an advisor and what the three steps are."),
        Step("data", "1 · Pull market data", """
**What:** the engine works from an immutable Norgate *snapshot* (prices, GICS, fundamentals, macro, index membership).

**How:** make sure Norgate Data Updater is running, then click **Refresh data** in the toolbar. About 20 seconds for ~640 names / 10 years.

Every run and model fit records the snapshot it used, so past results stay reproducible.""",
             None, lambda w: w.snap_lbl, lambda w: w.refresh_data(), lambda w: w.data_service.latest_snapshot() is not None,
             "What does a data snapshot contain and why is it the reproducibility unit?"),
        Step("demo", "2 · Load a portfolio (or the demo)", """
**What:** a *tax entity* (household) holds accounts; wash-sale rules are checked across all of them, including IRAs.

**How:** Settings → *Seed demo household* gives you three accounts with real lots, a recent loss sale and a scheduled DRIP.
For your own book: add an entity and accounts, then record buys or import a CSV (account, symbol, date, quantity, price).""",
             "Settings", lambda w: w.screens["Settings"].accounts, lambda w: w.screens["Settings"]._seed(),
             lambda w: bool(w.ctx.entities.list()), "Explain how accounts, entities and wash-sale scope relate."),
        Step("model", "3 · Fit the risk model", """
**What:** a Barra-style factor model (market, six styles, GICS sectors, optional macro) gives factor exposures, a covariance matrix and tracking error.

**How:** Risk model → **Fit model on latest snapshot** (about 3 s). Each fit is a versioned artifact; compare versions any time.""",
             "Risk model", lambda w: w.screens["Risk model"].fit_btn, lambda w: w.screens["Risk model"].fit(),
             lambda w: w.ctx.models.active() is not None, "Summarise the active risk model and what drives my tracking error."),
        Step("library", "3b · Pick a model from the library, or calibrate one", """
**Risk model › Model library** holds 13 presets: ERM standard / short / long horizon / robust / GARCH-dynamic / regime-conditional, hybrid
ERM + statistical, Potomac calibrated covariances (126d equal Ledoit-Wolf; 189d exponential), tight-pair sample, PCA, barra_lite.

**Risk lab › Calibration** re-runs the 2026 calibration study on your snapshot (lookback × weighting × estimator × horizon) and can
fit the winner in one click; the substitute-pair study shows why near-identical ETF pairs need the unshrunk sample matrix.""",
             "Risk model", lambda w: w.screens["Risk model"].preset, None,
             lambda w: len(w.ctx.models.list()) >= 2, "Which risk model preset should I use for a 3-month harvest horizon, and why?"),
        Step("portfolio", "4 · Read the portfolio", """
**Lots** shows every tax lot with unrealised P&L, ST/LT term, days to long-term and a **wash status**. Select a lot to read the plain-English wash-sale determination.

**Positions** aggregates by symbol with a treemap and a harvest-opportunity heatmap. **Wash calendar** draws every open 61-day window. **Realised** shows Schedule D netting and carryforward.""",
             "Portfolio", lambda w: w.screens["Portfolio"].lots_table, None,
             lambda w: w.ctx.current_entity_id is not None and bool(w.ctx.portfolio.held_symbols(w.ctx.current_entity_id)),
             "Which of my lots are wash-blocked today and why?"),
        Step("harvest", "5 · Run a harvest", """
**What:** the optimizer proposes wash-safe sells and correlated replacements that maximise after-tax loss while holding tracking error and factor drift within budgets.

**How:** drag the **constraint hierarchy** into the order you want (tax alpha / TE / factor neutrality), set budgets, click **Run harvest**. Every trade carries its wash explanation; blocked lots list their reasons; *Frontier* and *Compare priorities* show the trade-offs.""",
             "Harvest", lambda w: w.screens["Harvest"].run_btn, lambda w: w.screens["Harvest"].run(),
             lambda w: not w.ctx.runs.list(limit=200).query("run_type == 'harvest'").empty if not w.ctx.runs.list(limit=200).empty else False,
             "Explain the latest harvest recommendation trade by trade."),
        Step("basket", "6 · Build a model portfolio", """
**What:** a *basket* is a saved target portfolio. Baskets can be the **benchmark**, so a full-rebalance harvest migrates the book toward them while harvesting losses.

**How:** Model portfolios → set name, max names, sector band, optional style tilts → **Build basket**. Then *Set as benchmark*.""",
             "Model portfolios", lambda w: w.screens["Model portfolios"].build_btn, None,
             lambda w: not w.ctx.baskets.list().empty, "Build a 40-name quality-tilted basket excluding my current holdings."),
        Step("samples", "6a · Build the sample model portfolios", """
**Model portfolios › Sample model portfolios** builds 17 ready-made recipes in one click: 50/100-name index trackers, integrated
multi-factor (value + momentum + quality + low-vol), defensive equity with a beta cap, quality-momentum, min-variance, max-diversification,
risk parity, HRP, value / low-vol / growth tilts, min-CVaR, Black-Litterman, and 130/30 and 145/45 long/short tax engines.

Each is a normal saved basket: set one as the benchmark and harvest toward it.""",
             "Model portfolios", lambda w: w.screens["Model portfolios"].lib_btn, lambda w: w.screens["Model portfolios"].build_library(),
             lambda w: (not w.ctx.baskets.list().empty) and w.ctx.baskets.list()["name"].str.startswith("Sample").any(),
             "Which sample model portfolio fits a retiree who wants equity exposure with a smaller drawdown, and why?"),
        Step("concentration", "6b · Plan around embedded gains", """
**Concentration** shows how locked-in the book is (embedded gains, tax if liquidated, risk share, effective N) and plans the way out:
a bracket-aware multi-year **glide path** that uses harvested losses and carryforwards, honours gain budgets and step-up odds; **Monte Carlo**
of hold vs sell vs schedule; **collars** with constructive-sale and straddle flags; charitable, gifting and exchange-fund alternatives; and a
one-click **gain-offset trade plan** paired with wash-safe losses.""",
             "Concentration", lambda w: w.screens["Concentration"].plan_btn, lambda w: w.screens["Concentration"].run_plan(),
             lambda w: w.ctx.current_entity_id is not None and bool(w.ctx.portfolio.held_symbols(w.ctx.current_entity_id)),
             "Which of my positions is most concentrated after tax, and what is the cheapest 5-year path to cut it to 5%?"),
        Step("strategy", "7 · Try a construction strategy and backtest it", """
**Strategy lab** offers twelve methods: min-variance, max-diversification, risk parity, HRP, mean-variance with signal alphas, Black-Litterman with views, min-CVaR, stratified indexing, factor tilts and tax-aware transition.

**How:** pick a method, tune it, **Build basket**; then **Run backtest** for a walk-forward test with costs. Read the caveats (survivorship, fundamental look-ahead).""",
             "Strategy lab", lambda w: w.screens["Strategy lab"].build_btn, None,
             lambda w: not w.ctx.runs.list(limit=500).query("run_type == 'backtest'").empty if not w.ctx.runs.list(limit=500).empty else False,
             "Compare risk parity and minimum variance on my universe with a quarterly backtest since 2022."),
        Step("builder", "8 · Design a TLH model visually", """
**TLH model builder**: drag blocks (Universe → Filter → Rank → Benchmark → Construction → Transition → Harvest → Save/export) onto the canvas, order them left to right, edit parameters on the right, **Run model**.

Start from an example, save yours, export JSON, or click *Ask YANG to design…* and describe it in words.""",
             "TLH model builder", lambda w: w.screens["TLH model builder"].run_btn,
             lambda w: w.screens["TLH model builder"].examples.setCurrentIndex(1),
             lambda w: not w.ctx.pipelines.list().empty, "Design a TLH model pipeline that screens to liquid quality names and harvests toward a 40-name core."),
        Step("yang", "9 · Ask YANG", """
**YANG** is the embedded co-pilot. It reads the live portfolio, model and runs; runs harvests, builds baskets and strategies, evaluates hand-designed trade plans, fits model variants, and proposes code changes that you approve.

**How:** press **Ctrl+Space** (or **Ctrl+Alt+C** from anywhere on Windows) and type a job, or use the YANG tab for a full conversation.""",
             "AI co-pilot", lambda w: w.screens["AI co-pilot"].input, lambda w: w.show_quick(),
             lambda w: not w.ctx.conversations.list().empty, "What can you do for me in this app? Give me five concrete jobs."),
        Step("agent", "10 · Let YANG work unattended", """
**YANG Agent** runs jobs on a schedule (daily harvest scan, wash-window watch, weekly model health, …) and files reports; you get a tray balloon and a status-bar badge.

**How:** YANG Agent → **Install templates**, select one, **Enable / disable**. Headless: `python -m tlh --run-task "Daily harvest scan"`.""",
             "Agent", lambda w: w.screens["Agent"].tasks, lambda w: w.screens["Agent"]._templates(),
             lambda w: not w.ctx.agent.tasks().empty, "Which unattended tasks would you recommend for a weekly TLH routine?"),
        Step("export", "11 · Export the trade ticket", """
**Export** builds a formatted workbook per run: summary, paper trade ticket, trades with wash explanations, blocked lots, replacements, before/after exposures, TE decomposition, sectors, positions.

Nothing is ever sent to a broker; the ticket is the hand-off.""",
             "Export", lambda w: w.screens["Export"].run, lambda w: w.screens["Export"].export(),
             lambda w: any(w.ctx.settings.exports_dir.glob("*.xlsx")), "Walk me through the sheets in the export workbook."),
        Step("tax", "12 · Check the tax assumptions", """
Settings holds the marginal rates (federal ST/LT, state, NIIT), filing status, ordinary-income offset and prior-year carryforwards that drive every after-tax number.

The **presumed-identical** toggle decides whether same-index ETFs from different issuers count as substantially identical (conservative default: yes).""",
             "Settings", lambda w: w.screens["Settings"].fed_st, None, lambda w: True,
             "Explain how the after-tax benefit and tax alpha of a harvested loss are computed."),
        Step("tactical", "7b · Levered beta and the tactical overlay", """
**Tactical overlay** turns a Potomac signal into a target beta (0 to 1.5) and reaches it with leveraged or inverse S&P ETFs on top of the
core, never selling stock, never with futures or shorts, inside the Reg-T / house-maintenance margin policy you set.

**How:** save a signal (manual beta, a Potomac strategy CSV, an example rule, or a blend), **Recommend overlay** for today's ticket,
**Backtest active signal** for the day-by-day simulation. The **levered-beta model** builds an S&P stock + 2x/3x ETF basket at 1.5 beta.""",
             "Tactical overlay", lambda w: w.screens["Tactical overlay"].lb_btn, None,
             lambda w: bool(w.ctx.get("tactical_signals")), "Size today's overlay to take the household to a 1.5 beta and tell me the margin risk."),
        Step("research", "7c · Research lab: defend every parameter", """
**TLH research** backtests the harvesting rules on the point-in-time S&P 500 over rolling 5- or 10-year windows starting every
calendar year since 2000, monthly, with whole shares and wash windows. Sweep account size, basket size, harvest trigger, approach
(pairs within sector, pairs within index, SARD twin baskets, TE optimizer) and concentrated starts (position size x embedded gain).
Read losses harvested per year, harvest life before ossification, and tracking error as medians with interquartile ranges.

**How:** build the store once (step 1), set the base case, tick the sweeps, **Estimate**, **Run study**. Results resume if interrupted.
**Export write-up** produces the due-diligence workbook and markdown.""",
             "TLH research", lambda w: w.screens["TLH research"].build_btn, None,
             lambda w: bool(w.screens["TLH research"].svc.list_studies()) if "TLH research" in w.screens else False,
             "Run a quick research study over triggers and basket sizes and tell me which trigger harvests the most for a 2% tracking error."),
        Step("states", "12b · Every state's capital-gains rules", """
**Tax rates** maps all 50 states + DC: how each taxes short- and long-term gains (ordinary rates, exclusions, flat capital-gains rates,
Washington's excise), combined with federal brackets and NIIT at the client's income. Click a state on the map, read the plain-English
explanation, and press **Use for this household** to set the marginal rates the engine values every loss at.

Figures are approximate planning values; verify before filing.""",
             "Tax rates", lambda w: w.screens["Tax rates"].map, None, lambda w: w.ctx.get("tax_state", "") != "",
             "Compare what a $50,000 short-term loss is worth to a client in California, Texas and Massachusetts."),
    ]


class TourDock(QDockWidget):
    def __init__(self, win):
        super().__init__("Interactive how-to", win)
        self.win = win
        self.steps = _steps()
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Vertical)
        self.list = QListWidget()
        for s in self.steps:
            self.list.addItem(QListWidgetItem(s.title))
        self.list.currentRowChanged.connect(self._show)
        split.addWidget(self.list)
        self.text = QTextBrowser()
        self.text.document().setDefaultStyleSheet(f"body{{color:{theme.TEXT}}} code{{color:{theme.ACCENT2}}}")
        split.addWidget(self.text)
        split.setSizes([300, 400])
        lay.addWidget(split, 1)
        row = QHBoxLayout()
        self.show_btn = button("Show me", self._show_me, tooltip="Switch to the screen and highlight the control")
        self.do_btn = button("Do it", self._do_it, success=True, tooltip="Perform this step now")
        self.ask_btn = button("Ask YANG", self._ask)
        row.addWidget(self.show_btn)
        row.addWidget(self.do_btn)
        row.addWidget(self.ask_btn)
        lay.addLayout(row)
        nav = QHBoxLayout()
        nav.addWidget(button("◀ Prev", lambda: self.list.setCurrentRow(max(self.list.currentRow() - 1, 0))))
        nav.addWidget(button("Next ▶", lambda: self.list.setCurrentRow(min(self.list.currentRow() + 1, len(self.steps) - 1))))
        nav.addStretch(1)
        self.startup = QCheckBox("show at startup")
        self.startup.setChecked(not bool(win.ctx.get("tour_seen", False)))
        self.startup.toggled.connect(lambda v: win.ctx.set("tour_seen", not v))
        nav.addWidget(self.startup)
        lay.addLayout(nav)
        self.progress = QLabel("")
        self.progress.setProperty("muted", True)
        lay.addWidget(self.progress)
        self.setWidget(body)
        self.setMinimumWidth(380)
        self.list.setCurrentRow(0)

    # ------------------------------------------------------------------ state
    def refresh(self) -> None:
        done = 0
        for i, s in enumerate(self.steps):
            ok = False
            try:
                ok = bool(s.done(self.win)) if s.done else False
            except Exception:  # noqa: BLE001
                ok = False
            done += ok
            it = self.list.item(i)
            it.setText(("✓ " if ok else "○ ") + s.title)
            it.setForeground(Qt.green if ok else Qt.white)
        self.progress.setText(f"{done} of {len(self.steps)} steps done")
        if self.isVisible() and done < len(self.steps) and self.list.currentRow() < 0:
            self.list.setCurrentRow(next(i for i, s in enumerate(self.steps) if not (s.done and s.done(self.win))))

    def _show(self, row: int) -> None:
        if row < 0:
            return
        s = self.steps[row]
        self.text.setMarkdown(s.body.strip())
        self.do_btn.setEnabled(s.action is not None)

    def _cur(self) -> Step | None:
        r = self.list.currentRow()
        return self.steps[r] if r >= 0 else None

    def _show_me(self) -> None:
        s = self._cur()
        if not s:
            return
        if s.tab:
            self.win.goto(s.tab)
        if s.target:
            try:
                w = s.target(self.win)
                highlight(w)
            except Exception:  # noqa: BLE001
                pass

    def _do_it(self) -> None:
        s = self._cur()
        if not s or not s.action:
            return
        self._show_me()
        try:
            s.action(self.win)
        except Exception as e:  # noqa: BLE001
            self.win.status(f"Step failed: {e}")
        QTimer.singleShot(3000, self.refresh)

    def _ask(self) -> None:
        s = self._cur()
        if not s:
            return
        self.win.show_quick()
        self.win.quick.input.setText(s.ask)


def highlight(widget, ms: int = 2500) -> None:
    """Flash an amber outline on a widget."""
    if widget is None:
        return
    old = widget.styleSheet()
    widget.setStyleSheet(old + f"; border: 2px solid {theme.AMBER}; border-radius: 4px;")
    QTimer.singleShot(ms, lambda: widget.setStyleSheet(old))
