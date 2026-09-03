"""Start here: three steps anyone can follow (load holdings, set taxes, find savings) plus plain-English results."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...services.demo import seed_demo
from ...services.home_service import HomeService
from ...tax.state_rates import all_states
from .. import charts, theme
from ..widgets import FrameTable, KpiCard, TextPanel, button, hbox, money, pct
from ..workers import run_task


def _step_card(number: str, title: str, blurb: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setProperty("card", True)
    lay = QVBoxLayout(card)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(8)
    n = QLabel(number)
    n.setStyleSheet(f"font-size: 26px; font-weight: 800; color: {theme.ACCENT};")
    t = QLabel(title)
    t.setProperty("h1", True)
    b = QLabel(blurb)
    b.setWordWrap(True)
    b.setProperty("muted", True)
    lay.addWidget(n)
    lay.addWidget(t)
    lay.addWidget(b)
    return card, lay


class HomeScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.svc = HomeService(self.ctx)
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        w = QWidget()
        scroll.setWidget(w)
        root = QVBoxLayout(w)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        top = QHBoxLayout()
        title = QLabel("Find the tax savings hiding in a portfolio — in three clicks.")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        top.addWidget(title, 1)
        self.mode_btn = button("Switch to expert mode", self._toggle_mode)
        top.addWidget(self.mode_btn)
        top.addWidget(button("Ask YANG…", self.app.show_quick, primary=True, tooltip="Ctrl+Space anywhere in the app, Ctrl+Alt+C anywhere on the desktop"))
        root.addLayout(top)

        # KPI row
        k = QHBoxLayout()
        self.k_value = KpiCard("Portfolio value")
        self.k_loss = KpiCard("Losses available today")
        self.k_tax = KpiCard("Tax you could save now")
        self.k_ytd = KpiCard("Saved so far this year")
        self.k_te = KpiCard("Tracking error")
        self.k_state = KpiCard("Your marginal rates")
        for c in (self.k_value, self.k_loss, self.k_tax, self.k_ytd, self.k_te, self.k_state):
            k.addWidget(c)
        root.addLayout(k)

        # three steps
        grid = QGridLayout()
        grid.setSpacing(10)
        c1, l1 = _step_card("1", "Load the holdings", "Drop in a broker export (Schwab, Fidelity, IBKR, TradeStation, Vanguard or any CSV/Excel with symbol, quantity, cost and date). "
                                                   "Or press the demo button to see everything working on a realistic household.")
        l1.addWidget(button("Import broker file…", self._import, primary=True))
        l1.addWidget(button("Use the demo portfolio", self._demo))
        l1.addWidget(button("Add lots by hand (Portfolio tab)", lambda: self.app.goto("Portfolio")))
        self.step1 = QLabel("")
        self.step1.setProperty("muted", True)
        self.step1.setWordWrap(True)
        l1.addWidget(self.step1)
        l1.addStretch(1)

        c2, l2 = _step_card("2", "Tell us about the taxes", "State, filing status and other income set the marginal rates every dollar of loss is worth. "
                                                    "Federal brackets, NIIT and every state's capital-gains rules are built in.")
        f = QFormLayout()
        self.state = QComboBox()
        self.state.addItem("— select state —", "")
        for s in sorted(all_states().values(), key=lambda x: x.name):
            self.state.addItem(f"{s.name} ({s.abbrev})", s.abbrev)
        self.filing = QComboBox()
        self.filing.addItems(["single", "mfj", "mfs", "hoh"])
        self.income = QDoubleSpinBox()
        self.income.setRange(0, 1e9)
        self.income.setDecimals(0)
        self.income.setSingleStep(10_000)
        self.income.setPrefix("$ ")
        f.addRow("State", self.state)
        f.addRow("Filing status", self.filing)
        f.addRow("Other taxable income", self.income)
        l2.addLayout(f)
        l2.addWidget(button("Save tax settings", self._save_tax, primary=True))
        self.step2 = QLabel("")
        self.step2.setProperty("muted", True)
        self.step2.setWordWrap(True)
        l2.addWidget(self.step2)
        l2.addWidget(button("See every state's rates", lambda: self.app.goto("Tax rates")))
        l2.addStretch(1)

        c3, l3 = _step_card("3", "Find my tax savings", "One click: refresh market data, fit the risk model if needed, and run the wash-safe harvest. "
                                                 "You get trade tickets, a plain-English summary and the tax value.")
        self.go_btn = button("Find my tax savings now", self._one_click, primary=True)
        self.go_btn.setMinimumHeight(44)
        self.go_btn.setStyleSheet(f"font-size: 15px; background: {theme.GREEN}; border-color: {theme.GREEN};")
        l3.addWidget(self.go_btn)
        self.progress = QLabel("")
        self.progress.setProperty("muted", True)
        self.progress.setWordWrap(True)
        l3.addWidget(self.progress)
        l3.addWidget(hbox(button("Open the trade tickets", lambda: self.app.goto("Harvest")), button("Export to Excel", lambda: self.app.goto("Export"))))
        l3.addStretch(1)
        for i, c in enumerate((c1, c2, c3)):
            grid.addWidget(c, 0, i)
        root.addLayout(grid)

        # results
        split = QSplitter(Qt.Horizontal)
        self.result = TextPanel("What we found")
        self.result.set_text("Press the green button. The summary appears here in plain English, and the charts fill in.")
        self.tickets = FrameTable(["side", "symbol", "quantity", "est_price", "est_value", "realized_gain", "term", "tax_benefit", "wash_status"],
                                  pct_cols=set(), filter_box=False)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.result)
        ll.addWidget(self.tickets, 1)
        split.addWidget(left)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.gauge = charts.PlotlyView()
        self.gauge.setMinimumHeight(240)
        self.proj = charts.PlotlyView()
        self.proj.setMinimumHeight(300)
        rl.addWidget(self.gauge)
        rl.addWidget(self.proj, 1)
        split.addWidget(right)
        split.setSizes([700, 700])
        split.setMinimumHeight(560)
        root.addWidget(split, 1)
        self.note = QLabel("Wealth chart is an illustration with stated assumptions (see hover), not a forecast. State rates are approximate planning figures. "
                           "Nothing here places orders.")
        self.note.setProperty("muted", True)
        self.note.setWordWrap(True)
        root.addWidget(self.note)

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        self.mode_btn.setText("Switch to expert mode" if getattr(self.app, "ui_mode", "expert") == "simple" else "Switch to simple mode")
        ts = self.svc.tax_setup()
        i = self.state.findData(ts["state"])
        self.state.setCurrentIndex(max(i, 0))
        self.filing.setCurrentText(ts["filing_status"])
        self.income.setValue(ts["other_income"])
        self.k_state.set(f"{pct(ts['st_rate'], 1)} / {pct(ts['lt_rate'], 1)}", "short-term / long-term, all-in")
        run_task(self.svc.kpis, None, on_done=self._kpis, on_error=lambda m: self.app.status(m.splitlines()[0][:160]), wants_progress=False)

    def _kpis(self, k: dict) -> None:
        if not k.get("has_holdings"):
            self.k_value.set("—", "no holdings yet")
            self.k_loss.set("—", "")
            self.k_tax.set("—", "")
            self.k_ytd.set("—", "")
            self.k_te.set("—", "")
            self.step1.setText("No holdings loaded. Import a broker file or use the demo.")
            self.gauge.set_message("Load holdings to see what could be saved.")
            return
        self.k_value.set(money(k.get("market_value")), f"{k.get('n_positions', 0)} positions · {k.get('n_lots', 0)} lots")
        self.k_loss.set(money(k.get("harvestable_loss")), f"{money(k.get('blocked_loss'))} wash-blocked" if k.get("blocked_loss") else "all wash-safe today", theme.RED if k.get("harvestable_loss") else None)
        self.k_tax.set(money(k.get("potential_tax_value")), "at your marginal rates", theme.GREEN if k.get("potential_tax_value") else None)
        self.k_ytd.set(money(k.get("ytd_tax_value", 0.0)), f"{money(k.get('ytd_harvested', 0.0))} of losses realised")
        te = k.get("tracking_error")
        self.k_te.set(pct(te, 2) if te is not None else "—", f"vs {k.get('benchmark')}" if te is not None else "fit a model to measure")
        self.step1.setText(f"Loaded: {k.get('n_positions', 0)} positions worth {money(k.get('market_value'))}.")
        st = k.get("st_rate", 0.4)
        self.gauge.set_figure(charts.savings_gauge(k.get("potential_tax_value", 0.0), max(k.get("harvestable_loss", 0.0) * st, 1.0), "Tax you could save with today's losses"))
        pr = self.svc.wealth_projection(float(k.get("market_value") or 0.0), k.get("st_rate", 0.4), k.get("lt_rate", 0.24))
        fig = charts.wealth_projection(pr["years"], pr["hold"], pr["tlh_after_deferred_tax"])
        a = pr["assumptions"]
        fig.update_layout(title=f"After-tax wealth, 20 years: harvesting adds ≈ {money(pr['gain_vs_hold'])} "
                                f"(assumes {a['market_return']:.0%} return, losses harvested {a['harvest_yield'][0]:.0%}→{a['harvest_yield'][-1]:.1%} of value/yr)")
        self.proj.set_figure(fig)

    # ------------------------------------------------------------------ actions
    def _toggle_mode(self) -> None:
        self.app.set_ui_mode("expert" if getattr(self.app, "ui_mode", "expert") == "simple" else "simple")
        self.refresh()

    def _import(self) -> None:
        from ..import_dialog import ImportDialog
        d = ImportDialog(self.app, self)
        if d.exec():
            self.app.reload_entities()
            self.data_changed.emit()

    def _demo(self) -> None:
        self.step1.setText("Seeding the demo household (pulls market data the first time)…")
        run_task(seed_demo, self.ctx, on_done=lambda eid: (self.app.reload_entities(), self.data_changed.emit(), self.app.status("Demo portfolio loaded.")),
                 on_error=self.app.error, on_progress=lambda m: self.step1.setText(m))

    def _save_tax(self) -> None:
        state = self.state.currentData()
        if not state:
            QMessageBox.information(self, "State", "Pick the client's state of residence first.")
            return
        out = self.svc.apply_tax_setup(state, self.filing.currentText(), self.income.value())
        c = out.get("combined")
        if c:
            from ...explain import explain_state
            self.step2.setText(explain_state(c))
        self.app.status(f"Tax settings saved: ST {out['st_rate']:.1%}, LT {out['lt_rate']:.1%} all-in.")
        self.data_changed.emit()

    def _one_click(self) -> None:
        self.go_btn.setEnabled(False)
        self.progress.setText("Working…")
        run_task(self.svc.one_click, None, on_done=self._done, on_error=self._fail, on_progress=self.progress.setText)

    def _done(self, res) -> None:
        self.go_btn.setEnabled(True)
        self.progress.setText(" · ".join(res.steps))
        self.result.set_text("\n\n".join(res.sentences))
        self.tickets.set_frame(res.trades)
        self.app.status(f"Harvest run #{res.run_id}: {money(res.summary.get('harvested_loss'))} of losses, ≈ {money(res.summary.get('tax_benefit'))} of tax.")
        self.data_changed.emit()

    def _fail(self, msg: str) -> None:
        self.go_btn.setEnabled(True)
        first = msg.splitlines()[0]
        self.progress.setText(first[:300])
        self.result.set_text(first)
        self.app.status(first[:160])
