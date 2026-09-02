"""Concentration & embedded gains workbench: overview, glide-path planner, Monte Carlo, hedging, alternatives, gain-offset plan."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...optim.glidepath import GlidePathSpec, MonteCarloSpec
from ...services.concentration_service import ConcentrationService
from .. import charts, theme
from ..widgets import FrameTable, KpiCard, TextPanel, button, hbox, header, money, pct, vbox
from ..workers import run_task

POS_COLS = ["symbol", "weight", "market_value", "unrealized", "embedded_gain_pct", "unrealized_st", "unrealized_lt", "tax_if_sold", "tax_drag_pct",
            "pct_of_risk", "te_reduction_if_sold", "lock_in_ratio", "specific_vol", "beta", "idio_share_of_own_risk", "n_lots"]
SCHED_COLS = ["period", "year", "position_value", "sold", "sold_fraction", "cumulative_fraction", "realised_gain", "loss_offset_used", "taxable_gain",
              "tax", "marginal_rate", "remaining_value", "weight_after"]


def bubble_chart(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["weight"]).copy()
    size = (d["market_value"] / d["market_value"].max() * 60 + 8) if len(d) else []
    fig = go.Figure(go.Scatter(x=d["embedded_gain_pct"] * 100, y=d["weight"] * 100, mode="markers+text", text=d["symbol"], textposition="top center",
                               marker=dict(size=size, color=(d.get("pct_of_risk", d["weight"]) * 100), colorscale=[[0, theme.ACCENT], [1, theme.RED]],
                                           colorbar=dict(title="% of risk"), line=dict(color=theme.BORDER, width=1)),
                               hovertemplate="%{text}<br>weight %{y:.1f}%<br>embedded gain %{x:.0f}%<extra></extra>"))
    fig.update_layout(title="Concentration map: weight vs embedded gain (size = value, colour = risk share)", xaxis_title="embedded gain (% of value)",
                      yaxis_title="weight (%)", height=420)
    return fig


def schedule_chart(s: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=s["period"], y=s["sold"], name="Sold ($)", marker_color=theme.ACCENT, yaxis="y")
    fig.add_bar(x=s["period"], y=s["tax"], name="Tax ($)", marker_color=theme.RED, yaxis="y")
    fig.add_scatter(x=s["period"], y=s["weight_after"] * 100, name="Weight after (%)", mode="lines+markers", line=dict(color=theme.AMBER), yaxis="y2")
    fig.update_layout(title="Optimised glide path", barmode="group", height=360, yaxis=dict(title="$"),
                      yaxis2=dict(title="weight %", overlaying="y", side="right", gridcolor=theme.BORDER))
    return fig


def comparison_chart(c: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in ("pv_tax_paid", "pv_terminal_tax", "pv_risk_cost", "pv_alpha_forgone", "pv_costs"):
        fig.add_bar(name=col.replace("pv_", "PV ").replace("_", " "), x=c["policy"], y=c[col])
    fig.update_layout(title="Policy comparison (present-value cost components; lower is better)", barmode="stack", height=340, yaxis_title="$")
    return fig


def fan_chart(out: dict, policy: str) -> go.Figure:
    p = out["policies"][policy]
    fan = p["fan"]
    x = list(range(fan.shape[0]))
    fig = go.Figure()
    fig.add_scatter(x=x, y=fan[:, 4], line=dict(width=0), showlegend=False)
    fig.add_scatter(x=x, y=fan[:, 0], fill="tonexty", fillcolor="rgba(59,130,246,0.15)", line=dict(width=0), name="5–95%")
    fig.add_scatter(x=x, y=fan[:, 3], line=dict(width=0), showlegend=False)
    fig.add_scatter(x=x, y=fan[:, 1], fill="tonexty", fillcolor="rgba(59,130,246,0.30)", line=dict(width=0), name="25–75%")
    fig.add_scatter(x=x, y=fan[:, 2], line=dict(color=theme.ACCENT2, width=2), name="median")
    fig.update_layout(title=f"Pre-tax wealth paths · {policy}", xaxis_title="year", yaxis_title="$", height=340)
    return fig


def terminal_chart(out: dict) -> go.Figure:
    fig = go.Figure()
    names = list(out["policies"])
    fig.add_bar(name="P5", x=names, y=[out["policies"][n]["p5"] for n in names], marker_color=theme.RED)
    fig.add_bar(name="Median", x=names, y=[out["policies"][n]["median"] for n in names], marker_color=theme.ACCENT)
    fig.add_bar(name="P95", x=names, y=[out["policies"][n]["p95"] for n in names], marker_color=theme.GREEN)
    fig.update_layout(title="After-tax terminal wealth by policy", barmode="group", height=340, yaxis_title="$")
    return fig


def payoff_chart(rows: list[dict]) -> go.Figure:
    d = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(x=d["pct_move"] * 100, y=d["unhedged_pnl"], name="Unhedged", line=dict(color=theme.MUTED))
    fig.add_scatter(x=d["pct_move"] * 100, y=d["collar_pnl"], name="Collar", line=dict(color=theme.ACCENT))
    fig.update_layout(title="P&L at expiry vs spot move", xaxis_title="% move", yaxis_title="$", height=320)
    return fig


class ConcentrationScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.svc = ConcentrationService(self.ctx)
        self._plan = None
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        top.addWidget(header("Concentration & embedded gains", "How locked-in is the book, what would it cost to diversify, and what is the best multi-year path: "
                                                              "bracket-aware glide paths with loss offsets, Monte Carlo, collars, charitable and exchange-fund alternatives."))
        top.addStretch(1)
        self.symbol = QComboBox()
        self.symbol.setMinimumWidth(120)
        top.addWidget(QLabel("Position"))
        top.addWidget(self.symbol)
        self.income = QDoubleSpinBox()
        self.income.setRange(0, 1e8)
        self.income.setPrefix("$ ")
        self.income.setSingleStep(10_000)
        self.income.setToolTip("Other taxable income the gains stack on (drives the bracket)")
        top.addWidget(QLabel("Other taxable income"))
        top.addWidget(self.income)
        top.addWidget(button("Save", lambda: (self.svc.set_other_income(self.income.value()), self.refresh())))
        root.addLayout(top)
        k = QHBoxLayout()
        self.k_top = KpiCard("Largest position")
        self.k_effn = KpiCard("Effective N")
        self.k_gain = KpiCard("Embedded gain")
        self.k_tax = KpiCard("Tax if liquidated")
        self.k_lock = KpiCard("Locked-in %")
        self.k_rate = KpiCard("Marginal LTCG rate")
        for c in (self.k_top, self.k_effn, self.k_gain, self.k_tax, self.k_lock, self.k_rate):
            k.addWidget(c)
        root.addLayout(k)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # overview
        self.pos_table = FrameTable(POS_COLS, pct_cols={"weight", "embedded_gain_pct", "tax_drag_pct", "pct_of_risk", "specific_vol", "idio_share_of_own_risk", "te_reduction_if_sold"}, selection="multi")
        self.pos_table.row_selected.connect(lambda r: self.symbol.setCurrentText(str(r["symbol"])))
        self.bubble = charts.PlotlyView()
        osplit = QSplitter(Qt.Vertical)
        osplit.addWidget(self.bubble)
        osplit.addWidget(self.pos_table)
        self.comp_name = QLineEdit("Completion")
        self.comp_name.setMaximumWidth(220)
        comp_row = hbox(QLabel("Lock-in ratio = tax to liquidate per 1% of TE removed (lower = cheaper to exit)."), None, QLabel("Save as basket"), self.comp_name,
                        button("Completion portfolio around selected", self.run_completion, tooltip="Keep the selected positions; optimise the rest of the book to minimise TE"))
        self.tabs.addTab(vbox(osplit, comp_row), "Overview")

        # planner
        pl = QWidget()
        pll = QHBoxLayout(pl)
        form_box = QScrollArea()
        form_box.setWidgetResizable(True)
        fw = QWidget()
        f = QFormLayout(fw)
        self.horizon = QSpinBox()
        self.horizon.setRange(1, 30)
        self.horizon.setValue(5)
        self.ppy = QComboBox()
        self.ppy.addItems(["1", "2", "4"])
        self.ra = QDoubleSpinBox()
        self.ra.setRange(0, 500)
        self.ra.setValue(4)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(-50, 50)
        self.alpha.setSuffix(" %")
        self.alpha.setToolTip("Your view of the stock's expected excess return vs a diversified portfolio")
        self.er = QDoubleSpinBox()
        self.er.setRange(-50, 50)
        self.er.setValue(7)
        self.er.setSuffix(" %")
        self.disc = QDoubleSpinBox()
        self.disc.setRange(0, 20)
        self.disc.setValue(4)
        self.disc.setSuffix(" %")
        self.pstep = QDoubleSpinBox()
        self.pstep.setRange(0, 100)
        self.pstep.setSuffix(" %")
        self.pstep.setToolTip("Probability of a basis step-up (death) within the horizon")
        self.budget = QDoubleSpinBox()
        self.budget.setRange(0, 1e9)
        self.budget.setPrefix("$ ")
        self.budget.setSingleStep(50_000)
        self.budget.setSpecialValueText("none")
        self.min_by = QLineEdit("")
        self.min_by.setPlaceholderText("e.g. 3=0.5, 5=0.9 (cum. fraction sold by year)")
        self.use_losses = QComboBox()
        self.use_losses.addItems(["use expected harvest losses", "ignore losses"])
        self.cost = QDoubleSpinBox()
        self.cost.setRange(0, 200)
        self.cost.setValue(10)
        self.cost.setSuffix(" bps")
        for lbl, w in (("Horizon (years)", self.horizon), ("Periods per year", self.ppy), ("Risk aversion (λ)", self.ra), ("Alpha view", self.alpha),
                       ("Expected stock return", self.er), ("Discount rate", self.disc), ("P(step-up in horizon)", self.pstep),
                       ("Annual gain budget", self.budget), ("Min sold by year", self.min_by), ("Loss offsets", self.use_losses), ("Costs", self.cost)):
            f.addRow(lbl, w)
        self.plan_btn = button("Optimise glide path", self.run_plan, primary=True)
        f.addRow(self.plan_btn)
        self.mc_btn = button("Monte Carlo (5,000 paths)", self.run_mc)
        f.addRow(self.mc_btn)
        self.offset_value = QDoubleSpinBox()
        self.offset_value.setRange(0, 1e9)
        self.offset_value.setPrefix("$ ")
        self.offset_value.setSingleStep(25_000)
        self.offset_value.setValue(100_000)
        self.repl = QLineEdit("")
        self.repl.setPlaceholderText("replacement symbol (optional)")
        f.addRow("Year-1 sale amount", self.offset_value)
        f.addRow("Replacement", self.repl)
        f.addRow(button("Create gain-offset trade plan", self.run_offset, success=True,
                        tooltip="Sell from highest-basis lots and pair with wash-safe harvest losses; saved as a reviewable plan in Harvest"))
        form_box.setWidget(fw)
        form_box.setMaximumWidth(420)
        pll.addWidget(form_box)
        right = QSplitter(Qt.Vertical)
        self.sched_chart = charts.PlotlyView()
        self.comp_chart = charts.PlotlyView()
        crow = QSplitter(Qt.Horizontal)
        crow.addWidget(self.sched_chart)
        crow.addWidget(self.comp_chart)
        right.addWidget(crow)
        self.sched_table = FrameTable(SCHED_COLS, pct_cols={"sold_fraction", "cumulative_fraction", "marginal_rate", "weight_after"})
        self.comp_table = FrameTable(["policy", "feasible", "total_objective", "pv_tax_paid", "pv_terminal_tax", "pv_risk_cost", "pv_alpha_forgone", "pv_costs", "final_weight", "total_sold"],
                                     pct_cols={"final_weight"})
        tbl = QSplitter(Qt.Horizontal)
        tbl.addWidget(self.sched_table)
        tbl.addWidget(self.comp_table)
        right.addWidget(tbl)
        self.plan_note = TextPanel("Plan summary")
        right.addWidget(self.plan_note)
        right.setSizes([340, 300, 120])
        pll.addWidget(right, 1)
        self.tabs.addTab(pl, "Glide-path planner")

        # monte carlo
        self.fan_policy = QComboBox()
        self.fan_policy.addItems(["hold", "sell all now", "equal instalments", "optimised"])
        self.fan_policy.currentTextChanged.connect(self._refan)
        self.fan = charts.PlotlyView()
        self.term = charts.PlotlyView()
        self.mc_table = FrameTable(["policy", "mean", "median", "p5", "p95", "std", "cvar5", "p_beats_sell_now"], pct_cols={"p_beats_sell_now"})
        mrow = QSplitter(Qt.Horizontal)
        mrow.addWidget(self.fan)
        mrow.addWidget(self.term)
        mcol = QSplitter(Qt.Vertical)
        mcol.addWidget(mrow)
        mcol.addWidget(self.mc_table)
        self.tabs.addTab(vbox(hbox(QLabel("Fan chart policy"), self.fan_policy, None), mcol), "Monte Carlo")

        # hedging
        hd = QWidget()
        hl = QHBoxLayout(hd)
        hf_box = QGroupBox("Collar")
        hf = QFormLayout(hf_box)
        self.h_T = QDoubleSpinBox()
        self.h_T.setRange(0.05, 5)
        self.h_T.setValue(1.0)
        self.h_T.setSuffix(" yrs")
        self.h_put = QDoubleSpinBox()
        self.h_put.setRange(50, 100)
        self.h_put.setValue(90)
        self.h_put.setSuffix(" % of spot")
        self.h_call = QDoubleSpinBox()
        self.h_call.setRange(0, 200)
        self.h_call.setValue(0)
        self.h_call.setSuffix(" % of spot")
        self.h_call.setSpecialValueText("zero-cost")
        self.h_vol = QDoubleSpinBox()
        self.h_vol.setRange(0, 300)
        self.h_vol.setValue(0)
        self.h_vol.setSuffix(" %")
        self.h_vol.setSpecialValueText("model vol")
        self.h_r = QDoubleSpinBox()
        self.h_r.setRange(0, 20)
        self.h_r.setValue(4)
        self.h_r.setSuffix(" %")
        self.h_q = QDoubleSpinBox()
        self.h_q.setRange(0, 20)
        self.h_q.setSuffix(" %")
        for lbl, w in (("Tenor", self.h_T), ("Put strike", self.h_put), ("Call strike", self.h_call), ("Implied vol", self.h_vol), ("Rate", self.h_r), ("Dividend yield", self.h_q)):
            hf.addRow(lbl, w)
        hf.addRow(button("Analyse collar", self.run_hedge, primary=True))
        self.hedge_out = TextPanel("Result & tax flags")
        hf.addRow(self.hedge_out)
        hf_box.setMaximumWidth(420)
        hl.addWidget(hf_box)
        self.payoff = charts.PlotlyView()
        hl.addWidget(self.payoff, 1)
        self.tabs.addTab(hd, "Hedging")

        # alternatives
        al = QWidget()
        all_ = QVBoxLayout(al)
        arow = QHBoxLayout()
        self.agi = QDoubleSpinBox()
        self.agi.setRange(0, 1e9)
        self.agi.setPrefix("$ ")
        self.agi.setValue(400_000)
        self.agi.setSingleStep(25_000)
        self.alt_pstep = QDoubleSpinBox()
        self.alt_pstep.setRange(0, 100)
        self.alt_pstep.setValue(30)
        self.alt_pstep.setSuffix(" %")
        arow.addWidget(QLabel("AGI"))
        arow.addWidget(self.agi)
        arow.addWidget(QLabel("P(step-up, 10y)"))
        arow.addWidget(self.alt_pstep)
        arow.addWidget(button("Compare alternatives", self.run_alternatives, primary=True))
        arow.addStretch(1)
        all_.addLayout(arow)
        self.alt_out = QPlainTextEdit()
        self.alt_out.setReadOnly(True)
        all_.addWidget(self.alt_out, 1)
        self.tabs.addTab(al, "Charitable, gifting, exchange fund, step-up")

        # brackets
        br = QWidget()
        brl = QVBoxLayout(br)
        brl.addWidget(QLabel("Bracket schedule (approximate 2026 defaults; edit upper bounds/rates as JSON and save). Marginal rates stack gains on other taxable income."))
        self.br_edit = QPlainTextEdit()
        brl.addWidget(self.br_edit, 1)
        brl.addWidget(hbox(button("Save brackets", self._save_brackets), button("Reset to defaults", self._reset_brackets), None))
        self.tabs.addTab(br, "Tax brackets")

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        self.income.setValue(self.svc.other_income())
        import json
        self.br_edit.setPlainText(json.dumps(self.svc.brackets().to_dict(), indent=2))
        run_task(self.svc.overview, self.ctx.current_entity_id, on_done=self._overview_done, on_error=lambda m: self.app.status(m.splitlines()[0][:160]), wants_progress=False)

    def _overview_done(self, d: dict) -> None:
        pos, st = d["positions"], d["stats"]
        if pos.empty:
            return
        cur = self.symbol.currentText()
        self.symbol.blockSignals(True)
        self.symbol.clear()
        self.symbol.addItems(list(pos["symbol"]))
        if cur in set(pos["symbol"]):
            self.symbol.setCurrentText(cur)
        self.symbol.blockSignals(False)
        self.pos_table.set_frame(pos)
        self.bubble.set_figure(bubble_chart(pos))
        top = pos.iloc[0]
        self.k_top.set(f"{top['symbol']} {top['weight']:.1%}", f"top-5 {st['top5']:.0%} · HHI {st['hhi']:.3f}")
        self.k_effn.set(f"{st['effective_n']:.1f}", f"{st['n_positions']} positions")
        self.k_gain.set(money(st["total_embedded_gain"]), f"gain-weighted concentration {st['gain_weighted_concentration']:.2f}")
        self.k_tax.set(money(st["total_tax_if_liquidated"]), "taxable accounts, stacked on other income")
        self.k_lock.set(pct(st["locked_in_pct"]), "tax as % of portfolio value")
        b = self.svc.brackets()
        from ...tax.concentration import marginal_ltcg_rate
        self.k_rate.set(pct(marginal_ltcg_rate(1.0, st["other_taxable_income"], b)), f"first $ of LT gain · {b.filing_status}")

    # ------------------------------------------------------------------ planner
    def _spec(self) -> GlidePathSpec:
        min_by = {}
        for part in self.min_by.text().split(","):
            if "=" in part:
                y, fr = part.split("=", 1)
                try:
                    min_by[int(y)] = float(fr)
                except ValueError:
                    pass
        return GlidePathSpec(horizon_years=self.horizon.value(), periods_per_year=int(self.ppy.currentText()), other_taxable_income=self.income.value(),
                             discount_rate=self.disc.value() / 100, risk_aversion=self.ra.value(), alpha_view=self.alpha.value() / 100,
                             expected_return=self.er.value() / 100, cost_bps=self.cost.value(), p_stepup=self.pstep.value() / 100,
                             annual_gain_budget=self.budget.value() or None, min_sold_by=min_by)

    def run_plan(self) -> None:
        sym = self.symbol.currentText()
        if not sym:
            return
        self.plan_btn.setEnabled(False)
        self.app.status(f"Optimising glide path for {sym}…")
        run_task(self.svc.plan, sym, self._spec(), self.ctx.current_entity_id, self.use_losses.currentIndex() == 0,
                 on_done=self._plan_done, on_error=self._fail, wants_progress=False)

    def _plan_done(self, d: dict) -> None:
        self.plan_btn.setEnabled(True)
        self._plan = d
        s, c, summ = d["schedule"], d["comparison"], d["summary"]
        self.sched_table.set_frame(s)
        self.comp_table.set_frame(c)
        self.sched_chart.set_figure(schedule_chart(s))
        self.comp_chart.set_figure(comparison_chart(c))
        p = d["position"]
        best = c.iloc[0]["policy"]
        self.plan_note.set_text(
            f"{d['symbol']}: value {money(p['value'])}, basis {money(p['basis'])}, embedded gain {money(p['value'] - p['basis'])} · specific vol {p['specific_vol']:.0%}, beta {p['beta']:.2f}.\n"
            f"Optimised: sell {money(summ['total_sold'])} over {summ['years']} years, realise {money(summ['total_gain_realised'])} of gains, use {money(summ['losses_used'])} of losses, "
            f"pay {money(summ['total_tax'])} tax (effective {summ['effective_tax_rate']:.1%}; PV {money(summ['pv_tax'])}); final weight {summ['final_weight']:.1%}. "
            f"Best policy by total PV cost: {best}. Solver: {d['status']}.")
        self.tabs.setCurrentIndex(1)
        self.app.status("Glide path optimised.")

    def run_mc(self) -> None:
        sym = self.symbol.currentText()
        if not sym:
            return
        spec = self._spec()
        opt = np.asarray(self._plan["schedule"]["sold"].values) if self._plan and self._plan["symbol"] == sym else None
        mc = MonteCarloSpec(n_paths=5000, horizon_years=spec.horizon_years, rf=spec.discount_rate)
        self.mc_btn.setEnabled(False)
        self.app.status("Running Monte Carlo…")
        run_task(self.svc.monte_carlo, sym, spec, mc, self.ctx.current_entity_id, opt, on_done=self._mc_done, on_error=self._fail, wants_progress=False)

    def _mc_done(self, out: dict) -> None:
        self.mc_btn.setEnabled(True)
        self._mc = out
        rows = [{"policy": n, **{k: v for k, v in o.items() if k != "fan"}} for n, o in out["policies"].items()]
        self.mc_table.set_frame(pd.DataFrame(rows))
        self.term.set_figure(terminal_chart(out))
        self._refan()
        self.tabs.setCurrentIndex(2)
        self.app.status(f"Monte Carlo done: stock μ {out['mu_stock']:.1%} σ {out['sigma_stock']:.0%} vs market μ {out['mu_market']:.1%}.")

    def _refan(self) -> None:
        if getattr(self, "_mc", None):
            self.fan.set_figure(fan_chart(self._mc, self.fan_policy.currentText()))

    def run_offset(self) -> None:
        sym = self.symbol.currentText()
        if not sym or self.offset_value.value() <= 0:
            return
        self.app.status("Building gain-offset plan…")
        run_task(self.svc.gain_offset_plan, sym, self.offset_value.value(), None, self.ctx.current_entity_id, True, self.repl.text().strip() or None,
                 on_done=self._offset_done, on_error=self._fail, wants_progress=False)

    def _offset_done(self, out) -> None:
        rid, d = out
        s = d["summary"]
        QMessageBox.information(self, "Gain-offset plan saved", f"Plan saved as run #{rid} (Harvest › Run history).\n\nConcentrated sale gain: ${d['gain_from_concentrated_sale']:,.0f}\n"
                                                                f"Offset loss lots: {d['n_offset_lots']}\nNet realised: ${s['realized_gains'] - s['harvested_loss']:,.0f}\n"
                                                                f"Wash-unsafe trades: {s['n_unsafe_trades']}\nTE {s['te_before']:.2%} → {s['te_after']:.2%}")
        self.data_changed.emit()

    def run_completion(self) -> None:
        rows = self.pos_table.selected_rows()
        syms = [str(r["symbol"]) for r in rows] or ([self.symbol.currentText()] if self.symbol.currentText() else [])
        if not syms:
            return
        self.app.status(f"Building completion portfolio around {', '.join(syms)}…")
        run_task(self.svc.completion, syms, 60, 0.05, 0.03, self.comp_name.text().strip() or None, self.ctx.current_entity_id,
                 on_done=self._completion_done, on_error=self._fail, wants_progress=False)

    def _completion_done(self, d: dict) -> None:
        QMessageBox.information(self, "Completion portfolio", f"Locked: {', '.join(d['locked'])} ({1 - d['free_budget']:.0%} of the book)\n"
                                                              f"TE of current book: {d['te_current_book']:.2%}\nTE of locked names alone: {d['te_locked_alone']:.2%}\n"
                                                              f"TE with optimised completion sleeve ({d['n_free_names']} names): {d['te_completion']:.2%}"
                                                              + (f"\nSaved as basket '{d['saved_basket']}' (Model portfolios / Harvest benchmark)." if d.get("saved_basket") else ""))
        self.data_changed.emit()

    # ------------------------------------------------------------------ hedging & alternatives
    def run_hedge(self) -> None:
        sym = self.symbol.currentText()
        if not sym:
            return
        call = (self.h_call.value() / 100) if self.h_call.value() > 0 else None
        vol = (self.h_vol.value() / 100) if self.h_vol.value() > 0 else None
        run_task(self.svc.hedge, sym, self.h_T.value(), self.h_put.value() / 100, call, vol, self.h_r.value() / 100, self.h_q.value() / 100, self.ctx.current_entity_id,
                 on_done=self._hedge_done, on_error=self._fail, wants_progress=False)

    def _hedge_done(self, a: dict) -> None:
        txt = (f"{a['symbol']}: {a['shares']:,.0f} sh @ {a['spot']:.2f} = {money(a['position_value'])}, embedded gain {money(a['embedded_gain'])}, tax if sold now {money(a['tax_if_sold_now'])}.\n"
               f"Collar {a['T']:.2g}y, σ {a['sigma']:.0%}: put {a['put_strike']:.2f} ({a['floor_pct']:+.0%}) / call {a['call_strike']:.2f} ({a['cap_pct']:+.0%}); "
               f"net cost {money(a['net_cost_total'])} ({a['net_cost_pct']:.2%}, {a['annualised_cost_pct']:.2%}/yr). Floor {money(a['floor_value'])}, cap {money(a['cap_value'])}, "
               f"delta {a['delta_hedged']:.2f}.\nHedging defers {money(a['tax_deferred_by_hedging'])} of tax; selling now nets {money(a['sell_now_after_tax'])}.\n\n" + "\n".join("⚠ " + f for f in a["flags"]))
        self.hedge_out.set_text(txt)
        self.payoff.set_figure(payoff_chart(a["payoff"]))

    def run_alternatives(self) -> None:
        sym = self.symbol.currentText()
        if not sym:
            return
        run_task(self.svc.alternatives, sym, self.agi.value(), self.alt_pstep.value() / 100, 10.0, self.ctx.current_entity_id,
                 on_done=self._alt_done, on_error=self._fail, wants_progress=False)

    def _alt_done(self, d: dict) -> None:
        import json
        c, ef, su, g = d["charitable"], d["exchange_fund"], d["stepup"], d["gift"]
        lines = [f"{d['symbol']}: value {money(d['value'])}, basis {money(d['basis'])}, embedded gain {money(d['embedded_gain'])}, marginal LTCG {d['ltcg_marginal_rate']:.1%}",
                 f"SELL NOW: tax {money(d['sell_now']['tax'])}, after-tax {money(d['sell_now']['after_tax'])}", "",
                 "DONATE SHARES vs SELL-THEN-DONATE:",
                 f"  donate shares: charity gets {money(c['donate_shares']['charity_receives'])}, deduction saves {money(c['donate_shares']['tax_saved_by_deduction'])}, net cost to donor {money(c['donate_shares']['net_cost_to_donor'])}",
                 f"  sell then donate: charity gets {money(c['sell_then_donate']['charity_receives'])}, cap-gains tax {money(c['sell_then_donate']['cap_gains_tax_paid'])}, net cost {money(c['sell_then_donate']['net_cost_to_donor'])}",
                 f"  advantage of donating shares: {money(c['advantage_of_donating_shares'])}; extra to charity {money(c['extra_to_charity'])}",
                 *["  ⚠ " + f for f in c["flags"]], "",
                 f"GIFT TO LOWER-BRACKET FAMILY: tax saved up to {money(g['tax_saved'])} ({g['donor_rate']:.1%} → {g['donee_rate']:.1%})", *["  · " + n for n in g["notes"]], "",
                 f"EXCHANGE FUND (§721): defers {money(ef['tax_deferred'])}; PV of fees {money(ef['pv_of_fees'])}; PV value of deferral {money(ef['pv_value_of_deferral'])}; net {money(ef['net_benefit_vs_selling'])} over {ef['lockup_years']} years",
                 *["  · " + n for n in ef["notes"]], "",
                 f"STEP-UP: liability {money(su['tax_liability'])}; PV if deferred 10y {money(su['pv_if_deferred_to_horizon'])}; expected PV with step-up probability {money(su['expected_pv_with_stepup'])}; option value {money(su['value_of_stepup_option'])}",
                 "", *d["notes"]]
        self.alt_out.setPlainText("\n".join(lines))
        _ = json

    def _fail(self, msg: str) -> None:
        self.plan_btn.setEnabled(True)
        self.mc_btn.setEnabled(True)
        self.app.error(msg)

    # ------------------------------------------------------------------ brackets
    def _save_brackets(self) -> None:
        import json

        from ...tax.concentration import BracketSchedule
        try:
            b = BracketSchedule.from_dict(json.loads(self.br_edit.toPlainText()))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Brackets", f"Invalid JSON: {e}")
            return
        self.svc.save_brackets(b)
        self.refresh()
        self.app.status("Bracket schedule saved.")

    def _reset_brackets(self) -> None:
        self.ctx.db.execute("DELETE FROM settings WHERE key = 'bracket_schedule'")
        self.refresh()
