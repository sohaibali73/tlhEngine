"""Portfolio overview: lots, positions, wash calendar, realised ledger, transactions."""
from __future__ import annotations

from datetime import date

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import charts
from ..dialogs import InfoDialog, ScheduledEventDialog, TradeDialog, import_lots_csv
from ..widgets import FrameTable, KpiCard, TextPanel, button, hbox, header, money, pct, sign_color, vbox
from ..workers import run_task

LOT_COLS = ["lot_id", "account", "symbol", "acquired", "quantity", "cost_per_share", "price", "market_value", "cost_basis",
            "unrealized", "unrealized_pct", "term", "days_to_lt", "wash_status", "basis_adjustment", "source"]
POS_COLS = ["symbol", "accounts", "quantity", "price", "market_value", "weight", "cost_basis", "unrealized", "unrealized_pct",
            "unrealized_st", "unrealized_lt", "harvestable_loss", "n_lots"]
CLOSE_COLS = ["sale_date", "symbol", "account_name", "quantity", "proceeds", "cost_basis", "realized_gain", "term",
              "wash_disallowed", "wash_explanation"]
TX_COLS = ["trade_date", "account_name", "symbol", "tx_type", "quantity", "price", "fees", "source", "notes"]


class PortfolioScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.ps = app.portfolio_service
        self.lots = pd.DataFrame()
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(header("Portfolio", "Lot-level state across every account in the selected tax entity. Wash status is evaluated as of today."))
        top.addStretch(1)
        top.addWidget(button("Record buy", lambda: self._trade("BUY")))
        top.addWidget(button("Record sell", lambda: self._trade("SELL")))
        top.addWidget(button("Scheduled event", self._scheduled))
        top.addWidget(button("Import broker file…", self._import_broker, primary=True,
                             tooltip="Schwab / Fidelity / IBKR / TradeStation / Vanguard or any CSV/Excel with symbol, quantity, cost, date"))
        top.addWidget(button("Import lots CSV", self._import, tooltip="Simple template: account, symbol, date, quantity, price"))
        root.addLayout(top)

        kpis = QHBoxLayout()
        self.k_mv = KpiCard("Market value")
        self.k_unreal = KpiCard("Unrealised")
        self.k_harv = KpiCard("Harvestable loss")
        self.k_st = KpiCard("Short-term losses")
        self.k_blocked = KpiCard("Wash-blocked losses")
        self.k_te = KpiCard("Tracking error")
        for k in (self.k_mv, self.k_unreal, self.k_harv, self.k_st, self.k_blocked, self.k_te):
            kpis.addWidget(k)
        root.addLayout(kpis)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # --- Lots tab
        self.lots_table = FrameTable(LOT_COLS)
        self.lots_table.row_selected.connect(self._lot_selected)
        self.lots_table.row_activated.connect(self._lot_explain)
        self.explain = TextPanel("Wash-sale status")
        self.explain.set_text("Select a lot to see its wash-sale determination. Double-click for detail.")
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.lots_table)
        split.addWidget(self.explain)
        split.setSizes([600, 120])
        self.tabs.addTab(split, "Lots")

        # --- Positions tab
        self.pos_table = FrameTable(POS_COLS, pct_cols={"weight"})
        self.treemap = charts.PlotlyView()
        self.heat = charts.PlotlyView()
        chart_col = QSplitter(Qt.Vertical)
        chart_col.addWidget(self.treemap)
        chart_col.addWidget(self.heat)
        pos_split = QSplitter(Qt.Horizontal)
        pos_split.addWidget(self.pos_table)
        pos_split.addWidget(chart_col)
        pos_split.setSizes([700, 600])
        self.tabs.addTab(pos_split, "Positions")

        # --- Wash calendar tab
        self.cal_chart = charts.PlotlyView()
        self.cal_table = FrameTable(["symbol", "kind", "event_date", "window_start", "window_end", "account", "amount", "constraint"])
        cal_split = QSplitter(Qt.Vertical)
        cal_split.addWidget(self.cal_chart)
        cal_split.addWidget(self.cal_table)
        self.tabs.addTab(cal_split, "Wash calendar")

        # --- Realised tab
        self.year = QComboBox()
        y = date.today().year
        for yy in range(y, y - 6, -1):
            self.year.addItem(str(yy), yy)
        self.year.currentIndexChanged.connect(self._refresh_realized)
        self.k_r_st = KpiCard("Net short-term")
        self.k_r_lt = KpiCard("Net long-term")
        self.k_r_wash = KpiCard("Wash disallowed")
        self.k_r_ord = KpiCard("Ordinary offset used")
        self.k_r_cf = KpiCard("Carryforward → next yr")
        krow = hbox(self.k_r_st, self.k_r_lt, self.k_r_wash, self.k_r_ord, self.k_r_cf)
        self.closures_table = FrameTable(CLOSE_COLS)
        self.cum_chart = charts.PlotlyView()
        r_split = QSplitter(Qt.Vertical)
        r_split.addWidget(self.closures_table)
        r_split.addWidget(self.cum_chart)
        self.tabs.addTab(vbox(hbox(header("Realised gains & losses"), None, self.year), krow, r_split), "Realised")

        # --- Transactions tab
        self.tx_table = FrameTable(TX_COLS)
        self.tabs.addTab(self.tx_table, "Transactions")

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        eid = self.ctx.current_entity_id
        if eid is None:
            self.lots_table.set_frame(pd.DataFrame())
            return
        self.app.status("Loading portfolio…")
        run_task(self._load, eid, on_done=self._loaded, on_error=self.app.error, wants_progress=False)

    def _load(self, eid: int) -> dict:
        snap = self.app.data_service.latest_snapshot()
        lots = self.ps.lots_view(eid, snap=snap)
        out = {"lots": lots, "positions": self.ps.positions_view(lots), "summary": self.ps.summary(lots),
               "calendar": self.ps.wash_calendar(eid), "closures": self.ps.closures_view(eid),
               "tx": self.ctx.portfolio.transactions_frame(eid), "te": None}
        act = self.app.risk_service.active()
        if act and snap is not None and not lots.empty:
            _, model = act
            w = lots.groupby("symbol")["market_value"].sum()
            w = w[w.index.isin(model.symbols)]
            if w.sum() > 0:
                w = w / w.sum()
                try:
                    bench = self.app.risk_service.benchmark_weights(snap, model)
                    out["te"] = model.tracking_error(w.reindex(model.symbols).fillna(0.0), bench.reindex(model.symbols).fillna(0.0))
                except Exception:
                    out["te"] = None
        return out

    def _loaded(self, d: dict) -> None:
        self.lots = d["lots"]
        s = d["summary"]
        self.k_mv.set(money(s["market_value"]), f"{s['n_positions']} positions · {s['n_lots']} lots")
        self.k_unreal.set(money(s["unrealized"]), pct(s["unrealized"] / s["cost_basis"]) if s["cost_basis"] else "", sign_color(s["unrealized"]))
        self.k_harv.set(money(s["harvestable_loss"]), f"ST {money(s['harvestable_st'])} · LT {money(s['harvestable_lt'])}")
        self.k_st.set(money(s["harvestable_st"]), "worth more (ordinary rates)")
        self.k_blocked.set(money(s["blocked_loss"]), "cannot be harvested today", sign_color(-s["blocked_loss"]) if s["blocked_loss"] else None)
        self.k_te.set(pct(d["te"]) if d["te"] is not None else "—", "vs benchmark (active model)" if d["te"] is not None else "fit a risk model")
        self.lots_table.set_frame(self.lots)
        self.pos_table.set_frame(d["positions"])
        if not d["positions"].empty:
            self.treemap.set_figure(charts.treemap_positions(d["positions"]))
            self.heat.set_figure(charts.harvest_heatmap(d["positions"]))
        else:
            self.treemap.set_message("No positions. Record a buy, import lots, or seed the demo from Settings.")
            self.heat.set_message("")
        cal = d["calendar"]
        self.cal_table.set_frame(cal)
        self.cal_chart.set_figure(charts.wash_calendar_chart(cal, date.today()))
        self.closures = d["closures"]
        self.closures_table.set_frame(self.closures)
        self.cum_chart.set_figure(charts.cumulative_harvest_chart(self.closures))
        self.tx_table.set_frame(d["tx"])
        self._refresh_realized()
        self.app.status("Portfolio loaded.")

    def _refresh_realized(self) -> None:
        eid = self.ctx.current_entity_id
        if eid is None:
            return
        yr = self.year.currentData()
        try:
            r = self.ps.realized(eid, yr)
        except Exception as e:
            self.app.error(str(e))
            return
        n = r["netting"]
        self.k_r_st.set(money(r["net_st"]), f"gains {money(r['st_gains'])} · losses {money(r['st_losses'])}", sign_color(r["net_st"]))
        self.k_r_lt.set(money(r["net_lt"]), f"gains {money(r['lt_gains'])} · losses {money(r['lt_losses'])}", sign_color(r["net_lt"]))
        self.k_r_wash.set(money(r["wash_disallowed"]), f"{r['n_closures']} closures")
        self.k_r_ord.set(money(n["ordinary_deduction"]), f"prior CF: ST {money(r['prior_cf_st'])} / LT {money(r['prior_cf_lt'])}")
        self.k_r_cf.set(f"ST {money(n['carryforward_st'])}", f"LT {money(n['carryforward_lt'])}")

    # ------------------------------------------------------------------ interactions
    def _lot_selected(self, row: dict) -> None:
        txt = row.get("wash_explanation") or ("No unrealised loss on this lot; wash-sale rules only apply to loss sales."
                                            if (row.get("unrealized") or 0) >= 0 else "")
        grp = self.ctx.substitutes.explain_group(row["symbol"])
        self.explain.set_text(f"{row['symbol']} lot #{row['lot_id']} · {row['term']} ({row['days_to_lt']} days to long-term)\n{txt}\n\n{grp}")

    def _lot_explain(self, row: dict) -> None:
        InfoDialog(f"Lot #{row['lot_id']} {row['symbol']}", "\n".join(f"{k}: {v}" for k, v in row.items()), self).exec()

    def _trade(self, side: str) -> None:
        eid = self.ctx.current_entity_id
        if eid is None:
            QMessageBox.information(self, "No entity", "Create a tax entity and account in Settings first.")
            return
        accounts = self.ctx.entities.accounts(eid)
        sel = self.lots_table.selected_rows()
        dlg = TradeDialog(accounts, side, symbol=sel[0]["symbol"] if sel else "", parent=self)
        if dlg.exec():
            v = dlg.values()
            try:
                if side == "BUY":
                    self.ps.buy(v["account_id"], v["symbol"], v["trade_date"], v["quantity"], v["price"], v["fees"], v["source"], v["notes"])
                else:
                    cs = self.ps.sell(v["account_id"], v["symbol"], v["trade_date"], v["quantity"], v["price"], v["fees"], v["method"],
                                      v["specific_ids"], v["notes"])
                    washed = [c for c in cs if c.wash_disallowed]
                    if washed:
                        QMessageBox.warning(self, "Wash sale", "\n\n".join(c.wash_explanation for c in washed))
            except Exception as e:
                QMessageBox.critical(self, "Trade not recorded", str(e))
                return
            self.data_changed.emit()

    def _scheduled(self) -> None:
        eid = self.ctx.current_entity_id
        if eid is None:
            return
        dlg = ScheduledEventDialog(self.ctx.entities.accounts(eid), self)
        if dlg.exec():
            v = dlg.values()
            aid = self.ctx.resolve_assetid(v["symbol"])
            if aid is None:
                QMessageBox.warning(self, "Unknown symbol", v["symbol"])
                return
            self.ctx.portfolio.add_scheduled_event(v["account_id"], v["symbol"], aid, v["event_date"], v["event_type"], v["quantity"], notes=v["notes"])
            self.data_changed.emit()

    def _import(self) -> None:
        eid = self.ctx.current_entity_id
        if eid is None:
            return
        n = import_lots_csv(self, self.ps, self.ctx.entities.accounts(eid))
        if n:
            self.data_changed.emit()

    def _import_broker(self) -> None:
        from ..import_dialog import ImportDialog
        d = ImportDialog(self.app, self)
        if d.exec():
            self.app.reload_entities()
            self.data_changed.emit()
