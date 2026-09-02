"""Formatted Excel workbook for a harvest run (xlsxwriter).

Sheets: Summary, Trade Ticket, Recommended Trades, Wash-Sale Explanations, Blocked Lots, Replacements,
Factor Exposures, TE Decomposition, Sectors, Positions, Frontier (optional), Priority Comparison (optional).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import xlsxwriter

DARK = "#1F2A37"
ACCENT = "#2F6FED"
LIGHT = "#F3F5F9"
GREEN = "#1E8E5A"
RED = "#C0392B"


class _Fmt:
    def __init__(self, wb: xlsxwriter.Workbook):
        self.title = wb.add_format({"bold": True, "font_size": 16, "font_color": DARK})
        self.sub = wb.add_format({"italic": True, "font_color": "#6B7280"})
        self.hdr = wb.add_format({"bold": True, "font_color": "white", "bg_color": DARK, "border": 1, "text_wrap": True,
                                  "valign": "vcenter"})
        self.label = wb.add_format({"bold": True, "bg_color": LIGHT, "border": 1})
        self.cell = wb.add_format({"border": 1})
        self.money = wb.add_format({"num_format": "$#,##0.00;[Red]-$#,##0.00", "border": 1})
        self.money0 = wb.add_format({"num_format": "$#,##0;[Red]-$#,##0", "border": 1})
        self.pct = wb.add_format({"num_format": "0.00%;[Red]-0.00%", "border": 1})
        self.bps = wb.add_format({"num_format": "0.0", "border": 1})
        self.num = wb.add_format({"num_format": "#,##0.00", "border": 1})
        self.num4 = wb.add_format({"num_format": "0.0000", "border": 1})
        self.int = wb.add_format({"num_format": "#,##0", "border": 1})
        self.date = wb.add_format({"num_format": "yyyy-mm-dd", "border": 1})
        self.wrap = wb.add_format({"text_wrap": True, "valign": "top", "border": 1})
        self.good = wb.add_format({"font_color": GREEN})
        self.bad = wb.add_format({"font_color": RED})
        self.sell = wb.add_format({"bold": True, "font_color": RED, "border": 1})
        self.buy = wb.add_format({"bold": True, "font_color": GREEN, "border": 1})


COLUMN_FORMATS = {
    "money": {"est_value", "est_price", "realized_gain", "tax_benefit", "tax_alpha", "market_value", "cost_basis",
              "unrealized", "loss", "price", "cost_per_share", "basis_per_share", "proceeds", "amount", "sell_value",
              "buy_value", "harvested_loss", "basis_adjustment"},
    "pct": {"unrealized_pct", "weight", "te_after", "te_before", "te_budget", "turnover", "share", "correlation"},
    "int": {"lot_id", "account_id", "assetid", "days_to_lt", "n_lots", "n_trades", "run_id"},
    "num": {"quantity"},
    "num4": {"before", "after", "change", "portfolio", "benchmark", "active", "factor_vol", "variance", "te_contrib",
             "active_exposure"},
    "date": {"acquired", "holding_start", "sale_date", "event_date", "window_start", "window_end", "trade_date"},
    "wrap": {"wash_explanation", "explanation", "constraint", "rationale", "reason"},
}


def _fmt_for(col: str, f: _Fmt):
    for kind, cols in COLUMN_FORMATS.items():
        if col in cols:
            return getattr(f, kind)
    return f.cell


def write_frame(ws, df: pd.DataFrame, f: _Fmt, row: int = 0, col: int = 0, autofilter: bool = True,
                freeze: bool = True, widths: dict | None = None, side_col: str | None = "side") -> int:
    """Write a DataFrame with header formatting, number formats, autofilter and frozen header. Returns next row."""
    if df is None or df.empty:
        ws.write(row, col, "(none)", f.sub)
        return row + 2
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.date
    for j, c in enumerate(df.columns):
        ws.write(row, col + j, str(c).replace("_", " ").title(), f.hdr)
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        for j, (c, v) in enumerate(zip(df.columns, rec, strict=True)):
            fmt = _fmt_for(c, f)
            if side_col and c == side_col and isinstance(v, str):
                fmt = f.sell if v == "SELL" else f.buy
            if v is None or (isinstance(v, float) and pd.isna(v)):
                ws.write_blank(row + i, col + j, None, f.cell)
            elif hasattr(v, "isoformat") and not isinstance(v, str):
                ws.write_datetime(row + i, col + j, datetime(v.year, v.month, v.day), f.date)
            elif isinstance(v, int | float):
                ws.write_number(row + i, col + j, float(v), fmt)
            else:
                ws.write(row + i, col + j, str(v), fmt)
    n, m = df.shape
    if autofilter:
        ws.autofilter(row, col, row + n, col + m - 1)
    if freeze:
        ws.freeze_panes(row + 1, col)
    for j, c in enumerate(df.columns):
        w = (widths or {}).get(c)
        if w is None:
            sample = df[c].astype(str).str.len().quantile(0.9) if n else 10
            w = int(min(max(len(str(c)), sample) + 2, 60))
            if c in COLUMN_FORMATS["wrap"]:
                w = 70
        ws.set_column(col + j, col + j, w)
    return row + n + 2


def export_run_workbook(path: Path | str, run: dict, positions: pd.DataFrame | None = None,
                        frontier: pd.DataFrame | None = None, priority_table: pd.DataFrame | None = None,
                        title: str = "Tax-Loss Harvest Recommendation") -> Path:
    """`run` is the dict returned by HarvestService.load_run (summary, params, trades, blocked, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})
    f = _Fmt(wb)
    s = run.get("summary", {})
    p = run.get("params", {})

    # ---- Summary -------------------------------------------------------------------------------
    ws = wb.add_worksheet("Summary")
    ws.hide_gridlines(2)
    ws.set_column(0, 0, 34)
    ws.set_column(1, 1, 22)
    ws.set_column(3, 3, 30)
    ws.set_column(4, 4, 22)
    ws.write(0, 0, title, f.title)
    ws.write(1, 0, f"Run #{run.get('id', '')}  ·  as of {s.get('as_of', '')}  ·  generated {datetime.now():%Y-%m-%d %H:%M}", f.sub)
    ws.write(2, 0, "Decision support only. No orders are placed by this system. Not tax advice.", f.sub)
    left = [
        ("Portfolio value", s.get("portfolio_value"), f.money0),
        ("Harvested loss (total)", s.get("harvested_loss"), f.money0),
        ("  short-term", s.get("harvested_loss_st"), f.money0),
        ("  long-term", s.get("harvested_loss_lt"), f.money0),
        ("Tax benefit (current-year)", s.get("tax_benefit"), f.money0),
        ("Tax alpha (NPV of deferral)", s.get("tax_alpha"), f.money0),
        ("Tax alpha (bps of portfolio)", s.get("tax_alpha_bps"), f.bps),
        ("Total harvestable loss (pre-constraints)", s.get("total_harvestable_loss"), f.money0),
    ]
    right = [
        ("Tracking error before", s.get("te_before"), f.pct),
        ("Tracking error after", s.get("te_after"), f.pct),
        ("Turnover", s.get("turnover"), f.pct),
        ("Max style drift (z)", s.get("max_style_drift"), f.num4),
        ("Max sector drift", s.get("max_sector_drift"), f.pct),
        ("Sells / Buys", f"{s.get('n_sells', 0)} / {s.get('n_buys', 0)}", f.cell),
        ("Wash-blocked lots", s.get("n_blocked_lots"), f.int),
        ("Solver status", s.get("solver_status"), f.cell),
    ]
    for i, (k, v, fm) in enumerate(left, start=4):
        ws.write(i, 0, k, f.label)
        _w(ws, i, 1, v, fm, f)
    for i, (k, v, fm) in enumerate(right, start=4):
        ws.write(i, 3, k, f.label)
        _w(ws, i, 4, v, fm, f)
    r = 14
    ws.write(r, 0, "Configuration", f.title)
    cfg_rows = [("Mode", p.get("mode")), ("Constraint priority", " > ".join(p.get("priority", []))),
                ("Priority weights", ", ".join(str(x) for x in p.get("priority_weights", []))),
                ("TE budget", f"{p.get('te_budget', 0):.2%} ({'hard' if p.get('te_hard') else 'soft'})"),
                ("Sector drift max", f"{p.get('sector_drift_max', 0):.2%}"), ("Turnover max", f"{p.get('turnover_max', 0):.0%}"),
                ("Min trade value", p.get("min_trade_value")), ("Min loss to harvest", p.get("min_loss_value")),
                ("Cost (bps)", p.get("cost_bps")), ("Tax horizon (years)", p.get("tax_horizon_years")),
                ("Benchmark", p.get("benchmark")), ("Data snapshot", p.get("snapshot_id")),
                ("Risk model version", p.get("model_version_id"))]
    for i, (k, v) in enumerate(cfg_rows, start=r + 1):
        ws.write(i, 0, k, f.label)
        ws.write(i, 1, "" if v is None else str(v), f.cell)

    trades = run.get("trades", pd.DataFrame())
    # ---- Trade Ticket ----------------------------------------------------------------------------
    ws = wb.add_worksheet("Trade Ticket")
    ws.write(0, 0, "Trade Ticket (paper) — for manual entry / hand-off", f.title)
    ws.write(1, 0, "Quantities are whole shares; prices are last close estimates. Sells reference specific lots.", f.sub)
    if not trades.empty:
        t = trades.copy()
        cols = [c for c in ["side", "account_name", "symbol", "quantity", "est_price", "est_value", "lot_id", "term",
                            "replacement_for"] if c in t.columns]
        t = t[cols].sort_values(["side", "est_value"], ascending=[False, False])
        t["order_type"] = "MOC"
        t["time_in_force"] = "DAY"
        t["lot_instruction"] = t["lot_id"].apply(lambda x: f"Specific lot #{int(x)}" if pd.notna(x) else "")
        write_frame(ws, t, f, row=3)
    else:
        ws.write(3, 0, "No trades recommended.", f.sub)

    # ---- Recommended trades -----------------------------------------------------------------------
    ws = wb.add_worksheet("Recommended Trades")
    write_frame(ws, trades, f)

    # ---- Wash-sale explanations -------------------------------------------------------------------
    ws = wb.add_worksheet("Wash-Sale Explanations")
    ws.write(0, 0, "Every recommended trade was screened against the 61-day window across all linked accounts.", f.sub)
    if not trades.empty:
        we = trades[[c for c in ["side", "symbol", "account_name", "lot_id", "quantity", "wash_status", "wash_explanation"] if c in trades.columns]]
        write_frame(ws, we, f, row=2)

    # ---- Blocked lots ----------------------------------------------------------------------------
    ws = wb.add_worksheet("Blocked Lots")
    ws.write(0, 0, "Loss lots excluded because a sale today would be (partly) a wash sale.", f.sub)
    write_frame(ws, run.get("blocked", pd.DataFrame()), f, row=2, side_col=None)

    # ---- Replacements ------------------------------------------------------------------------------
    ws = wb.add_worksheet("Replacements")
    ws.write(0, 0, "Replacement candidates considered per harvested name (correlation over trailing year).", f.sub)
    write_frame(ws, run.get("replacements", pd.DataFrame()), f, row=2, side_col=None)

    # ---- Exposures / TE / sectors --------------------------------------------------------------------
    ex = run.get("exposures", pd.DataFrame())
    ws = wb.add_worksheet("Factor Exposures")
    if not ex.empty:
        e = ex.copy()
        e["change"] = e["after"] - e["before"]
        e = e.reset_index().rename(columns={"index": "factor"})
        write_frame(ws, e, f, side_col=None)
        n = len(e)
        ws.conditional_format(1, 3, n, 3, {"type": "3_color_scale", "min_color": "#F8D7DA", "mid_color": "#FFFFFF", "max_color": "#D4EDDA"})
    ws = wb.add_worksheet("TE Decomposition")
    tb, ta = run.get("te_before", pd.DataFrame()), run.get("te_after", pd.DataFrame())
    row = 0
    for label, d in (("Before", tb), ("After", ta)):
        ws.write(row, 0, f"{label}: TE = {d.attrs.get('tracking_error', float('nan')):.2%}" if hasattr(d, 'attrs') and d.attrs.get('tracking_error') else label, f.title)
        row = write_frame(ws, d.reset_index().rename(columns={"index": "factor"}) if not d.empty else d, f, row=row + 1, side_col=None, freeze=False)
    ws = wb.add_worksheet("Sectors")
    sec = run.get("sectors", pd.DataFrame())
    if not sec.empty:
        s2 = sec.copy()
        s2["change"] = s2["after"] - s2["before"]
        s2 = s2.reset_index().rename(columns={"index": "sector"})
        for c in ("before", "after", "change"):
            s2[c] = s2[c].astype(float)
        write_frame(ws, s2, f, side_col=None)

    # ---- Positions --------------------------------------------------------------------------------
    if positions is not None and not positions.empty:
        ws = wb.add_worksheet("Positions")
        cols = [c for c in positions.columns if c not in ("wash_explanation",)]
        write_frame(ws, positions[cols], f, side_col=None)
        n = len(positions)
        if "unrealized" in cols:
            j = cols.index("unrealized")
            ws.conditional_format(1, j, n, j, {"type": "cell", "criteria": "<", "value": 0, "format": f.bad})
            ws.conditional_format(1, j, n, j, {"type": "cell", "criteria": ">", "value": 0, "format": f.good})

    if frontier is not None and not frontier.empty:
        ws = wb.add_worksheet("Frontier")
        ws.write(0, 0, "Tax alpha vs tracking error across TE budgets (hard cap).", f.sub)
        write_frame(ws, frontier, f, row=2, side_col=None)
    if priority_table is not None and not priority_table.empty:
        ws = wb.add_worksheet("Priority Comparison")
        ws.write(0, 0, "How the recommendation changes under each constraint-hierarchy ordering.", f.sub)
        write_frame(ws, priority_table, f, row=2, side_col=None)

    wb.close()
    return path


def _w(ws, r, c, v, fm, f: _Fmt):
    if v is None:
        ws.write_blank(r, c, None, f.cell)
    elif isinstance(v, int | float):
        ws.write_number(r, c, float(v), fm)
    else:
        ws.write(r, c, str(v), fm)
