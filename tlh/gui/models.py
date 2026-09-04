"""Qt table model over a pandas DataFrame with per-column number formatting and gain/loss colouring."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor

from . import theme

MONEY = {"est_value", "est_price", "realized_gain", "tax_benefit", "tax_alpha", "market_value", "cost_basis", "unrealized",
         "loss", "price", "cost_per_share", "basis_per_share", "proceeds", "amount", "sell_value", "buy_value",
         "harvested_loss", "basis_adjustment", "st_amount", "lt_amount", "wash_disallowed", "value"}
PCT = {"unrealized_pct", "weight", "te_after", "te_before", "te_budget", "turnover", "share", "correlation", "fed_st_rate",
       "fed_lt_rate", "state_rate", "niit_rate", "before", "after", "change"}
PCT_ONLY_SECTOR = {"before", "after", "change"}
INT = {"lot_id", "account_id", "assetid", "days_to_lt", "n_lots", "n_trades", "id", "run_id", "version_no", "n", "n_chars",
       "model_version_id", "conversation_id", "quantity_original"}
QTY = {"quantity", "quantity_open"}
Z = {"portfolio", "benchmark", "active", "factor_vol", "variance", "te_contrib", "active_exposure", "max_style_drift"}
SIGNED = {"realized_gain", "unrealized", "unrealized_pct", "active", "change", "tax_alpha", "tax_benefit", "harvested_loss"}


def fmt_value(col: str, v, pct_mode: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, bool | np.bool_):
        return "yes" if v else "no"
    if isinstance(v, datetime | pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, int | np.integer) and col not in MONEY and col not in PCT:
        return f"{int(v):,}"
    if isinstance(v, float | np.floating):
        f = float(v)
        if col in MONEY:
            return f"{f:,.2f}" if abs(f) < 1e5 else f"{f:,.0f}"
        if col in PCT or pct_mode:
            return f"{f * 100:.2f}%"
        if col in INT:
            return f"{f:,.0f}"
        if col in QTY:
            return f"{f:,.4g}" if f != int(f) else f"{int(f):,}"
        if col in Z:
            return f"{f:.3f}"
        return f"{f:,.4g}"
    return str(v)


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None, pct_cols: set[str] | None = None, parent=None):
        super().__init__(parent)
        self._pct_cols = pct_cols or set()
        self._df = pd.DataFrame()
        self._display: list[list[str]] = []
        self._vals: list[list] = []
        self._numeric: list[bool] = []
        self._cols: list[str] = []
        self.set_frame(df if df is not None else pd.DataFrame(), _reset=False)

    # ------------------------------------------------------------------ data
    def set_frame(self, df: pd.DataFrame, _reset: bool = True) -> None:
        """Cache display strings, raw values and alignment per column once, so the view's thousands of data() calls
        (painting, sorting, column sizing) are list lookups instead of pandas scalar access plus formatting."""
        if _reset:
            self.beginResetModel()
        self._df = df.reset_index(drop=True) if df is not None else pd.DataFrame()
        self._cols = [str(c) for c in self._df.columns]
        cols_vals = []
        cols_disp = []
        numeric = []
        for c in self._cols:
            s = self._df[c]
            vals = s.tolist()
            pct = c in self._pct_cols
            cols_disp.append([fmt_value(c, v, pct_mode=pct) for v in vals])
            cols_vals.append(vals)
            numeric.append(bool(pd.api.types.is_numeric_dtype(s)) and not bool(pd.api.types.is_bool_dtype(s)))
        # row-major for O(1) access in data()
        n = len(self._df)
        self._display = [[cols_disp[j][i] for j in range(len(self._cols))] for i in range(n)]
        self._vals = [[cols_vals[j][i] for j in range(len(self._cols))] for i in range(n)]
        self._numeric = numeric
        if _reset:
            self.endResetModel()

    def display_at(self, row: int, col: int) -> str:
        return self._display[row][col]

    @property
    def frame(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self._df)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self._df.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section]).replace("_", " ")
        return str(section + 1)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if role == Qt.DisplayRole:
            return self._display[r][c]
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter) if self._numeric[c] else int(Qt.AlignLeft | Qt.AlignVCenter)
        col = self._cols[c]
        v = self._vals[r][c]
        if role == Qt.ForegroundRole:
            if col in SIGNED and isinstance(v, int | float | np.integer | np.floating) and not (isinstance(v, float) and np.isnan(v)):
                if v < 0:
                    return QColor(theme.RED)
                if v > 0:
                    return QColor(theme.GREEN)
            if col == "side":
                return QColor(theme.RED if v == "SELL" else theme.GREEN)
            if col in ("wash_status", "status"):
                s = str(v)
                if s in ("SAFE", "promoted", "optimal", "ok"):
                    return QColor(theme.GREEN)
                if s in ("WASH", "BLOCKED_FORWARD", "WOULD_WASH", "rejected", "failed"):
                    return QColor(theme.RED)
                if s in ("PARTIAL_WASH", "tested", "drafted", "approved"):
                    return QColor(theme.AMBER)
            if col == "term":
                return QColor(theme.AMBER if v == "ST" else theme.ACCENT2)
        if role == Qt.UserRole:
            return v
        if role == Qt.ToolTipRole and col in ("wash_explanation", "constraint", "explanation", "rationale"):
            return str(v)
        return None

    def sort_key(self, row: int, column: int):
        return self._vals[row][column]

    def row_dict(self, row: int) -> dict:
        return self._df.iloc[row].to_dict()


class FrameProxy(QSortFilterProxyModel):
    """Sort numerically on raw values; filter on any column text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.setFilterKeyColumn(-1)
        self.setSortRole(Qt.UserRole)

    def lessThan(self, left, right):
        a = self.sourceModel().data(left, Qt.UserRole)
        b = self.sourceModel().data(right, Qt.UserRole)
        try:
            if a is None or (isinstance(a, float) and np.isnan(a)):
                return True
            if b is None or (isinstance(b, float) and np.isnan(b)):
                return False
            return bool(a < b)          # numpy bools are not accepted by Qt's C++ override
        except TypeError:
            return str(a) < str(b)

    def source_row_dict(self, proxy_index) -> dict:
        src = self.mapToSource(proxy_index)
        return self.sourceModel().row_dict(src.row())
