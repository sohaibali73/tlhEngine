"""Dark, data-dense PM-desk theme (Qt stylesheet + palette + Plotly template)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BG = "#0F1419"
BG2 = "#161C24"
BG3 = "#1E2730"
BORDER = "#2A3441"
TEXT = "#D7DEE7"
MUTED = "#8B98A8"
ACCENT = "#3B82F6"
ACCENT2 = "#60A5FA"
GREEN = "#22C55E"
RED = "#EF4444"
AMBER = "#F59E0B"
PURPLE = "#A78BFA"

PLOTLY_COLORWAY = ["#3B82F6", "#F59E0B", "#22C55E", "#A78BFA", "#EF4444", "#14B8A6", "#F472B6", "#94A3B8"]

QSS = f"""
* {{ font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 12px; color: {TEXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
QSplitter::handle {{ background: {BORDER}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {BG}; }}
QTabBar::tab {{ background: {BG2}; padding: 7px 16px; border: 1px solid {BORDER}; border-bottom: none; color: {MUTED}; }}
QTabBar::tab:selected {{ background: {BG}; color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; margin-top: 10px; padding: 8px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {MUTED}; font-weight: 600; }}
QTableView, QTreeView, QListView {{ background: {BG2}; alternate-background-color: {BG3}; gridline-color: {BORDER};
    border: 1px solid {BORDER}; selection-background-color: #1D3A6E; selection-color: {TEXT}; }}
QHeaderView::section {{ background: {BG3}; color: {MUTED}; padding: 4px 6px; border: none; border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER}; font-weight: 600; }}
QTableView QTableCornerButton::section {{ background: {BG3}; border: none; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QPlainTextEdit, QTextEdit {{ background: {BG2}; border: 1px solid {BORDER};
    border-radius: 3px; padding: 4px 6px; selection-background-color: #1D3A6E; }}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{ border: 1px solid {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {BG2}; border: 1px solid {BORDER}; selection-background-color: #1D3A6E; }}
QPushButton {{ background: {BG3}; border: 1px solid {BORDER}; border-radius: 3px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BG2}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT2}; }}
QPushButton[danger="true"] {{ background: #7F1D1D; border-color: #991B1B; color: white; }}
QPushButton[success="true"] {{ background: #14532D; border-color: #166534; color: white; }}
QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel[kpi="true"] {{ font-size: 20px; font-weight: 700; }}
QLabel[kpiLabel="true"] {{ color: {MUTED}; font-size: 11px; text-transform: uppercase; }}
QLabel[h1="true"] {{ font-size: 16px; font-weight: 700; }}
QLabel[banner="error"] {{ background: #7F1D1D; color: white; padding: 6px 10px; border-radius: 3px; }}
QLabel[banner="warn"] {{ background: #78350F; color: white; padding: 6px 10px; border-radius: 3px; }}
QLabel[banner="ok"] {{ background: #14532D; color: white; padding: 6px 10px; border-radius: 3px; }}
QStatusBar {{ background: {BG2}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QToolBar {{ background: {BG2}; border-bottom: 1px solid {BORDER}; spacing: 6px; padding: 3px; }}
QMenuBar {{ background: {BG2}; }} QMenuBar::item:selected {{ background: {BG3}; }}
QMenu {{ background: {BG2}; border: 1px solid {BORDER}; }} QMenu::item:selected {{ background: #1D3A6E; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }} QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar:horizontal {{ background: {BG}; height: 10px; }} QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QProgressBar {{ background: {BG2}; border: 1px solid {BORDER}; border-radius: 3px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QToolTip {{ background: {BG3}; color: {TEXT}; border: 1px solid {BORDER}; }}
QFrame[card="true"] {{ background: {BG2}; border: 1px solid {BORDER}; border-radius: 4px; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(BG2))
    pal.setColor(QPalette.AlternateBase, QColor(BG3))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(BG3))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor("#1D3A6E"))
    pal.setColor(QPalette.HighlightedText, QColor(TEXT))
    pal.setColor(QPalette.ToolTipBase, QColor(BG3))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(MUTED))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    f = QFont("Segoe UI", 9)
    app.setFont(f)


def plotly_layout(**overrides) -> dict:
    base = dict(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=TEXT, family="Segoe UI, Inter, sans-serif", size=12),
        colorway=PLOTLY_COLORWAY, margin=dict(l=50, r=20, t=40, b=40), hoverlabel=dict(bgcolor=BG3, bordercolor=BORDER),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER), yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    base.update(overrides)
    return base
