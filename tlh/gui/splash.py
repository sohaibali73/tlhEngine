"""Splash screen drawn in code (no image assets) so the app has a window on screen within ~200 ms."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QSplashScreen

from . import theme


def make_splash() -> QSplashScreen:
    w, h = 560, 300
    pm = QPixmap(w, h)
    pm.fill(QColor(theme.BG))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.fillRect(0, 0, 8, h, QColor(theme.ACCENT))
    p.setPen(QColor(theme.TEXT))
    f = QFont("Segoe UI", 30, QFont.Bold)
    p.setFont(f)
    p.drawText(40, 110, "TLH ENGINE")
    p.setPen(QColor(theme.ACCENT2))
    p.setFont(QFont("Segoe UI", 13, QFont.DemiBold))
    p.drawText(40, 150, "with YANG · tax-loss harvesting for real portfolios")
    p.setPen(QColor(theme.MUTED))
    p.setFont(QFont("Segoe UI", 10))
    p.drawText(40, 190, "Lot-level tax engine · wash-sale compliance · equity risk models · convex optimiser")
    p.drawText(40, 212, "Nothing places orders. Trade tickets are the terminal output.")
    p.end()
    s = QSplashScreen(pm, Qt.WindowStaysOnTopHint)
    s.setFont(QFont("Segoe UI", 9))
    return s


class _Msg(QSplashScreen):
    pass


def show_message(splash: QSplashScreen, text: str) -> None:
    splash.showMessage(text, Qt.AlignBottom | Qt.AlignLeft, QColor(theme.MUTED))
