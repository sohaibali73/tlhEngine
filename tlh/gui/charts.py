"""Plotly figures rendered inside QWebEngineView with a click bridge back into Qt.

plotly.min.js is written once to var/ and referenced by file URL so charts work offline and render fast.
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView

from ..config import get_settings
from . import theme

_PLOTLY_JS: Path | None = None


def plotly_js_path() -> Path:
    global _PLOTLY_JS
    if _PLOTLY_JS is None:
        import plotly.offline
        p = get_settings().var_dir / "plotly.min.js"
        if not p.exists():
            p.write_text(plotly.offline.get_plotlyjs(), encoding="utf-8")
        _PLOTLY_JS = p
    return _PLOTLY_JS


class _Bridge(QObject):
    clicked = Signal(dict)

    @Slot(str)
    def onClick(self, payload: str) -> None:
        try:
            self.clicked.emit(json.loads(payload))
        except json.JSONDecodeError:
            pass


HTML = """<!doctype html><html><head><meta charset="utf-8">
<script src="{js}"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<style>html,body{{margin:0;padding:0;background:{bg};overflow:hidden}} #c{{width:100vw;height:100vh}}</style>
</head><body><div id="c"></div>
<script>
var fig = {fig};
Plotly.newPlot('c', fig.data, fig.layout, {{responsive:true, displaylogo:false, modeBarButtonsToRemove:['lasso2d','select2d']}});
new QWebChannel(qt.webChannelTransport, function(ch) {{
  var bridge = ch.objects.bridge;
  document.getElementById('c').on('plotly_click', function(d) {{
    var pts = d.points.map(function(p) {{ return {{x: p.x, y: p.y, text: p.text, curve: p.curveNumber, index: p.pointNumber,
      customdata: p.customdata, name: p.data.name}}; }});
    bridge.onClick(JSON.stringify({{points: pts}}));
  }});
}});
window.addEventListener('resize', function() {{ Plotly.Plots.resize('c'); }});
</script></body></html>"""


class PlotlyView(QWebEngineView):
    point_clicked = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bridge = _Bridge()
        self._channel = QWebChannel(self.page())
        self._channel.registerObject("bridge", self._bridge)
        self.page().setWebChannel(self._channel)
        self._bridge.clicked.connect(self.point_clicked)
        self.page().setBackgroundColor(theme.BG)
        self.setMinimumHeight(200)

    def set_figure(self, fig: go.Figure) -> None:
        fig.update_layout(**theme.plotly_layout(**{k: v for k, v in (fig.layout.to_plotly_json() or {}).items()
                                                   if k in ("title", "height", "margin", "barmode", "showlegend", "legend",
                                                            "xaxis", "yaxis", "xaxis2", "yaxis2", "polar", "annotations", "shapes")}))
        js = QUrl.fromLocalFile(str(plotly_js_path())).toString()
        html = HTML.format(js=js, bg=theme.BG, fig=fig.to_json())
        self.setHtml(html, QUrl.fromLocalFile(str(get_settings().var_dir) + "/"))

    def set_message(self, text: str) -> None:
        self.setHtml(f"<html><body style='background:{theme.BG};color:{theme.MUTED};font-family:Segoe UI;padding:20px'>{text}</body></html>")


# ====================================================================================== figure builders
def exposure_bars(exp_table, title: str = "Factor exposures: portfolio vs benchmark") -> go.Figure:
    """exp_table: DataFrame index=factor, columns portfolio/benchmark/active/kind."""
    df = exp_table[exp_table["kind"].isin(["style", "market", "macro"])]
    fig = go.Figure()
    fig.add_bar(name="Portfolio", x=df.index, y=df["portfolio"], marker_color=theme.ACCENT)
    fig.add_bar(name="Benchmark", x=df.index, y=df["benchmark"], marker_color=theme.MUTED)
    fig.add_scatter(name="Active", x=df.index, y=df["active"], mode="markers", marker=dict(color=theme.AMBER, size=9, symbol="diamond"))
    fig.update_layout(title=title, barmode="group", yaxis_title="exposure (z / beta)", height=360)
    return fig


def sector_bars(sec_table, title: str = "Sector weights") -> go.Figure:
    """sec_table: DataFrame index=sector, columns portfolio/benchmark (weights)."""
    fig = go.Figure()
    for col, color in (("portfolio", theme.ACCENT), ("benchmark", theme.MUTED)):
        if col in sec_table:
            fig.add_bar(name=col.title(), x=sec_table.index, y=sec_table[col] * 100, marker_color=color)
    if {"portfolio", "benchmark"} <= set(sec_table.columns):
        fig.add_scatter(name="Active (pp)", x=sec_table.index, y=(sec_table["portfolio"] - sec_table["benchmark"]) * 100,
                        mode="markers", marker=dict(color=theme.AMBER, size=9, symbol="diamond"))
    fig.update_layout(title=title, barmode="group", yaxis_title="%", height=360, xaxis=dict(tickangle=-30))
    return fig


def radar(exp_table, title: str = "Style profile") -> go.Figure:
    df = exp_table[exp_table["kind"] == "style"]
    cats = list(df.index) + [df.index[0]] if len(df) else []
    fig = go.Figure()
    for col, color in (("portfolio", theme.ACCENT), ("benchmark", theme.MUTED)):
        vals = list(df[col]) + [df[col].iloc[0]] if len(df) else []
        fig.add_scatterpolar(r=vals, theta=cats, name=col.title(), line=dict(color=color), fill="toself", opacity=0.6)
    fig.update_layout(title=title, height=360, polar=dict(bgcolor=theme.BG2, radialaxis=dict(gridcolor=theme.BORDER),
                                                            angularaxis=dict(gridcolor=theme.BORDER)))
    return fig


def te_decomposition_bars(dec, title: str = "Tracking-error decomposition") -> go.Figure:
    d = dec.copy()
    d = d[d["variance"].abs() > 0].sort_values("variance", ascending=True)
    te = dec.attrs.get("tracking_error", float("nan"))
    fig = go.Figure(go.Bar(x=d["te_contrib"] * 1e4, y=d.index, orientation="h",
                           marker_color=[theme.RED if v < 0 else theme.ACCENT for v in d["te_contrib"]],
                           hovertemplate="%{y}: %{x:.1f} bps of TE<extra></extra>"))
    fig.update_layout(title=f"{title} — TE {te:.2%}", xaxis_title="contribution to TE (bps)", height=max(300, 22 * len(d) + 80))
    return fig


def harvest_heatmap(positions, title: str = "Harvest opportunities (unrealised loss by position)") -> go.Figure:
    """positions: DataFrame with symbol, unrealized_st, unrealized_lt, market_value, harvestable_loss."""
    df = positions[positions["harvestable_loss"] > 0].sort_values("harvestable_loss", ascending=False).head(40)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No unrealised losses", height=300)
        return fig
    z = [[-min(v, 0) for v in df["unrealized_st"]], [-min(v, 0) for v in df["unrealized_lt"]]]
    fig = go.Figure(go.Heatmap(z=z, x=df["symbol"], y=["Short-term", "Long-term"], colorscale=[[0, theme.BG2], [1, theme.RED]],
                               hovertemplate="%{x} %{y}: $%{z:,.0f}<extra></extra>", colorbar=dict(title="$ loss")))
    fig.update_layout(title=title, height=260, xaxis=dict(tickangle=-45))
    return fig


def treemap_positions(positions, title: str = "Positions (size = value, colour = unrealised %)") -> go.Figure:
    df = positions[positions["market_value"] > 0]
    fig = go.Figure(go.Treemap(labels=df["symbol"], parents=[""] * len(df), values=df["market_value"],
                               marker=dict(colors=df["unrealized_pct"] * 100, colorscale=[[0, theme.RED], [0.5, theme.BG3], [1, theme.GREEN]],
                                           cmid=0, colorbar=dict(title="%")),
                               customdata=df["symbol"], texttemplate="%{label}<br>%{value:$,.0f}",
                               hovertemplate="%{label}: $%{value:,.0f}<br>unrealised %{color:.1f}%<extra></extra>"))
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def wash_calendar_chart(cal, as_of, title: str = "Wash-sale windows (61 days)") -> go.Figure:
    import pandas as pd
    fig = go.Figure()
    if cal is None or cal.empty:
        fig.update_layout(title="No open wash-sale windows", height=260)
        return fig
    colors = {"loss_sale": theme.RED, "purchase": theme.AMBER, "scheduled_drip": theme.PURPLE, "scheduled_buy": theme.PURPLE}
    for _i, r in cal.reset_index(drop=True).iterrows():
        y = f"{r['symbol']} · {r['kind']}"
        fig.add_trace(go.Scatter(x=[pd.Timestamp(r["window_start"]), pd.Timestamp(r["window_end"])], y=[y, y], mode="lines",
                                 line=dict(color=colors.get(r["kind"], theme.ACCENT), width=10), name=r["kind"], showlegend=False,
                                 hovertemplate=f"{r['constraint']}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[pd.Timestamp(r["event_date"])], y=[y], mode="markers", marker=dict(color="white", size=8),
                                 showlegend=False, hovertemplate=f"event {r['event_date']}<extra></extra>"))
    fig.add_vline(x=pd.Timestamp(as_of).timestamp() * 1000, line=dict(color=theme.ACCENT2, dash="dot"))
    fig.update_layout(title=title, height=max(260, 30 * len(cal) + 100), xaxis_title="date")
    return fig


def frontier_chart(fr, current=None, title: str = "Tax alpha vs tracking error") -> go.Figure:
    fig = go.Figure()
    ok = fr[fr["status"].astype(str).str.startswith("optimal")] if "status" in fr else fr
    fig.add_scatter(x=ok["te_after"] * 100, y=ok["tax_alpha"], mode="lines+markers", name="Frontier (TE budget sweep)",
                    line=dict(color=theme.ACCENT), text=[f"budget {b:.2%}" for b in ok["te_budget"]],
                    hovertemplate="TE %{x:.2f}% → tax alpha $%{y:,.0f}<br>%{text}<extra></extra>")
    if current is not None:
        fig.add_scatter(x=[current["te_after"] * 100], y=[current["tax_alpha"]], mode="markers", name="Current recommendation",
                        marker=dict(color=theme.AMBER, size=13, symbol="star"))
    fig.update_layout(title=title, xaxis_title="tracking error after (%)", yaxis_title="tax alpha ($)", height=360)
    return fig


def priority_chart(table, title: str = "Constraint-hierarchy comparison") -> go.Figure:
    fig = go.Figure()
    ok = table[table["status"].astype(str).str.startswith("optimal")] if "status" in table else table
    fig.add_scatter(x=ok["te_after"] * 100, y=ok["tax_alpha"], mode="markers+text", text=ok["priority"], textposition="top center",
                    marker=dict(size=12, color=theme.PURPLE), hovertemplate="%{text}<br>TE %{x:.2f}% · tax alpha $%{y:,.0f}<extra></extra>")
    fig.update_layout(title=title, xaxis_title="tracking error after (%)", yaxis_title="tax alpha ($)", height=360)
    return fig


def cumulative_harvest_chart(closures, title: str = "Cumulative realised losses & tax value") -> go.Figure:
    import pandas as pd
    fig = go.Figure()
    if closures is None or closures.empty:
        fig.update_layout(title="No realised trades yet", height=300)
        return fig
    df = closures.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values("sale_date")
    df["allowed"] = df["realized_gain"] + df["wash_disallowed"]
    losses = df[df["allowed"] < 0]
    fig.add_scatter(x=losses["sale_date"], y=(-losses["allowed"]).cumsum(), mode="lines", name="Cumulative harvested loss", line=dict(color=theme.RED))
    fig.add_scatter(x=df["sale_date"], y=df["allowed"].cumsum(), mode="lines", name="Cumulative net realised", line=dict(color=theme.ACCENT))
    if "tax_value" in df:
        fig.add_scatter(x=df["sale_date"], y=df["tax_value"].cumsum(), mode="lines", name="Cumulative tax value", line=dict(color=theme.GREEN))
    fig.update_layout(title=title, yaxis_title="$", height=340)
    return fig


def factor_returns_chart(fr, title: str = "Cumulative factor returns") -> go.Figure:
    fig = go.Figure()
    cols = [c for c in fr.columns if not c.startswith("sec:")]
    cum = (1 + fr[cols]).cumprod() - 1
    for c in cols:
        fig.add_scatter(x=cum.index, y=cum[c] * 100, mode="lines", name=c)
    fig.update_layout(title=title, yaxis_title="%", height=360)
    return fig


def factor_vol_compare(df, title="Factor volatility by model version") -> go.Figure:
    fig = go.Figure()
    for c in df.columns:
        if c.startswith("vol_"):
            fig.add_bar(name=c, x=df.index, y=df[c] * 100)
    fig.update_layout(title=title, barmode="group", yaxis_title="annualised vol (%)", height=340)
    return fig
