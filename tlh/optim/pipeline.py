"""TLH model pipelines: the schema behind the drag-and-drop builder.

A pipeline is an ordered chain of typed nodes (left to right on the canvas). Node parameters are declared here so
the GUI can render property editors generically and YANG can author pipelines as JSON. Pure helpers (`validate`,
`ordered`, `apply_filter`, `rank_keep`, `spec_from_nodes`) have no I/O; execution lives in
`services/pipeline_service.py`. This module is AI-editable (ai/registry.py).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .strategies import STRATEGIES, StrategySpec

# ---------------------------------------------------------------------------------------- node schema
NODE_TYPES: dict[str, dict] = {
    "universe": {
        "title": "Universe", "category": "Source", "color": "#3B82F6",
        "help": "Where candidate names come from. Start every pipeline with one Universe block.",
        "params": [
            {"name": "source", "label": "Source", "type": "choice", "choices": ["model", "watchlist", "basket", "custom"], "default": "model",
             "help": "model = every stock in the active risk model; watchlist = a Norgate watchlist; basket = a saved basket's names; custom = symbols below"},
            {"name": "name", "label": "Watchlist / basket / symbols", "type": "str", "default": "", "help": "e.g. 'S&P 500', 'Core 40', or 'AAPL, MSFT, NVDA'"},
            {"name": "exclude_etps", "label": "Exclude ETFs/ETNs", "type": "bool", "default": True},
            {"name": "exclude_held", "label": "Exclude currently held", "type": "bool", "default": False},
            {"name": "exclude_wash_blocked", "label": "Exclude wash-blocked buys", "type": "bool", "default": True},
        ]},
    "filter": {
        "title": "Filter", "category": "Screen", "color": "#14B8A6",
        "help": "Screen the universe on size, sector and trailing return/vol.",
        "params": [
            {"name": "min_mktcap_musd", "label": "Min market cap ($M)", "type": "float", "default": 0.0},
            {"name": "sectors_include", "label": "Sectors to include", "type": "str", "default": "", "help": "comma-separated GICS sectors; blank = all"},
            {"name": "sectors_exclude", "label": "Sectors to exclude", "type": "str", "default": ""},
            {"name": "min_ret_1y", "label": "Min 1y return", "type": "float", "default": -9.0, "help": "-9 = off (fraction, e.g. 0.05)"},
            {"name": "max_vol_1y", "label": "Max 1y vol", "type": "float", "default": 9.0, "help": "9 = off (fraction, e.g. 0.35)"},
            {"name": "exclude_symbols", "label": "Exclude symbols", "type": "str", "default": ""},
        ]},
    "rank": {
        "title": "Rank & keep top N", "category": "Screen", "color": "#A78BFA",
        "help": "Score names on a composite of style signals and keep the best N.",
        "params": [
            {"name": "signal_weights", "label": "Signal weights", "type": "str", "default": "momentum=1, quality=0.5",
             "help": "styles: momentum, value, quality, size, lowvol, growth (and any custom factor)"},
            {"name": "top_n", "label": "Keep top N", "type": "int", "default": 100},
            {"name": "ascending", "label": "Lowest scores first", "type": "bool", "default": False},
        ]},
    "benchmark": {
        "title": "Benchmark", "category": "Reference", "color": "#94A3B8",
        "help": "What tracking error is measured against for the rest of the pipeline.",
        "params": [{"name": "name", "label": "Watchlist / ETF / basket:<name>", "type": "str", "default": "S&P 500"}],
    },
    "construct": {
        "title": "Construction", "category": "Portfolio", "color": "#F59E0B",
        "help": "Turn the screened universe into weights with a construction strategy.",
        "params": [
            {"name": "strategy", "label": "Strategy", "type": "choice", "choices": [k for k in STRATEGIES if k != "tax_aware_transition"], "default": "min_variance"},
            {"name": "n_max", "label": "Max names", "type": "int", "default": 50},
            {"name": "max_weight", "label": "Max weight", "type": "float", "default": 0.08},
            {"name": "sector_band", "label": "Sector band (0 = off)", "type": "float", "default": 0.02},
            {"name": "signal_weights", "label": "Alpha signals (mean-variance)", "type": "str", "default": "momentum=1"},
            {"name": "ic", "label": "IC", "type": "float", "default": 0.05},
            {"name": "risk_aversion", "label": "Risk aversion", "type": "float", "default": 5.0},
            {"name": "tilts", "label": "Style tilts (factor_tilt)", "type": "str", "default": ""},
            {"name": "views", "label": "BL views (JSON)", "type": "str", "default": ""},
            {"name": "cvar_alpha", "label": "CVaR confidence", "type": "float", "default": 0.95},
            {"name": "cov_source", "label": "Covariance", "type": "choice", "choices": ["model", "sample"], "default": "model"},
        ]},
    "transition": {
        "title": "Tax-aware transition", "category": "Portfolio", "color": "#EF4444",
        "help": "Move the live portfolio toward the constructed weights within a net realised-gain budget.",
        "params": [
            {"name": "gain_budget", "label": "Net gain budget (fraction of value)", "type": "float", "default": 0.01},
            {"name": "turnover_max", "label": "Turnover cap", "type": "float", "default": 0.5},
        ]},
    "harvest": {
        "title": "Harvest toward result", "category": "Trade", "color": "#22C55E",
        "help": "Run the wash-safe harvest optimizer with the constructed weights as the benchmark.",
        "params": [
            {"name": "mode", "label": "Mode", "type": "choice", "choices": ["full_rebalance", "opportunistic"], "default": "full_rebalance"},
            {"name": "te_budget", "label": "TE budget", "type": "float", "default": 0.02},
            {"name": "te_hard", "label": "Hard TE cap", "type": "bool", "default": False},
            {"name": "min_trade_value", "label": "Min trade ($)", "type": "float", "default": 500.0},
        ]},
    "output": {
        "title": "Save / export", "category": "Output", "color": "#60A5FA",
        "help": "Save the weights as a basket, optionally make it the app benchmark and export the harvest workbook.",
        "params": [
            {"name": "basket_name", "label": "Save as basket", "type": "str", "default": ""},
            {"name": "set_benchmark", "label": "Set as app benchmark", "type": "bool", "default": False},
            {"name": "export_excel", "label": "Export harvest workbook", "type": "bool", "default": False},
            {"name": "description", "label": "Description", "type": "str", "default": ""},
        ]},
}

ORDER_HINT = ["universe", "filter", "rank", "benchmark", "construct", "transition", "harvest", "output"]


@dataclass
class Node:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0

    def title(self) -> str:
        return NODE_TYPES[self.type]["title"]


@dataclass
class Pipeline:
    name: str = "Untitled TLH model"
    nodes: list[Node] = field(default_factory=list)
    description: str = ""

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "description": self.description, "nodes": [asdict(n) for n in self.nodes]}, indent=2)

    @classmethod
    def from_json(cls, text: str | dict) -> Pipeline:
        d = json.loads(text) if isinstance(text, str) else text
        nodes = [Node(**{k: v for k, v in n.items() if k in ("id", "type", "params", "x", "y")}) for n in d.get("nodes", [])]
        return cls(name=d.get("name", "Untitled TLH model"), nodes=nodes, description=d.get("description", ""))


def defaults(node_type: str) -> dict:
    return {p["name"]: p["default"] for p in NODE_TYPES[node_type]["params"]}


def new_node(node_type: str, node_id: str, x: float = 0.0, y: float = 0.0, **params) -> Node:
    d = defaults(node_type)
    d.update(params)
    return Node(node_id, node_type, d, x, y)


def ordered(p: Pipeline) -> list[Node]:
    """Execution order = left-to-right on the canvas, ties broken by the natural stage order."""
    return sorted(p.nodes, key=lambda n: (round(n.x / 40), ORDER_HINT.index(n.type) if n.type in ORDER_HINT else 99, n.y))


def validate(p: Pipeline) -> list[str]:
    errs: list[str] = []
    types = [n.type for n in ordered(p)]
    if not p.nodes:
        return ["pipeline is empty: drag a Universe block onto the canvas"]
    if types.count("universe") != 1:
        errs.append("exactly one Universe block is required")
    elif types[0] != "universe":
        errs.append("the Universe block must be the left-most block")
    if types.count("construct") > 1:
        errs.append("only one Construction block is allowed")
    if "transition" in types and "construct" not in types:
        errs.append("Tax-aware transition needs a Construction block before it")
    if "harvest" in types and "construct" not in types:
        errs.append("Harvest needs a Construction block before it")
    if "universe" in types:
        ui = types.index("universe")
        for stage in ("construct", "transition", "harvest", "output"):
            if stage in types and types.index(stage) < ui:
                errs.append(f"{NODE_TYPES[stage]['title']} must come after Universe")
    if "construct" in types:
        ci = types.index("construct")
        for stage in ("filter", "rank"):
            if any(t == stage and i > ci for i, t in enumerate(types)):
                errs.append(f"{NODE_TYPES[stage]['title']} blocks must come before Construction")
    for n in p.nodes:
        if n.type not in NODE_TYPES:
            errs.append(f"unknown block type '{n.type}'")
    return errs


# ---------------------------------------------------------------------------------------- pure stage helpers
def parse_kv(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(text or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def parse_list(text: str) -> list[str]:
    return [s.strip().upper() for s in str(text or "").replace(";", ",").split(",") if s.strip()]


def apply_filter(symbols: list[str], securities: pd.DataFrame, mktcap_musd: pd.Series | None, ret_1y: pd.Series | None,
                 vol_1y: pd.Series | None, params: dict) -> list[str]:
    keep = list(symbols)
    if params.get("min_mktcap_musd", 0) and mktcap_musd is not None:
        keep = [s for s in keep if float(mktcap_musd.get(s, 0) or 0) >= float(params["min_mktcap_musd"])]
    inc = {s.lower() for s in str(params.get("sectors_include", "")).split(",") if s.strip()}
    exc = {s.lower() for s in str(params.get("sectors_exclude", "")).split(",") if s.strip()}
    if (inc or exc) and "gics_sector" in securities:
        sec = securities["gics_sector"].reindex(keep).fillna("").str.lower()
        if inc:
            keep = [s for s in keep if any(i in sec[s] for i in inc)]
        if exc:
            keep = [s for s in keep if not any(e in sec[s] for e in exc)]
    if ret_1y is not None and float(params.get("min_ret_1y", -9)) > -9:
        keep = [s for s in keep if float(ret_1y.get(s, np.nan)) >= float(params["min_ret_1y"])]
    if vol_1y is not None and float(params.get("max_vol_1y", 9)) < 9:
        keep = [s for s in keep if float(vol_1y.get(s, np.nan)) <= float(params["max_vol_1y"])]
    ex = set(parse_list(params.get("exclude_symbols", "")))
    return [s for s in keep if s not in ex]


def rank_keep(symbols: list[str], signals: pd.DataFrame, params: dict) -> tuple[list[str], pd.Series]:
    w = parse_kv(params.get("signal_weights", "momentum=1")) or {"momentum": 1.0}
    score = pd.Series(0.0, index=symbols)
    for k, wt in w.items():
        if k in signals.columns:
            score = score + wt * signals[k].reindex(symbols).fillna(0.0)
    score = score.sort_values(ascending=bool(params.get("ascending", False)))
    n = int(params.get("top_n", 100) or len(score))
    return list(score.index[:n]), score


def spec_from_nodes(construct: dict, transition: dict | None = None) -> StrategySpec:
    views = []
    if construct.get("views"):
        try:
            views = json.loads(construct["views"])
        except json.JSONDecodeError:
            views = []
    band = float(construct.get("sector_band", 0) or 0)
    spec = StrategySpec(kind=construct.get("strategy", "min_variance"), n_max=int(construct.get("n_max") or 0) or None,
                        max_weight=float(construct.get("max_weight", 0.08)), sector_band=band if band > 0 else None,
                        signal_weights=parse_kv(construct.get("signal_weights", "momentum=1")) or {"momentum": 1.0},
                        ic=float(construct.get("ic", 0.05)), risk_aversion=float(construct.get("risk_aversion", 5.0)),
                        tilts=parse_kv(construct.get("tilts", "")), views=views, cvar_alpha=float(construct.get("cvar_alpha", 0.95)))
    if transition:
        spec.gain_budget = float(transition.get("gain_budget", 0.01))
        spec.turnover_max = float(transition.get("turnover_max", 0.5))
    return spec


EXAMPLES: dict[str, Pipeline] = {
    "Quality core + harvest": Pipeline("Quality core + harvest", [
        new_node("universe", "u1", 0, 120, source="model", exclude_wash_blocked=True),
        new_node("filter", "f1", 200, 120, min_mktcap_musd=10000),
        new_node("rank", "r1", 400, 120, signal_weights="quality=1, lowvol=0.5", top_n=150),
        new_node("benchmark", "b1", 600, 120, name="S&P 500"),
        new_node("construct", "c1", 800, 120, strategy="min_variance", n_max=40, max_weight=0.06, sector_band=0.03),
        new_node("harvest", "h1", 1000, 120, mode="full_rebalance", te_budget=0.02),
        new_node("output", "o1", 1200, 120, basket_name="Quality core 40", set_benchmark=False),
    ], "Screen to liquid quality names, build a 40-name min-variance core, harvest toward it."),
    "Momentum tilt via mean-variance": Pipeline("Momentum tilt via mean-variance", [
        new_node("universe", "u1", 0, 120, source="watchlist", name="S&P 500"),
        new_node("benchmark", "b1", 250, 120, name="S&P 500"),
        new_node("construct", "c1", 500, 120, strategy="mean_variance", n_max=60, signal_weights="momentum=1, growth=0.3", ic=0.05, risk_aversion=8),
        new_node("output", "o1", 750, 120, basket_name="Momentum 60"),
    ], "Benchmark-relative mean-variance tilt toward momentum and growth."),
    "Transition to a core basket": Pipeline("Transition to a core basket", [
        new_node("universe", "u1", 0, 120, source="basket", name="Core 40"),
        new_node("construct", "c1", 250, 120, strategy="cap_weight", n_max=40, max_weight=0.1),
        new_node("transition", "t1", 500, 120, gain_budget=0.005, turnover_max=0.35),
        new_node("harvest", "h1", 750, 120, mode="full_rebalance", te_budget=0.015),
        new_node("output", "o1", 1000, 120, basket_name="Core 40 transition target"),
    ], "Take an existing basket, plan a gain-budgeted transition and the wash-safe trades toward it."),
}
