"""Tactical overlay signals: where the target beta comes from.

Potomac's tactical strategies are proprietary; this module is the plug-in point for them. A signal is a daily series of
target beta in [0, beta_max] (or a risk-on / risk-off state that maps to two betas). Sources:

    manual        one number, set by the operator or YANG ("today's target beta is 1.2")
    potomac       a Potomac strategy (Bull Bear, Focused Growth, Guardian, Income Plus, Navigrowth) read from its funds' NAVs
                  (optim/potomac.py): flat NAV = risk-off, otherwise rolling beta; 80/5/5/5/5 target allocations
    csv           a file exported from a Potomac strategy: columns date + (target_beta | state | score)
    rule:*        transparent example rules for testing and demonstration (trend, volatility regime, composite);
                  they are NOT Potomac's models
    blend         weighted average of several signals (e.g. three Potomac strategies with allocations)

Timing: every signal except `manual` is generated on the prior close and traded on the next close (`lag_days = 1`): the
target beta used on day t is the value observed at t-1.

Signals are persisted as Parquet under var/tactical/ with a small registry in the settings table. This module is
AI-editable (ai/registry.py): YANG can add rules or blends.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RULES = {
    "rule:trend": "risk-on (beta_max) when the index is above its 200-day average, risk-off (beta_min) below; 3% band to avoid whipsaw",
    "rule:vol_regime": "beta scaled by target vol / realised 21-day vol, clipped to [beta_min, beta_max]",
    "rule:composite": "average of trend and vol_regime",
    "rule:drawdown": "risk-off when the index is more than 8% below its 1-year high, risk-on otherwise",
}


@dataclass
class SignalSpec:
    name: str = "manual"
    kind: str = "manual"               # manual | potomac | csv | rule:trend | rule:vol_regime | rule:composite | rule:drawdown | blend
    strategy: str | None = None        # potomac: Bull Bear | Focused Growth | Guardian | Income Plus | Navigrowth
    lag_days: int = 1                  # signal known at the prior close, traded at the next close (0 = same-day, for research only)
    nav_window: int = 60               # potomac: risk-on days used for the slow beta (what a risk-on day is worth)
    nav_confirm_days: int = 1          # potomac: consecutive flat-NAV days before calling a fund risk-off
    beta_min: float = 0.0
    beta_max: float = 1.5
    manual_beta: float = 1.0
    path: str | None = None            # csv
    date_col: str = "date"
    value_col: str | None = None       # target_beta | state | score (auto-detected)
    target_vol: float = 0.12           # rule:vol_regime
    components: list[dict] = field(default_factory=list)   # blend: [{"name": ..., "weight": ...}]
    description: str = ""


def _clip(s: pd.Series, spec: SignalSpec) -> pd.Series:
    return s.clip(spec.beta_min, spec.beta_max)


def rule_signal(index_prices: pd.Series, spec: SignalSpec) -> pd.Series:
    px = index_prices.dropna()
    if spec.kind == "rule:trend":
        ma = px.rolling(200, min_periods=100).mean()
        state = pd.Series(np.nan, index=px.index)
        state[px > ma * 1.03] = 1.0
        state[px < ma * 0.97] = 0.0
        state = state.ffill().fillna(1.0)
        return _clip(spec.beta_min + state * (spec.beta_max - spec.beta_min), spec)
    if spec.kind == "rule:vol_regime":
        vol = px.pct_change().rolling(21).std() * np.sqrt(252)
        tgt = (spec.target_vol / vol).fillna(1.0)
        return _clip(tgt.ewm(halflife=5).mean(), spec)
    if spec.kind == "rule:drawdown":
        dd = px / px.rolling(252, min_periods=60).max() - 1
        state = (dd > -0.08).astype(float)
        return _clip(spec.beta_min + state * (spec.beta_max - spec.beta_min), spec)
    if spec.kind == "rule:composite":
        a = rule_signal(index_prices, SignalSpec(**{**spec.__dict__, "kind": "rule:trend"}))
        b = rule_signal(index_prices, SignalSpec(**{**spec.__dict__, "kind": "rule:vol_regime"}))
        return _clip((a + b) / 2, spec)
    raise ValueError(f"unknown rule {spec.kind}")


def load_csv_signal(path: str | Path, spec: SignalSpec) -> pd.Series:
    """CSV from a Potomac strategy. Accepts target_beta directly, a state column (risk_on/risk_off, 1/0, long/cash) or a
    score in [-1, 1] / [0, 1] mapped linearly onto [beta_min, beta_max]."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    dcol = cols.get(spec.date_col.lower()) or next((cols[c] for c in cols if "date" in c or "time" in c), df.columns[0])
    vcol = spec.value_col or next((cols[c] for c in ("target_beta", "beta", "state", "signal", "score", "exposure", "allocation") if c in cols), None)
    if vcol is None:
        vcol = [c for c in df.columns if c != dcol][0]
    s = df.set_index(pd.to_datetime(df[dcol]))[vcol]
    if not pd.api.types.is_numeric_dtype(s):
        m = s.astype(str).str.lower().str.strip()
        on = m.isin(["risk_on", "risk-on", "on", "long", "invested", "bull", "1", "true", "yes"])
        off = m.isin(["risk_off", "risk-off", "off", "cash", "flat", "bear", "0", "false", "no"])
        val = pd.Series(np.nan, index=s.index)
        val[on] = spec.beta_max
        val[off] = spec.beta_min
        val = val.fillna(pd.to_numeric(s, errors="coerce"))
        s = val
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    lo, hi = float(s.min()), float(s.max())
    if hi <= 1.0 and lo >= -1.0 and vcol.lower() in ("score", "signal", "exposure", "allocation") or (hi <= 1.0 and lo < 0):
        s = spec.beta_min + (s - lo) / max(hi - lo, 1e-9) * (spec.beta_max - spec.beta_min)
    return _clip(s, spec)


def build_signal(spec: SignalSpec, index_prices: pd.Series | None = None, library: dict[str, pd.Series] | None = None,
                 dates: pd.DatetimeIndex | None = None, navs: pd.DataFrame | None = None, cache_dir=None) -> pd.Series:
    if spec.kind == "manual":
        idx = dates if dates is not None else (index_prices.index if index_prices is not None else pd.DatetimeIndex([pd.Timestamp.today().normalize()]))
        return pd.Series(float(np.clip(spec.manual_beta, spec.beta_min, spec.beta_max)), index=idx, name=spec.name)
    if spec.kind == "potomac":
        from .potomac import strategy_signal
        if not spec.strategy:
            raise ValueError("potomac signal needs a strategy name")
        s, _info = strategy_signal(spec.strategy, spec.beta_min, spec.beta_max, lag_days=spec.lag_days, navs=navs, cache_dir=cache_dir, window=spec.nav_window,
                                   confirm_days=spec.nav_confirm_days)
    elif spec.kind == "csv":
        if not spec.path:
            raise ValueError("csv signal needs a path")
        s = load_csv_signal(spec.path, spec)
    elif spec.kind.startswith("rule:"):
        if index_prices is None:
            raise ValueError("rule signals need index prices")
        s = rule_signal(index_prices, spec)
    elif spec.kind == "blend":
        if not spec.components or not library:
            raise ValueError("blend needs components and a signal library")
        parts, wts = [], []
        for c in spec.components:
            if c["name"] in library:
                parts.append(library[c["name"]])
                wts.append(float(c.get("weight", 1.0)))
        if not parts:
            raise ValueError("no blend components found")
        idx = parts[0].index
        for p in parts[1:]:
            idx = idx.union(p.index)
        s = sum(w * p.reindex(idx).ffill() for w, p in zip(wts, parts, strict=True)) / sum(wts)
        s = _clip(s, spec).dropna()          # dates before a component starts are dropped, not guessed
    else:
        raise ValueError(f"unknown signal kind {spec.kind}")
    if spec.kind in ("csv", "blend") or spec.kind.startswith("rule:"):
        s = s.shift(spec.lag_days).dropna() if spec.lag_days else s    # generated at the prior close, traded at the next close
    if dates is not None:
        s = s.reindex(dates).ffill()
    s.name = spec.name
    return s


def signal_stats(s: pd.Series) -> dict:
    s = s.dropna()
    changes = (s.diff().abs() > 0.05).sum()               # moves of at least 0.05 beta count as a change
    return {"n_days": int(len(s)), "start": str(s.index.min().date()) if len(s) else None, "end": str(s.index.max().date()) if len(s) else None,
            "mean_beta": float(s.mean()) if len(s) else None, "pct_days_risk_on": float((s >= s.max() - 1e-9).mean()) if len(s) else None,
            "pct_days_min": float((s <= s.min() + 1e-9).mean()) if len(s) else None, "changes": int(changes),
            "changes_per_year": float(changes / max(len(s) / 252.0, 1e-9)) if len(s) else None, "latest": float(s.iloc[-1]) if len(s) else None}


class SignalStore:
    """Parquet files under root/, registry in a dict the caller persists (settings table)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60]
        return self.root / f"{safe}.parquet"

    def save(self, name: str, series: pd.Series) -> Path:
        p = self.path(name)
        series.rename("target_beta").to_frame().to_parquet(p)
        return p

    _cache: dict[str, tuple[float, pd.Series]] = {}

    def load(self, name: str) -> pd.Series | None:
        p = self.path(name)
        if not p.exists():
            return None
        mtime = p.stat().st_mtime
        hit = SignalStore._cache.get(str(p))
        if hit is not None and hit[0] == mtime:
            return hit[1].copy()
        df = pd.read_parquet(p)
        s = df["target_beta"]
        s.index = pd.to_datetime(s.index)
        SignalStore._cache[str(p)] = (mtime, s)
        return s.copy()

    def delete(self, name: str) -> None:
        p = self.path(name)
        if p.exists():
            p.unlink()
