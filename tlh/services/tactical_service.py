"""Tactical overlay service: Potomac signals -> target beta -> leveraged / inverse ETF overlay on the tax-sensitive core.

Signals are stored as Parquet under var/tactical with a registry in the settings table (`tactical_signals`); the active
signal name and today's target beta live in settings too. Nothing here trades.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd

from ..optim.leverage import (
    DEFAULT_INVERSE,
    DEFAULT_LONG_LEVERED,
    INSTRUMENTS,
    MarginPolicy,
    instrument_table,
    simulate_tactical,
    tactical_overlay,
)
from ..optim.tactical import RULES, SignalSpec, SignalStore, build_signal, signal_stats
from .context import AppContext
from .data_service import DataService
from .risk_service import RiskService

log = logging.getLogger(__name__)


class TacticalService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.data = DataService(ctx)
        self.risk = RiskService(ctx)
        self.store = SignalStore(ctx.settings.var_dir / "tactical")

    # ------------------------------------------------------------------ policy
    def policy(self) -> MarginPolicy:
        saved = self.ctx.get("margin_policy")
        if saved:
            try:
                return MarginPolicy(**{k: v for k, v in saved.items() if k in MarginPolicy.__dataclass_fields__})
            except TypeError:
                pass
        return MarginPolicy()

    def save_policy(self, pol: MarginPolicy) -> None:
        self.ctx.set("margin_policy", asdict(pol))

    @staticmethod
    def instruments() -> pd.DataFrame:
        return instrument_table()

    @staticmethod
    def rules() -> dict[str, str]:
        return dict(RULES)

    # ------------------------------------------------------------------ signals
    def registry(self) -> dict[str, dict]:
        return dict(self.ctx.get("tactical_signals", {}) or {})

    def list_signals(self) -> pd.DataFrame:
        rows = []
        for name, meta in self.registry().items():
            s = self.store.load(name)
            st = signal_stats(s) if s is not None else {}
            rows.append({"name": name, "kind": meta.get("kind"), "description": meta.get("description", ""), **st, "active": name == self.active_name()})
        return pd.DataFrame(rows)

    def active_name(self) -> str | None:
        return self.ctx.get("tactical_active_signal")

    def set_active(self, name: str | None) -> None:
        self.ctx.set("tactical_active_signal", name)

    def index_prices(self) -> pd.Series | None:
        snap = self.data.latest_snapshot()
        if snap is None:
            return None
        close = snap.close_matrix("close")
        for sym in ("SPY", "IVV", "VOO"):
            if sym in close.columns:
                return close[sym].dropna()
        return None

    def save_signal(self, spec: SignalSpec) -> dict:
        idx = self.index_prices()
        library = {n: self.store.load(n) for n in self.registry()} if spec.kind == "blend" else None
        s = build_signal(spec, index_prices=idx, library=library)
        self.store.save(spec.name, s)
        reg = self.registry()
        reg[spec.name] = {"kind": spec.kind, "description": spec.description or RULES.get(spec.kind, ""), "spec": asdict(spec)}
        self.ctx.set("tactical_signals", reg)
        if self.active_name() is None:
            self.set_active(spec.name)
        self.ctx.db.audit("user", "tactical.signal.save", spec.name, kind=spec.kind, **{k: v for k, v in signal_stats(s).items() if k in ("n_days", "latest")})
        return {"name": spec.name, **signal_stats(s)}

    def delete_signal(self, name: str) -> None:
        reg = self.registry()
        reg.pop(name, None)
        self.ctx.set("tactical_signals", reg)
        self.store.delete(name)
        if self.active_name() == name:
            self.set_active(None)

    def signal(self, name: str | None = None) -> pd.Series | None:
        name = name or self.active_name()
        return self.store.load(name) if name else None

    def target_beta_today(self, name: str | None = None) -> float | None:
        s = self.signal(name)
        return float(s.dropna().iloc[-1]) if s is not None and len(s.dropna()) else None

    # ------------------------------------------------------------------ current overlay recommendation
    def core_state(self, entity_id: int | None = None) -> dict:
        """Core value, cov-implied beta of the stock sleeve, existing overlay legs, cash proxy, prices."""
        from .portfolio_service import PortfolioService
        eid = entity_id or self.ctx.current_entity_id
        snap = self.data.latest_snapshot()
        act = self.risk.active()
        if eid is None or snap is None or act is None:
            raise RuntimeError("need holdings, a snapshot and an active risk model")
        model = act[1]
        lots = PortfolioService(self.ctx).lots_view(eid, snap=snap)
        if lots.empty:
            raise RuntimeError("no holdings")
        mv = lots.groupby("symbol")["market_value"].sum()
        overlay = {s: float(v) for s, v in mv.items() if s in INSTRUMENTS}
        core = mv[[s for s in mv.index if s not in INSTRUMENTS and s in model.symbols]]
        core_value = float(core.sum())
        # beta is measured against the *index* (target beta means beta to the S&P), never against a saved basket that may
        # itself hold leveraged funds; leveraged / inverse funds are stripped from the benchmark weights
        name = self.risk.benchmark_name()
        bench = self.risk.benchmark_weights(snap, model, self.ctx.settings.default_benchmark if name.lower().startswith("basket:") else name)
        bench = bench[[s for s in bench.index if s not in INSTRUMENTS]]
        bench = bench / bench.sum()
        syms = sorted(set(core.index) | set(bench.index) | {s for s in INSTRUMENTS if s in model.symbols})
        S = model.covariance(syms).values
        wb = bench.reindex(syms).fillna(0.0).values
        wb = wb / wb.sum()
        betas = pd.Series((S @ wb) / float(wb @ S @ wb), index=syms)
        from ..optim.leverage import nominal_betas
        # the benchmark is the index itself, so a k-times fund has beta k (a fitted SPY row under-states even SPY's beta)
        betas = nominal_betas(betas, None, [s for s in INSTRUMENTS if s in betas.index])
        w = core.reindex(syms).fillna(0.0).values / max(core_value, 1e-9)
        core_beta = float(betas.values @ w)
        gains = lots[lots["symbol"].isin(core.index)]
        gain_frac = float(gains["unrealized"].clip(lower=0).sum() / max(core_value, 1e-9))
        return {"entity_id": eid, "core_value": core_value, "core_beta": core_beta, "existing_overlay": overlay,
                "instrument_betas": {s: float(betas[s]) for s in INSTRUMENTS if s in betas.index}, "core_gain_frac": gain_frac,
                "prices": snap.last_prices(), "index_vol": float(np.sqrt(wb @ S @ wb)), "benchmark": self.ctx.settings.default_benchmark if name.lower().startswith("basket:") else name,
                "rf": self._rf(snap), "unknown_value": float(mv.sum() - core_value - sum(overlay.values()))}

    @staticmethod
    def _rf(snap) -> float:
        """3-month rate from the macro series (fraction), falling back to the module default when unavailable."""
        from ..optim.leverage import DEFAULT_RF
        try:
            macro = snap.macro()
            if not macro.empty and "rate_3m" in macro and macro["rate_3m"].notna().any():
                return float(macro["rate_3m"].dropna().iloc[-1]) / 100
        except Exception:
            pass
        return DEFAULT_RF

    def recommend(self, target_beta: float | None = None, cash: float = 0.0, long_instruments: tuple[str, ...] | None = None,
                  inverse_instruments: tuple[str, ...] | None = None, entity_id: int | None = None) -> dict:
        st = self.core_state(entity_id)
        tb = target_beta if target_beta is not None else self.target_beta_today()
        if tb is None:
            raise RuntimeError("no target beta: set a manual beta or save a signal first")
        prof = self.ctx.tax.default_profile()
        out = tactical_overlay(st["core_value"], st["core_beta"], float(tb), self.policy(), long_instruments or DEFAULT_LONG_LEVERED,
                               inverse_instruments or DEFAULT_INVERSE, instrument_betas=st["instrument_betas"], cash=float(cash),
                               index_vol=st["index_vol"], existing_overlay=st["existing_overlay"], core_gain_frac=st["core_gain_frac"],
                               lt_rate=prof.lt_rate, prices=st["prices"], rf=st.get("rf", 0.04))
        out["core"] = {k: v for k, v in st.items() if k not in ("prices",)}
        out["signal"] = self.active_name()
        out["as_of"] = date.today().isoformat()
        self.ctx.db.audit("user", "tactical.recommend", str(st["entity_id"]), target_beta=float(tb), beta_now=out["beta_now"])
        return out

    # ------------------------------------------------------------------ backtest
    def backtest(self, name: str | None = None, start: str | None = None, long_instrument: str = "SSO", inverse_instrument: str = "SDS",
                 use_core_returns: bool = True, entity_id: int | None = None, rebalance_threshold: float = 0.1) -> dict:
        s = self.signal(name)
        idx = self.index_prices()
        if s is None or idx is None:
            raise RuntimeError("need a saved signal and an index price series (SPY) in the snapshot")
        r_idx = idx.pct_change().dropna()
        if start:
            r_idx = r_idx.loc[pd.Timestamp(start):]
        core_r = None
        if use_core_returns:
            try:
                st = self.core_state(entity_id)
                snap = self.data.latest_snapshot()
                close = snap.close_matrix("close")
                core_syms = [c for c in close.columns if c not in INSTRUMENTS and c in st["prices"].index]
                from .portfolio_service import PortfolioService
                lots = PortfolioService(self.ctx).lots_view(st["entity_id"], snap=snap)
                mv = lots.groupby("symbol")["market_value"].sum()
                mv = mv[[c for c in mv.index if c in core_syms]]
                if len(mv):
                    w = mv / mv.sum()
                    core_r = (close[w.index].pct_change() * w).sum(axis=1).reindex(r_idx.index).fillna(0.0)
            except Exception as e:  # fall back to the index as the core
                log.debug("core returns unavailable, using index: %s", e)
        res = simulate_tactical(r_idx, s, core_returns=core_r, policy=self.policy(), long_instrument=long_instrument,
                                inverse_instrument=inverse_instrument, rebalance_threshold=rebalance_threshold)
        res["signal"] = name or self.active_name()
        res["core_source"] = "holdings" if core_r is not None else "index"
        self.ctx.runs.create("tactical_backtest", date.today(), entity_id or self.ctx.current_entity_id, None, None,
                             {"signal": res["signal"], "long": long_instrument, "inverse": inverse_instrument, "start": start},
                             {k: v for k, v in res["metrics"].items()})
        return res
