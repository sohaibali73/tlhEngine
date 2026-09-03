"""Strategy lab: build model portfolios with any construction strategy and backtest them walk-forward."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..optim.backtest import BacktestResult, BacktestSpec, run_backtest
from ..optim.basket import analyze_basket
from ..optim.strategies import STRATEGIES, StrategyInputs, StrategySpec, run_strategy, shrunk_sample_cov
from .basket_service import BasketService
from .context import AppContext
from .data_service import DataService
from .risk_service import RiskService

log = logging.getLogger(__name__)

PARAM_HINTS = {
    "common": ["n_max", "max_weight", "min_weight", "sector_band", "exclude"],
    "mean_variance": ["signal_weights (e.g. {'momentum':1,'quality':0.5})", "ic", "risk_aversion", "benchmark_relative"],
    "black_litterman": ["views [{assets:{SYM:1,...}, return, confidence}]", "tau", "market_sharpe", "risk_aversion (<=0 -> implied delta)"],
    "min_cvar": ["cvar_alpha"], "factor_tilt": ["tilts {style: target active z}", "tilt_weight"],
    "tax_aware_transition": ["target_weights {SYM: w} or target basket", "gain_budget (fraction of value)", "turnover_max", "cost_bps"],
    "stratified_index": ["size_buckets"], "min_variance": ["te_penalty"], "risk_parity": [], "hrp": [], "max_diversification": [],
    "equal_weight": [], "cap_weight": [],
    "multi_factor": ["signal_weights (default value/momentum/quality/lowvol = 1)", "integrated (True) | mixed sleeves (False)", "sector_neutral_scores", "ic", "risk_aversion"],
    "defensive_equity": ["beta_cap (0.85)", "signal_weights (lowvol/quality)", "risk_aversion", "te_penalty"],
    "quality_momentum": ["ic", "risk_aversion", "sector_band"],
    "long_short_extension": ["extension (0.30 = 130/30)", "short_max_weight", "beta_target", "beta_tolerance", "extension_neutral", "signal_weights", "ic", "risk_aversion"],
    "overlay_neutral": ["extension (0.30 = 30/30 around current holdings)", "short_max_weight", "extension_neutral", "signal_weights", "ic", "risk_aversion"],
    "levered_beta": ["target_beta (1.5)", "lev_instruments (SSO, UPRO, SPXL, SPUU)", "etf_max_weight", "margin_max (0 = cash only)", "margin_rate", "margin_buffer", "cost_weight"],
}


class StrategyService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.data = DataService(ctx)
        self.risk = RiskService(ctx)
        self.baskets = BasketService(ctx)

    # ------------------------------------------------------------------ catalogue
    @staticmethod
    def catalogue() -> list[dict]:
        return [{"kind": k, "description": v, "params": PARAM_HINTS["common"] + PARAM_HINTS.get(k, [])} for k, v in STRATEGIES.items()]

    # ------------------------------------------------------------------ inputs
    def inputs(self, strategy: StrategySpec, universe: list[str] | None = None, benchmark_name: str | None = None,
               entity_id: int | None = None, cov_source: str = "model") -> tuple[StrategyInputs, dict]:
        act = self.risk.active()
        snap = self.data.latest_snapshot()
        if act is None or snap is None:
            raise RuntimeError("need a data snapshot and an active risk model")
        model = act[1]
        sec = snap.securities().set_index("symbol")
        fund = snap.fundamentals().set_index("symbol") if not snap.fundamentals().empty else pd.DataFrame(index=sec.index)
        is_etp = sec["subtype1"].fillna("").str.lower().str.startswith("exchange traded") if "subtype1" in sec else pd.Series(False, index=sec.index)
        if universe:
            syms = [s.upper() for s in universe if s.upper() in model.symbols]
        else:
            syms = [s for s in model.symbols if s in sec.index and not bool(is_etp.get(s, False))]
        if strategy.kind == "tax_aware_transition" and strategy.target_weights is None:
            raise ValueError("tax_aware_transition needs target_weights (or a target basket)")
        close = snap.close_matrix("close")
        R = close.pct_change().iloc[-504:]
        if cov_source == "sample":
            cov = shrunk_sample_cov(R[[s for s in syms if s in R.columns]])
        else:
            cov = model.covariance(sorted(set(syms) | set(model.symbols)))
        bench_used = benchmark_name
        if strategy.kind == "levered_beta":
            # target is `beta x the index`: use the index watchlist even if the house benchmark is currently a saved basket
            name = benchmark_name or self.risk.benchmark_name()
            if not benchmark_name or name.lower().startswith("basket:"):
                bench_used = self.ctx.settings.default_benchmark
        bench = self.risk.benchmark_weights(snap, model, bench_used)
        styles = [c for c in model.factors if c in model.spec.styles]
        px = snap.last_prices()
        shares = pd.to_numeric(sec.get("shares_outstanding"), errors="coerce")
        mktcap = (shares.reindex(syms) * px.reindex(syms)).fillna(0.0)
        if "mktcap" in fund:
            rep = pd.to_numeric(fund["mktcap"], errors="coerce").reindex(syms) * 1e6
            mktcap = mktcap.where(mktcap > 0, rep).fillna(0.0)
        cur_w = gain_frac = None
        eid = entity_id or self.ctx.current_entity_id
        if eid is not None:
            from .portfolio_service import PortfolioService
            lots = PortfolioService(self.ctx).lots_view(eid, snap=snap)
            if not lots.empty:
                g = lots.groupby("symbol").agg(mv=("market_value", "sum"), ug=("unrealized", "sum"))
                g = g[g["mv"] > 0]
                cur_w = g["mv"] / g["mv"].sum()
                gain_frac = g["ug"] / g["mv"]
        macro = snap.macro()
        rf = float(macro["rate_3m"].dropna().iloc[-1]) / 100 if (not macro.empty and "rate_3m" in macro and macro["rate_3m"].notna().any()) else 0.0
        inp = StrategyInputs(symbols=syms, cov=cov, benchmark=bench, returns=R, signals=model.exposures[styles],
                             exposures=model.exposures, sectors=sec["gics_sector"] if "gics_sector" in sec else None,
                             mktcap=mktcap, current_weights=cur_w, gain_frac=gain_frac, rf=rf)
        meta = {"snapshot_id": snap.id, "model_version_id": act[0], "benchmark": bench_used or self.risk.benchmark_name(),
                "cov_source": cov_source, "n_universe": len(syms)}
        return inp, meta

    def resolve_target(self, strategy: StrategySpec, target_basket: str | None) -> StrategySpec:
        if target_basket:
            b = self.ctx.baskets.get(target_basket.replace("basket:", ""))
            if not b:
                raise KeyError(f"basket '{target_basket}' not found")
            strategy.target_weights = b["weights"].to_dict()
        return strategy

    # ------------------------------------------------------------------ build
    def build(self, name: str | None, strategy: StrategySpec, benchmark_name: str | None = None, universe: list[str] | None = None,
              entity_id: int | None = None, description: str | None = None, save: bool = True, cov_source: str = "model",
              source: str = "strategy") -> dict:
        inp, meta = self.inputs(strategy, universe, benchmark_name, entity_id, cov_source)
        res = run_strategy(strategy, inp)
        model = self.risk.active()[1]
        ana = analyze_basket(model, res.weights, inp.benchmark) if set(res.weights.index) & set(model.symbols) else None
        out = {"name": name, "strategy": strategy.kind, "status": res.status, "n_names": res.n_names, **meta,
               "diagnostics": res.diagnostics, "top_weights": res.weights.head(15).round(4).to_dict()}
        if ana is not None:
            out["tracking_error_model"] = ana.tracking_error
            out["active_style_exposures"] = {k: round(float(v), 3) for k, v in ana.exposures["active"].items() if k != "market"}
            out["sector_active_pp"] = {k: round(float(v) * 100, 2) for k, v in ana.sectors["active"].items()} if len(ana.sectors) else {}
        if save and name:
            metrics = {**res.diagnostics, "tracking_error_model": ana.tracking_error if ana else None}
            self.ctx.baskets.save(name, res.weights, description or f"{strategy.kind}: {STRATEGIES[strategy.kind]}",
                                  source=f"{source}:{strategy.kind}", benchmark_name=meta["benchmark"],
                                  params={**asdict(strategy), **meta}, metrics=metrics, resolve=self.ctx.resolve_assetid)
            out["saved_basket"] = name
        return out

    # ------------------------------------------------------------------ backtest
    def backtest(self, strategy: StrategySpec, bspec: BacktestSpec, name: str | None = None, entity_id: int | None = None,
                 progress=None) -> tuple[int, BacktestResult]:
        snap = self.data.latest_snapshot()
        if snap is None:
            raise RuntimeError("no data snapshot")
        prices = snap.close_matrix("close")
        sec = snap.securities().set_index("symbol")
        fund = snap.fundamentals().set_index("symbol") if not snap.fundamentals().empty else pd.DataFrame(index=sec.index)
        membership = snap.membership() if bspec.use_membership else None
        macro = snap.macro()
        rf_daily = None
        if not macro.empty and "rate_3m" in macro:
            rf_daily = (pd.to_numeric(macro["rate_3m"], errors="coerce") / 100 / 252).reindex(prices.index).ffill().fillna(0.0)
        res = run_backtest(prices, sec, fund, strategy, bspec, membership=membership, rf_daily=rf_daily, progress=progress)
        rid = self.ctx.runs.create("backtest", date.today(), entity_id or self.ctx.current_entity_id, snap.id, None,
                                   {"strategy": asdict(strategy), "backtest": asdict(bspec), "name": name},
                                   {**res.metrics, "warnings": res.warnings, "name": name or f"{strategy.kind} backtest"},
                                   notes=name)
        folder = self.ctx.settings.runs_dir / f"run_{rid:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"equity": res.equity, "benchmark": res.bench_equity}).to_parquet(folder / "equity.parquet")
        res.weights.to_parquet(folder / "weights_history.parquet")
        res.turnover.rename("turnover").to_frame().to_parquet(folder / "turnover.parquet")
        (folder / "summary.json").write_text(json.dumps(res.to_dict(), indent=2, default=str), encoding="utf-8")
        self.ctx.db.update("runs", "id = ?", (rid,), artifact_path=str(folder))
        return rid, res

    def load_backtest(self, run_id: int) -> dict | None:
        row = self.ctx.runs.get(run_id)
        if not row or row["run_type"] != "backtest":
            return None
        folder = Path(row["artifact_path"]) if row.get("artifact_path") else None
        out = dict(row)

        def rd(n):
            p = folder / n if folder else None
            return pd.read_parquet(p) if p and p.exists() else pd.DataFrame()

        out["equity"] = rd("equity.parquet")
        out["weights"] = rd("weights_history.parquet")
        out["turnover"] = rd("turnover.parquet")
        return out

    def list_backtests(self, limit: int = 50) -> pd.DataFrame:
        df = self.ctx.runs.list(limit=500)
        if df.empty:
            return df
        df = df[df["run_type"] == "backtest"].head(limit)
        rows = []
        for _, r in df.iterrows():
            s, p = r["summary"], r["params"]
            rows.append({"id": r["id"], "created_at": r["created_at"], "name": s.get("name"), "strategy": p.get("strategy", {}).get("kind"),
                         "cagr": s.get("cagr"), "bench_cagr": s.get("bench_cagr"), "sharpe": s.get("sharpe"), "max_drawdown": s.get("max_drawdown"),
                         "tracking_error": s.get("tracking_error"), "information_ratio": s.get("information_ratio"),
                         "annual_turnover": s.get("annual_turnover"), "n_rebalances": s.get("n_rebalances")})
        return pd.DataFrame(rows)


def spec_from_params(kind: str, params: dict | None) -> StrategySpec:
    """Build a StrategySpec from loosely-typed params (from the GUI or the co-pilot)."""
    p = dict(params or {})
    p["kind"] = kind
    valid = {k: v for k, v in p.items() if k in StrategySpec.__dataclass_fields__}
    if "target_weights" in valid and valid["target_weights"] is not None:
        valid["target_weights"] = {str(k).upper(): float(v) for k, v in dict(valid["target_weights"]).items()}
    if "signal_weights" in valid and valid["signal_weights"] is not None:
        valid["signal_weights"] = {str(k): float(v) for k, v in dict(valid["signal_weights"]).items()}
    if "tilts" in valid and valid["tilts"] is not None:
        valid["tilts"] = {str(k): float(v) for k, v in dict(valid["tilts"]).items()}
    if "exclude" in valid and valid["exclude"]:
        valid["exclude"] = [str(s).upper() for s in valid["exclude"]]
    if valid.get("n_max") in (0, "", "none", "None"):
        valid["n_max"] = None
    return StrategySpec(**valid)


def _nan_safe(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}
