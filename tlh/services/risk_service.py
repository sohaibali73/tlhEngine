"""Risk-model lifecycle: fit from a snapshot, persist as a versioned artifact, load the active model, compare,
calibrate (which window / weighting / estimator forecasts best) and run the model library."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..data.cache import Snapshot
from ..risk import benchmark as bm
from ..risk.model import (
    RISK_MODEL_PRESETS,
    FactorRiskModel,
    FittedRiskModel,
    RiskModelSpec,
    preset_spec,
    preset_table,
)
from .context import AppContext

log = logging.getLogger(__name__)


class RiskService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    # ------------------------------------------------------------------ fitting
    def fit(self, snap: Snapshot, spec: RiskModelSpec | None = None, name: str | None = None, notes: str | None = None,
            make_active: bool = True, progress=None) -> tuple[int, FittedRiskModel]:
        spec = spec or self.default_spec()
        from ..risk.custom import load_all
        load_all()                                  # register any AI-authored style factors before fitting
        prices = snap.close_matrix("close")
        sec = snap.securities().set_index("symbol")
        fund = snap.fundamentals().set_index("symbol") if not snap.fundamentals().empty else pd.DataFrame(index=sec.index)
        macro = snap.macro() if spec.use_macro else None
        volume = snap.close_matrix("volume") if spec.model_kind in ("erm", "hybrid") else None
        model = FactorRiskModel(spec).fit(prices, sec, fund, macro_levels=macro, universe_name=snap.universe_name,
                                          snapshot_id=snap.id, progress=progress, volume=volume)
        label = spec.preset or spec.name or spec.model_kind
        name = name or f"{label} {model.as_of:%Y-%m-%d}"
        code_ver = self.ctx.code.latest_version("tlh/risk/model.py")
        # Allocate the version row first so the artifact can be written straight into its final folder
        # (renaming a just-written folder on Windows can fail with "Access is denied" while handles settle).
        mid = self.ctx.models.create(name=name, snapshot_id=snap.id, as_of=model.as_of, universe_name=snap.universe_name,
                                     lookback_days=spec.lookback_days if spec.is_fundamental else spec.stat_lookback, factor_list=model.factors,
                                     diagnostics=model.diagnostics, artifact_path="",
                                     code_version_id=code_ver["id"] if code_ver else None, notes=notes,
                                     make_active=make_active)
        final = self.ctx.settings.models_dir / f"model_{mid:04d}"
        model.save(final)
        self.ctx.db.update("model_versions", "id = ?", (mid,), artifact_path=str(final))
        return mid, model

    def fit_preset(self, snap: Snapshot, preset: str, make_active: bool = True, progress=None, **overrides) -> tuple[int, FittedRiskModel]:
        return self.fit(snap, preset_spec(preset, **overrides), make_active=make_active, progress=progress)

    def fit_library(self, snap: Snapshot, presets: list[str] | None = None, progress=None) -> pd.DataFrame:
        """Fit several presets back to back (the active model is left unchanged) and return a comparison table."""
        say = progress or (lambda m: None)
        rows = []
        for p in presets or list(RISK_MODEL_PRESETS):
            try:
                say(f"Fitting {p}…")
                mid, m = self.fit(snap, preset_spec(p), make_active=False, progress=say)
                d = m.diagnostics
                rows.append({"preset": p, "model_id": mid, "kind": m.kind, "factors": len(m.factors), "symbols": len(m.symbols),
                             "avg_r2": d.get("avg_r2"), "median_specific_vol": d.get("median_specific_vol"),
                             "market_vol": float(m.factor_vols().get("market", float("nan"))) if "market" in m.factors else None, "status": "ok"})
            except Exception as e:  # keep going through the library
                log.warning("preset %s failed: %s", p, e)
                rows.append({"preset": p, "model_id": None, "status": f"failed: {str(e)[:120]}"})
        return pd.DataFrame(rows)

    @staticmethod
    def presets() -> pd.DataFrame:
        return preset_table()

    def default_spec(self) -> RiskModelSpec:
        saved = self.ctx.get("risk_spec")
        if saved:
            try:
                return RiskModelSpec(**{k: v for k, v in saved.items() if k in RiskModelSpec.__dataclass_fields__})
            except TypeError:
                pass
        return RiskModelSpec()

    def save_spec(self, spec: RiskModelSpec) -> None:
        self.ctx.set("risk_spec", asdict(spec))

    # ------------------------------------------------------------------ loading
    def load(self, model_id: int) -> FittedRiskModel | None:
        row = self.ctx.models.get(model_id)
        if not row or not Path(row["artifact_path"]).exists():
            return None
        return FittedRiskModel.load(Path(row["artifact_path"]))

    def active(self) -> tuple[int, FittedRiskModel] | None:
        row = self.ctx.models.active()
        if not row:
            return None
        cached = getattr(self, "_active_cache", None)
        if cached and cached[0] == row["id"] and cached[2] == row.get("artifact_path"):
            return cached[0], cached[1]
        m = self.load(row["id"])
        if m is None:
            return None
        self._active_cache = (row["id"], m, row.get("artifact_path"))
        return row["id"], m

    # ------------------------------------------------------------------ benchmark
    def benchmark_name(self) -> str:
        return self.ctx.get("benchmark_name", self.ctx.settings.default_benchmark)

    def benchmark_weights(self, snap: Snapshot, model: FittedRiskModel, name: str | None = None) -> pd.Series:
        """Benchmark by name: 'basket:<name>' (saved model portfolio), an ETF ticker, or a Norgate watchlist."""
        name = name or self.benchmark_name()
        if name.lower().startswith("basket:"):
            b = self.ctx.baskets.get(name.split(":", 1)[1].strip())
            if not b:
                raise KeyError(f"basket '{name}' not found")
            w = b["weights"]
            w = w[w.index.isin(model.symbols)]
            if w.empty:
                raise ValueError(f"basket {name} has no symbols in the risk model")
            return w / w.sum()
        if name.upper() in model.symbols and name.isupper() and len(name) <= 5:
            return bm.single_etf(name.upper())
        try:
            members = self.ctx.norgate.watchlist_symbols(name)
        except Exception:
            members = [s for s in model.symbols if not s.isupper() or len(s) > 5]
        members = [s for s in members if s in model.symbols]
        w = bm.cap_weighted(snap.securities(), snap.last_prices(), members)
        return w

    # ------------------------------------------------------------------ comparisons
    def exposure_table(self, model: FittedRiskModel, weights: pd.Series, bench: pd.Series) -> pd.DataFrame:
        idx = weights.index.union(bench.index)
        idx = [s for s in idx if s in model.symbols]
        w = weights.reindex(idx).fillna(0.0)
        b = bench.reindex(idx).fillna(0.0)
        w = w / w.sum() if w.sum() else w
        b = b / b.sum() if b.sum() else b
        pe = model.portfolio_exposures(w)
        be = model.portfolio_exposures(b)
        df = pd.DataFrame({"portfolio": pe, "benchmark": be})
        df["active"] = df["portfolio"] - df["benchmark"]
        df["factor_vol"] = model.factor_vols()
        df["kind"] = ["market" if f == "market" else "sector" if str(f).startswith(("sec:", "ind:")) else "macro" if str(f).startswith("macro:")
                      else "statistical" if str(f).startswith("stat:") else "style" for f in df.index]
        return df

    # ------------------------------------------------------------------ analytics (Risk lab)
    def holdings_weights(self, entity_id: int, snap: Snapshot, model: FittedRiskModel) -> pd.Series | None:
        from .portfolio_service import PortfolioService
        lots = PortfolioService(self.ctx).lots_view(entity_id, snap=snap)
        if lots.empty:
            return None
        w = lots.groupby("symbol")["market_value"].sum()
        w = w[w.index.isin(model.symbols)]
        return (w / w.sum()) if w.sum() > 0 else None

    def _ctx_for_analytics(self, entity_id: int | None, weights: pd.Series | None = None):
        act = self.active()
        snap = self.data_snapshot()
        if act is None or snap is None:
            raise RuntimeError("need an active risk model and a data snapshot")
        model = act[1]
        if weights is None:
            eid = entity_id or self.ctx.current_entity_id
            weights = self.holdings_weights(eid, snap, model) if eid is not None else None
        if weights is None or weights.empty:
            raise RuntimeError("no holdings in the risk-model universe")
        bench = self.benchmark_weights(snap, model)
        return model, snap, weights, bench

    def data_snapshot(self) -> Snapshot | None:
        return self.ctx.store.latest()

    def decomposition(self, entity_id: int | None = None, active: bool = True, weights: pd.Series | None = None) -> dict:
        from ..risk.analytics import risk_decomposition
        model, _, w, bench = self._ctx_for_analytics(entity_id, weights)
        return risk_decomposition(model, w, bench if active else None)

    def stress(self, shocks: dict[str, float], entity_id: int | None = None, active: bool = False, propagate: bool = True,
               weights: pd.Series | None = None) -> dict:
        from ..risk.analytics import stress_test
        model, _, w, bench = self._ctx_for_analytics(entity_id, weights)
        return stress_test(model, w, shocks, bench if active else None, propagate=propagate)

    def scenario(self, start, end, entity_id: int | None = None, active: bool = False) -> dict:
        from ..risk.analytics import historical_scenario
        model, _, w, bench = self._ctx_for_analytics(entity_id)
        return historical_scenario(model, w, start, end, bench if active else None)

    def var(self, entity_id: int | None = None, horizon_days: int = 21, alpha: float = 0.99, active: bool = False) -> dict:
        from ..risk.analytics import parametric_var
        model, _, w, bench = self._ctx_for_analytics(entity_id)
        return parametric_var(model, w, horizon_days, alpha, bench if active else None)

    def bias_test(self, spec: RiskModelSpec | None = None, n_periods: int = 6, period_days: int = 21, entity_id: int | None = None,
                  progress=None) -> pd.DataFrame:
        from ..risk.analytics import bias_test
        snap = self.data_snapshot()
        if snap is None:
            raise RuntimeError("no snapshot")
        spec = spec or (self.active()[1].spec if self.active() else self.default_spec())
        prices = snap.close_matrix("close")
        sec = snap.securities().set_index("symbol")
        fund = snap.fundamentals().set_index("symbol") if not snap.fundamentals().empty else pd.DataFrame(index=sec.index)
        volume = snap.close_matrix("volume") if spec.model_kind in ("erm", "hybrid") else None
        holdings = None
        eid = entity_id or self.ctx.current_entity_id
        act = self.active()
        if eid is not None and act is not None:
            holdings = self.holdings_weights(eid, snap, act[1])
        return bias_test(prices, sec, fund, spec, n_periods=n_periods, period_days=period_days, volume=volume,
                         macro=snap.macro() if spec.use_macro else None, holdings=holdings, progress=progress)

    def compare_versions(self, a_id: int, b_id: int) -> pd.DataFrame:
        a, b = self.load(a_id), self.load(b_id)
        if a is None or b is None:
            return pd.DataFrame()
        va, vb = a.factor_vols(), b.factor_vols()
        df = pd.DataFrame({f"vol_v{a_id}": va, f"vol_v{b_id}": vb})
        df["change"] = df.iloc[:, 1] - df.iloc[:, 0]
        return df

    # ------------------------------------------------------------------ calibration study
    def calibrate(self, quick: bool = True, include_pca: bool = False, include_holdings: bool = True, entity_id: int | None = None,
                  lookbacks: tuple[int, ...] | None = None, horizons: tuple[int, ...] | None = None, progress=None) -> dict:
        """Walk-forward calibration of lookback x weighting x estimator x horizon on the snapshot universe."""
        from ..risk.calibration import CalibrationGrid, run_calibration
        snap = self.data_snapshot()
        if snap is None:
            raise RuntimeError("no snapshot")
        grid = CalibrationGrid.quick() if quick else CalibrationGrid()
        if include_pca:
            grid.estimators = tuple(list(grid.estimators) + ["pca"])
        if lookbacks:
            grid.lookbacks = tuple(lookbacks)
        if horizons:
            grid.horizons = tuple(horizons)
        eid = entity_id or self.ctx.current_entity_id
        if include_holdings and eid is not None:
            from .portfolio_service import PortfolioService
            lots = PortfolioService(self.ctx).lots_view(eid, snap=snap)
            if not lots.empty:
                grid.holdings = lots.groupby("symbol")["market_value"].sum()
        prices = snap.close_matrix("close")
        out = run_calibration(prices, grid, progress=progress)
        out["snapshot_id"] = snap.id
        self.ctx.set("last_calibration", {"recommendation": out["recommendation"], "snapshot_id": snap.id,
                                          "winners": out["winners"][["Horizon", "Lookback", "Weighting", "Estimator", "Score", "TEBiasRatio", "TESpearman"]].to_dict("records")})
        self.ctx.db.audit("user", "risk.calibrate", snap.id, quick=quick, scenarios=int(len(out["scoreboard"])))
        return out

    def pair_study(self, pairs: list[tuple[str, str]] | None = None, horizon: int = 63, progress=None) -> pd.DataFrame:
        from ..risk.calibration import pair_study
        snap = self.data_snapshot()
        if snap is None:
            raise RuntimeError("no snapshot")
        prices = snap.close_matrix("close")
        if not pairs:
            pairs = self.default_pairs(list(prices.columns))
        uni = [c for c in prices.columns if prices[c].notna().mean() > 0.95][:300]
        return pair_study(prices, pairs, horizon=horizon, universe_for_shrinkage=uni, progress=progress)

    def default_pairs(self, available: list[str]) -> list[tuple[str, str]]:
        """Tight substitute pairs from the substitutes map (identical / presumed-identical tiers) present in the snapshot."""
        pairs: list[tuple[str, str]] = []
        try:
            sm = self.ctx.substitutes
            groups = [sorted(v) for v in list(sm.identical.values()) + list(sm.presumed.values()) if len(v) >= 2]
        except Exception:
            groups = []
        avail = set(available)
        for g in groups:
            syms = [s for s in g if s in avail]
            for i in range(len(syms) - 1):
                pairs.append((syms[i], syms[i + 1]))
        if not pairs:
            for a, b in (("IVV", "SPY"), ("VOO", "SPY"), ("SCHX", "SPY"), ("QQQM", "QQQ"), ("VTI", "ITOT"), ("IWM", "VTWO")):
                if a in avail and b in avail:
                    pairs.append((a, b))
        return pairs[:12]

    @staticmethod
    def spec_from_recommendation(rec: dict, **overrides) -> RiskModelSpec:
        return RiskModelSpec(name=f"calibrated_{rec['lookback']}_{rec['weighting'][:3]}_{rec['estimator'][:2]}", model_kind="statistical",
                             stat_lookback=int(rec["lookback"]), stat_weighting=rec["weighting"], stat_estimator=rec["estimator"],
                             preset="Calibration study recommendation", **overrides)
