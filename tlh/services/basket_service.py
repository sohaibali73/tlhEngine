"""Model portfolios (baskets): create from weights, construct by optimisation, analyse vs benchmark."""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date

import pandas as pd

from ..optim.basket import BasketSpec, analyze_basket, build_basket
from ..tax.washsale import screen_proposed_buy
from .context import AppContext
from .data_service import DataService
from .risk_service import RiskService

log = logging.getLogger(__name__)


class BasketService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.data = DataService(ctx)
        self.risk = RiskService(ctx)

    def _model_snap(self):
        act = self.risk.active()
        snap = self.data.latest_snapshot()
        if act is None or snap is None:
            raise RuntimeError("need a data snapshot and an active risk model")
        return act[1], snap

    def _bench(self, snap, model, benchmark_name: str | None) -> tuple[pd.Series, str]:
        name = benchmark_name or self.risk.benchmark_name()
        return self.risk.benchmark_weights(snap, model, name), name

    # ------------------------------------------------------------------ CRUD
    def list(self) -> pd.DataFrame:
        return self.ctx.baskets.list()

    def get(self, name: str) -> dict | None:
        return self.ctx.baskets.get(name)

    def delete(self, name: str) -> None:
        self.ctx.baskets.delete(name)

    def create(self, name: str, weights: pd.Series, description: str | None = None, source: str = "manual",
               benchmark_name: str | None = None) -> dict:
        model, snap = self._model_snap()
        bench, bname = self._bench(snap, model, benchmark_name)
        unknown = [s for s in weights.index if s not in model.symbols]
        res = analyze_basket(model, weights.drop(index=unknown), bench)
        metrics = res.metrics()
        metrics["unknown_symbols_dropped"] = unknown
        self.ctx.baskets.save(name, res.weights, description, source=source, benchmark_name=bname,
                              params={"method": "explicit"}, metrics=metrics, resolve=self.ctx.resolve_assetid)
        return self._summary(name, res, bname, unknown)

    def optimize(self, name: str, spec: BasketSpec, benchmark_name: str | None = None, description: str | None = None,
                 source: str = "optimizer", exclude_held: bool = False, exclude_wash_blocked: bool = True,
                 entity_id: int | None = None) -> dict:
        model, snap = self._model_snap()
        bench, bname = self._bench(snap, model, benchmark_name)
        sec = snap.securities().set_index("symbol")
        exclude = set(spec.exclude)
        eid = entity_id or self.ctx.current_entity_id
        if eid is not None and (exclude_held or exclude_wash_blocked):
            from .portfolio_service import PortfolioService
            book = PortfolioService(self.ctx).book(eid)
            if exclude_held:
                exclude |= {lot.symbol for lot in book.open_lots()}
            if exclude_wash_blocked:
                sales = book.loss_sales(since=date.today().replace(day=1) if False else None)
                recent = [s for s in sales if (date.today() - s.sale_date).days <= 30]
                if recent:
                    for sym in model.symbols:
                        aid = self.ctx.securities.resolve(sym)
                        if aid is not None and screen_proposed_buy(aid, sym, date.today(), recent, book.groups).status != "SAFE":
                            exclude.add(sym)
        spec.exclude = sorted(exclude)
        res = build_basket(model, bench, None, spec, securities=sec)
        params = asdict(spec) | {"exclude_held": exclude_held, "exclude_wash_blocked": exclude_wash_blocked}
        self.ctx.baskets.save(name, res.weights, description, source=source, benchmark_name=bname, params=params,
                              metrics=res.metrics(), resolve=self.ctx.resolve_assetid)
        return self._summary(name, res, bname)

    def analyze(self, name: str, benchmark_name: str | None = None) -> dict:
        b = self.ctx.baskets.get(name)
        if not b:
            raise KeyError(f"basket '{name}' not found")
        model, snap = self._model_snap()
        bench, bname = self._bench(snap, model, benchmark_name or b.get("benchmark_name"))
        res = analyze_basket(model, b["weights"], bench)
        return self._summary(name, res, bname)

    def result(self, name: str, benchmark_name: str | None = None):
        """Full BasketResult for the GUI charts."""
        b = self.ctx.baskets.get(name)
        if not b:
            return None
        model, snap = self._model_snap()
        bench, _ = self._bench(snap, model, benchmark_name or b.get("benchmark_name"))
        return analyze_basket(model, b["weights"], bench)

    # ------------------------------------------------------------------ sample library
    def build_library(self, names: list[str] | None = None, audience: str | None = None, benchmark_name: str | None = None,
                      prefix: str = "Sample · ", progress=None) -> pd.DataFrame:
        """Build the sample model-portfolio library (optim/basket_library.py) against the live snapshot and active model."""
        from ..optim.basket_library import recipes
        from .strategy_service import StrategyService, spec_from_params
        say = progress or (lambda m: None)
        svc = StrategyService(self.ctx)
        rows = []
        todo = [r for r in recipes(audience) if not names or r.name in set(names)]
        for i, r in enumerate(todo, 1):
            say(f"Building {r.name} ({i}/{len(todo)})…")
            try:
                spec = spec_from_params(r.kind, r.params)
                out = svc.build(prefix + r.name, spec, benchmark_name or r.benchmark, description=r.pitch, cov_source=r.cov_source, source="library")
                d = out.get("diagnostics", {})
                rows.append({"name": prefix + r.name, "strategy": r.kind, "audience": r.audience, "status": out.get("status"),
                             "n_names": out.get("n_names"), "tracking_error": out.get("tracking_error_model") or d.get("tracking_error"),
                             "volatility": d.get("volatility"), "gross": d.get("gross"), "beta": d.get("beta"), "pitch": r.pitch})
            except Exception as e:  # keep going through the library
                log.warning("library basket %s failed: %s", r.name, e)
                rows.append({"name": prefix + r.name, "strategy": r.kind, "audience": r.audience, "status": f"failed: {str(e)[:140]}", "pitch": r.pitch})
        return pd.DataFrame(rows)

    def _summary(self, name: str, res, bname: str, unknown: list[str] | None = None) -> dict:
        exp = res.exposures
        return {
            "name": name, "benchmark": bname, "n_names": res.n_names, "tracking_error": res.tracking_error, "status": res.status,
            "top_weights": res.weights.head(15).round(4).to_dict(), "weights_sum": float(res.weights.sum()),
            "active_style_exposures": {k: round(float(v), 3) for k, v in exp["active"].items() if k != "market"},
            "market_beta": float(exp.loc["market", "basket"]) if "market" in exp.index else None,
            "sector_active_pp": {k: round(float(v) * 100, 2) for k, v in res.sectors["active"].items()} if len(res.sectors) else {},
            "unknown_symbols_dropped": unknown or [],
        }
