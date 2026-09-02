"""Execute TLH model pipelines built in the drag-and-drop builder (or authored by YANG as JSON)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..optim.pipeline import Pipeline, apply_filter, ordered, rank_keep, spec_from_nodes, validate
from ..optim.strategies import run_strategy
from ..tax.washsale import screen_proposed_buy
from .context import AppContext
from .data_service import DataService
from .harvest_service import HarvestService
from .risk_service import RiskService
from .strategy_service import StrategyService

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    ok: bool
    log: list[str] = field(default_factory=list)
    universe_size: int = 0
    weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    diagnostics: dict = field(default_factory=dict)
    benchmark: str | None = None
    basket_name: str | None = None
    harvest_run_id: int | None = None
    harvest_summary: dict = field(default_factory=dict)
    export_path: str | None = None
    duration_s: float = 0.0

    def summary(self) -> dict:
        return {"ok": self.ok, "universe_size": self.universe_size, "n_names": int((self.weights > 0).sum()), "benchmark": self.benchmark,
                "basket_name": self.basket_name, "harvest_run_id": self.harvest_run_id, "harvest": self.harvest_summary,
                "diagnostics": self.diagnostics, "top_weights": self.weights.head(15).round(4).to_dict(), "export_path": self.export_path,
                "log": self.log, "duration_s": round(self.duration_s, 1)}


class PipelineService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.data = DataService(ctx)
        self.risk = RiskService(ctx)
        self.strat = StrategyService(ctx)
        self.harvest = HarvestService(ctx)

    # ------------------------------------------------------------------ persistence
    def list(self) -> pd.DataFrame:
        return self.ctx.pipelines.list()

    def save(self, p: Pipeline, source: str = "builder") -> int:
        return self.ctx.pipelines.save(p.name, p.to_json(), p.description, source)

    def load(self, name: str) -> Pipeline | None:
        row = self.ctx.pipelines.get(name)
        return Pipeline.from_json(row["spec"]) if row else None

    def delete(self, name: str) -> None:
        self.ctx.pipelines.delete(name)

    # ------------------------------------------------------------------ execution
    def run(self, p: Pipeline, entity_id: int | None = None, progress=None) -> PipelineResult:
        t0 = time.time()
        say = progress or (lambda m: None)
        res = PipelineResult(ok=False)

        def log_(m: str) -> None:
            res.log.append(m)
            say(m)

        errs = validate(p)
        if errs:
            res.log.extend(f"invalid: {e}" for e in errs)
            return res
        act = self.risk.active()
        snap = self.data.latest_snapshot()
        if act is None or snap is None:
            res.log.append("need a data snapshot and an active risk model")
            return res
        model = act[1]
        eid = entity_id or self.ctx.current_entity_id
        sec = snap.securities().set_index("symbol")
        fund = snap.fundamentals().set_index("symbol") if not snap.fundamentals().empty else pd.DataFrame(index=sec.index)
        nodes = ordered(p)
        by_type = {n.type: n for n in nodes}
        bench_name = by_type["benchmark"].params.get("name") if "benchmark" in by_type else self.risk.benchmark_name()
        res.benchmark = bench_name

        # ---- universe
        u = by_type["universe"].params
        syms = self._universe(u, model, sec, eid, log_)
        log_(f"Universe: {len(syms)} names ({u.get('source')})")

        # ---- filters / ranks in order
        stats = None
        for n in nodes:
            if n.type == "filter":
                if stats is None:
                    stats = self._stats(snap, sec, fund, syms)
                before = len(syms)
                syms = apply_filter(syms, sec, stats["mktcap"], stats["ret"], stats["vol"], n.params)
                log_(f"Filter: {before} -> {len(syms)}")
            elif n.type == "rank":
                styles = [c for c in model.factors if c in model.spec.styles]
                sig = model.exposures[styles].reindex(syms)
                before = len(syms)
                syms, score = rank_keep(syms, sig, n.params)
                log_(f"Rank ({n.params.get('signal_weights')}): kept top {len(syms)} of {before}")
        res.universe_size = len(syms)
        if len(syms) < 2:
            res.log.append("universe too small after screens")
            return res

        # ---- construction
        if "construct" not in by_type:
            res.weights = pd.Series(1.0 / len(syms), index=syms)
            res.diagnostics = {"note": "no Construction block: equal-weight screened universe"}
            log_("No Construction block: returning equal-weighted screened universe")
        else:
            c = by_type["construct"].params
            spec = spec_from_nodes(c, by_type["transition"].params if "transition" in by_type else None)
            inp, meta = self.strat.inputs(spec, universe=syms, benchmark_name=bench_name, entity_id=eid, cov_source=c.get("cov_source", "model"))
            inp.symbols = [s for s in syms if s in inp.cov.index]
            r = run_strategy(spec, inp)
            res.weights, res.diagnostics = r.weights, dict(r.diagnostics)
            log_(f"Construction ({spec.kind}): {r.n_names} names, TE {r.diagnostics.get('tracking_error', float('nan')):.2%}, status {r.status}")
            # ---- transition
            if "transition" in by_type:
                if inp.current_weights is None:
                    log_("Transition skipped: no current holdings")
                else:
                    spec_t = spec_from_nodes(c, by_type["transition"].params)
                    spec_t.kind = "tax_aware_transition"
                    spec_t.target_weights = r.weights.to_dict()
                    tr = run_strategy(spec_t, inp)
                    res.weights = tr.weights
                    res.diagnostics = {"construction": res.diagnostics, "transition": tr.diagnostics}
                    d = tr.diagnostics
                    log_(f"Transition: TE-to-target {d['te_to_target_before']:.2%} -> {d['te_to_target']:.2%}, turnover {d['turnover']:.0%}, "
                         f"net realised gain {d['realised_gain_frac']:.2%} (budget {d['gain_budget']:.2%})")

        # ---- output: save basket (needed for harvest-toward)
        out = by_type["output"].params if "output" in by_type else {}
        basket_name = out.get("basket_name") or (f"{p.name} result" if "harvest" in by_type else None)
        if basket_name:
            self.ctx.baskets.save(basket_name, res.weights, out.get("description") or p.description or f"pipeline {p.name}", source="pipeline",
                                  benchmark_name=bench_name, params={"pipeline": p.name}, metrics=_jsonable(res.diagnostics), resolve=self.ctx.resolve_assetid)
            res.basket_name = basket_name
            log_(f"Saved basket '{basket_name}'")
            if out.get("set_benchmark"):
                self.ctx.set("benchmark_name", f"basket:{basket_name}")
                log_(f"App benchmark set to basket:{basket_name}")

        # ---- harvest toward result
        if "harvest" in by_type and eid is not None:
            h = by_type["harvest"].params
            from dataclasses import replace
            cfg = replace(self.harvest.load_config(), mode=h.get("mode", "full_rebalance"), te_budget=float(h.get("te_budget", 0.02)),
                          te_hard=bool(h.get("te_hard", False)), min_trade_value=float(h.get("min_trade_value", 500)))
            try:
                rid, hres = self.harvest.run(eid, cfg, notes=f"pipeline {p.name}", benchmark_name=f"basket:{basket_name}")
                res.harvest_run_id, res.harvest_summary = rid, hres.summary
                s = hres.summary
                log_(f"Harvest run #{rid}: {s['n_sells']} sells / {s['n_buys']} buys, harvested loss ${s['harvested_loss']:,.0f}, "
                     f"TE {s['te_before']:.2%} -> {s['te_after']:.2%}")
                if out.get("export_excel"):
                    from ..export.excel import export_run_workbook
                    path = self.ctx.settings.exports_dir / f"pipeline_{_slug(p.name)}_run_{rid:04d}.xlsx"
                    export_run_workbook(path, self.harvest.load_run(rid))
                    res.export_path = str(path)
                    log_(f"Exported {path.name}")
            except Exception as e:
                log_(f"Harvest failed: {e}")
        res.ok = True
        res.duration_s = time.time() - t0
        self.ctx.db.audit("user", "pipeline.run", p.name, ok=res.ok, basket=res.basket_name, run=res.harvest_run_id)
        return res

    # ------------------------------------------------------------------ helpers
    def _universe(self, u: dict, model, sec: pd.DataFrame, eid: int | None, log_) -> list[str]:
        src = u.get("source", "model")
        if src == "watchlist":
            try:
                syms = [s for s in self.ctx.norgate.watchlist_symbols(u.get("name") or "S&P 500") if s in model.symbols]
            except Exception as e:
                log_(f"watchlist unavailable ({e}); falling back to model universe")
                syms = list(model.symbols)
        elif src == "basket":
            b = self.ctx.baskets.get((u.get("name") or "").replace("basket:", ""))
            syms = [s for s in (b["weights"].index if b else []) if s in model.symbols]
        elif src == "custom":
            want = [s.strip().upper() for s in str(u.get("name", "")).replace(";", ",").split(",") if s.strip()]
            syms = [s for s in want if s in model.symbols]
        else:
            syms = list(model.symbols)
        if u.get("exclude_etps", True) and "subtype1" in sec:
            etp = sec["subtype1"].fillna("").str.lower().str.startswith("exchange traded")
            syms = [s for s in syms if not bool(etp.get(s, False))]
        if eid is not None and (u.get("exclude_held") or u.get("exclude_wash_blocked", True)):
            from .portfolio_service import PortfolioService
            book = PortfolioService(self.ctx).book(eid)
            if u.get("exclude_held"):
                held = {lot.symbol for lot in book.open_lots()}
                syms = [s for s in syms if s not in held]
            if u.get("exclude_wash_blocked", True):
                recent = [s for s in book.loss_sales() if (date.today() - s.sale_date).days <= 30]
                if recent:
                    blocked = set()
                    for s in syms:
                        aid = self.ctx.securities.resolve(s)
                        if aid is not None and screen_proposed_buy(aid, s, date.today(), recent, book.groups).status != "SAFE":
                            blocked.add(s)
                    if blocked:
                        log_(f"Excluded wash-blocked buys: {sorted(blocked)}")
                    syms = [s for s in syms if s not in blocked]
        return syms

    def _stats(self, snap, sec: pd.DataFrame, fund: pd.DataFrame, syms: list[str]) -> dict:
        close = snap.close_matrix("close")
        sub = close[[s for s in syms if s in close.columns]].iloc[-253:]
        ret = (sub.iloc[-1] / sub.iloc[0] - 1.0) if len(sub) > 20 else pd.Series(dtype=float)
        vol = sub.pct_change().std() * np.sqrt(252) if len(sub) > 20 else pd.Series(dtype=float)
        px = snap.last_prices()
        shares = pd.to_numeric(sec.get("shares_outstanding"), errors="coerce")
        mc = (shares.reindex(syms) * px.reindex(syms)) / 1e6
        if "mktcap" in fund:
            rep = pd.to_numeric(fund["mktcap"], errors="coerce").reindex(syms)
            mc = mc.where(mc > 0, rep)
        return {"mktcap": mc, "ret": ret, "vol": vol}


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")[:40]


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, float | np.floating):
        return None if not np.isfinite(d) else float(d)
    return d
