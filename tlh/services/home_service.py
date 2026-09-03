"""Start-here service: the one-click path from "I have a portfolio" to "here are my tax savings", plus the KPIs and
plain-English summaries the dashboard shows.

Everything here composes existing services; nothing computes tax or risk on its own.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..explain import explain_harvest, explain_kpis
from ..tax.rates import TaxProfile
from ..tax.state_rates import combined_marginal
from .context import AppContext
from .data_service import DataService
from .harvest_service import HarvestService
from .portfolio_service import PortfolioService
from .risk_service import RiskService

log = logging.getLogger(__name__)


@dataclass
class OneClickResult:
    run_id: int | None
    summary: dict
    sentences: list[str]
    steps: list[str] = field(default_factory=list)
    kpis: dict = field(default_factory=dict)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


class HomeService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.data = DataService(ctx)
        self.portfolio = PortfolioService(ctx)
        self.risk = RiskService(ctx)
        self.harvest = HarvestService(ctx)

    # ------------------------------------------------------------------ tax setup
    def tax_setup(self) -> dict:
        prof = self.ctx.tax.default_profile()
        eid = self.ctx.current_entity_id
        filing = prof.filing_status
        if eid is not None:
            e = next((x for x in self.ctx.entities.list() if x["id"] == eid), None)
            if e:
                filing = e["filing_status"]
        return {"state": self.ctx.get("tax_state", ""), "filing_status": filing, "other_income": float(self.ctx.get("other_income", 250_000.0)),
                "st_rate": prof.st_rate, "lt_rate": prof.lt_rate, "state_rate": prof.state_rate}

    def apply_tax_setup(self, state: str, filing_status: str, other_income: float) -> dict:
        """Derive the default TaxProfile from state + filing status + income and persist it (plus the bracket schedule's state rate)."""
        c = combined_marginal(state, filing_status, other_income) if state else None
        prof_old = self.ctx.tax.default_profile()
        if c:
            prof = TaxProfile(name="default", fed_st_rate=c["fed_st"], fed_lt_rate=c["fed_lt"], state_rate=c["state_lt"],
                              niit_rate=0.038, apply_niit=c["niit"] > 0, ordinary_offset=prof_old.ordinary_offset, filing_status=filing_status)
        else:
            prof = TaxProfile(name="default", fed_st_rate=prof_old.fed_st_rate, fed_lt_rate=prof_old.fed_lt_rate, state_rate=prof_old.state_rate,
                              niit_rate=prof_old.niit_rate, ordinary_offset=prof_old.ordinary_offset, filing_status=filing_status)
        self.ctx.tax.save(prof, make_default=True)
        self.ctx.set("tax_state", state)
        self.ctx.set("other_income", float(other_income))
        eid = self.ctx.current_entity_id
        if eid is not None:
            self.ctx.db.update("entities", "id = ?", (eid,), filing_status=filing_status)
        # keep the concentration workbench's bracket schedule in step
        try:
            from ..tax.concentration import BracketSchedule
            saved = self.ctx.get("bracket_schedule")
            sched = BracketSchedule.from_dict(saved) if saved else BracketSchedule.default(filing_status)
            sched.filing_status = filing_status
            sched.state_rate = prof.state_rate
            self.ctx.set("bracket_schedule", sched.to_dict())
        except Exception as e:  # pragma: no cover - defensive
            log.debug("bracket schedule sync skipped: %s", e)
        self.ctx.db.audit("user", "tax.setup", state, filing=filing_status, income=other_income)
        out = {"profile": prof, "st_rate": prof.st_rate, "lt_rate": prof.lt_rate}
        if c:
            out["combined"] = c
        return out

    # ------------------------------------------------------------------ KPIs
    def kpis(self, entity_id: int | None = None) -> dict:
        eid = entity_id or self.ctx.current_entity_id
        out: dict = {"entity_id": eid, "has_holdings": False, "snapshot": None, "model": None}
        snap = self.data.latest_snapshot()
        out["snapshot"] = snap.id if snap else None
        out["snapshot_as_of"] = str(snap.as_of_date) if snap else None
        act = self.ctx.models.active()
        out["model"] = f"#{act['id']}" if act else None
        out["norgate"] = self.data.norgate_ok()
        if eid is None:
            return out
        lots = self.portfolio.lots_view(eid, snap=snap)
        s = self.portfolio.summary(lots)
        out.update(s)
        out["has_holdings"] = bool(len(lots))
        prof = self.ctx.tax.default_profile()
        out["st_rate"], out["lt_rate"] = prof.st_rate, prof.lt_rate
        yr = date.today().year
        try:
            rs = self.portfolio.realized(eid, yr)
            harvested = float(-min(rs.get("net_st", 0.0) + rs.get("net_lt", 0.0), 0.0))
            out["ytd_harvested"] = harvested
            out["ytd_tax_value"] = float(-min(rs.get("net_st", 0.0), 0.0)) * prof.st_rate + float(-min(rs.get("net_lt", 0.0), 0.0)) * prof.lt_rate
            out["ytd_net_st"], out["ytd_net_lt"] = rs.get("net_st", 0.0), rs.get("net_lt", 0.0)
        except Exception as e:
            log.debug("realized summary unavailable: %s", e)
        out["potential_tax_value"] = s.get("harvestable_st", 0.0) * prof.st_rate + s.get("harvestable_lt", 0.0) * prof.lt_rate
        out["benchmark"] = self.risk.benchmark_name()
        if act and snap is not None:
            try:
                model = self.risk.load(act["id"])
                w = self.risk.holdings_weights(eid, snap, model)
                if w is not None:
                    bench = self.risk.benchmark_weights(snap, model)
                    out["tracking_error"] = float(model.tracking_error(w.reindex(model.symbols).fillna(0.0), bench.reindex(model.symbols).fillna(0.0)))
            except Exception as e:
                log.debug("TE unavailable: %s", e)
        runs = self.ctx.runs.list(limit=50)
        if not runs.empty:
            h = runs[runs["run_type"] == "harvest"]
            if not h.empty:
                out["last_run_id"] = int(h.iloc[0]["id"])
                out["last_run_at"] = str(h.iloc[0]["created_at"])
        out["sentence"] = explain_kpis(out)
        return out

    # ------------------------------------------------------------------ one click
    def one_click(self, entity_id: int | None = None, progress=None, fit_if_missing: bool = True) -> OneClickResult:
        say = progress or (lambda m: None)
        eid = entity_id or self.ctx.current_entity_id
        steps: list[str] = []
        if eid is None or not self.ctx.portfolio.held_symbols(eid):
            raise RuntimeError("No holdings yet. Import a broker file or load the demo portfolio first (step 1).")
        # 1. data
        snap = self.data.latest_snapshot()
        if snap is None or not self.data.snapshot_is_current(snap, eid):
            if not self.data.norgate_ok():
                if snap is None:
                    raise RuntimeError("Norgate Data Updater is not running and there is no saved market data. Start NDU and try again.")
                steps.append(f"Using saved market data as of {snap.as_of_date} (Norgate Data Updater is not running).")
            else:
                say("Pulling fresh market data…")
                snap = self.data.ensure_snapshot(eid, force=True, progress=say)
                steps.append(f"Pulled market data as of {snap.as_of_date}.")
        else:
            steps.append(f"Market data as of {snap.as_of_date} is current.")
        # 2. model
        act = self.risk.active()
        if act is None:
            if not fit_if_missing:
                raise RuntimeError("no risk model; fit one on the Risk model screen")
            say("Fitting the risk model (first time only)…")
            mid, _ = self.risk.fit(snap, progress=say)
            steps.append(f"Fitted risk model #{mid}.")
        else:
            steps.append(f"Using risk model #{act[0]}.")
        # 3. harvest
        say("Searching for wash-safe tax losses…")
        cfg = self.harvest.load_config()
        rid, res = self.harvest.run(eid, cfg, notes="Start-here one-click harvest")
        steps.append(f"Harvest run #{rid} saved with {len(res.trades)} tickets.")
        prof = self.ctx.tax.default_profile()
        sentences = explain_harvest(res.summary, prof.st_rate, prof.lt_rate, self.risk.benchmark_name())
        kp = self.kpis(eid)
        return OneClickResult(run_id=rid, summary=res.summary, sentences=sentences, steps=steps, kpis=kp, trades=res.trades)

    # ------------------------------------------------------------------ projection
    @staticmethod
    def wealth_projection(value: float, st_rate: float, lt_rate: float, years: int = 20, market_return: float = 0.07,
                          harvest_yield: tuple[float, ...] = (0.05, 0.035, 0.025, 0.02, 0.015, 0.012, 0.01), liquidate_at_end: bool = True) -> dict:
        """Illustrative after-tax wealth with and without harvesting.

        Losses harvested each year (a declining share of value, as a long-only book runs out of losers) save tax at the
        short-term rate, the savings are reinvested, and the basis reduction is taxed at the long-term rate at the end if
        `liquidate_at_end`. This is a teaching chart, not a forecast; assumptions are shown next to it."""
        yrs = np.arange(0, years + 1)
        v_hold = value * (1 + market_return) ** yrs
        v_tlh = np.zeros_like(v_hold)
        basis_reduction = 0.0
        w = value
        for t in range(years + 1):
            v_tlh[t] = w
            if t == years:
                break
            y = harvest_yield[min(t, len(harvest_yield) - 1)]
            harvested = w * y
            saving = harvested * st_rate
            basis_reduction += harvested
            w = (w + saving) * (1 + market_return)
        end_tax_tlh = basis_reduction * lt_rate if liquidate_at_end else 0.0
        return {"years": yrs.tolist(), "hold": v_hold.tolist(), "tlh": v_tlh.tolist(),
                "tlh_after_deferred_tax": (v_tlh - np.linspace(0, end_tax_tlh, years + 1)).tolist(),
                "assumptions": {"market_return": market_return, "harvest_yield": list(harvest_yield), "st_rate": st_rate, "lt_rate": lt_rate,
                                "end_tax_on_basis_reduction": end_tax_tlh},
                "gain_vs_hold": float(v_tlh[-1] - end_tax_tlh - v_hold[-1])}
