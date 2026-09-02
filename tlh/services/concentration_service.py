"""Embedded gains & concentration workbench: overview, glide-path planning, Monte Carlo, hedging, alternatives,
and gain-offset trade plans that pair concentrated sales with harvestable losses."""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd

from ..optim.glidepath import (
    GlidePathSpec,
    MonteCarloSpec,
    PositionFacts,
    monte_carlo,
    solve_glidepath,
    tax_curve,
)
from ..tax.concentration import (
    BracketSchedule,
    charitable_comparison,
    collar_analysis,
    concentration_stats,
    exchange_fund_breakeven,
    gift_to_lower_bracket,
    ltcg_tax,
    payoff_table,
    stepup_value,
)
from .context import AppContext
from .data_service import DataService
from .harvest_service import HarvestService
from .portfolio_service import PortfolioService
from .risk_service import RiskService

log = logging.getLogger(__name__)


class ConcentrationService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.data = DataService(ctx)
        self.risk = RiskService(ctx)
        self.portfolio = PortfolioService(ctx)
        self.harvest = HarvestService(ctx)

    # ------------------------------------------------------------------ tax assumptions
    def brackets(self) -> BracketSchedule:
        saved = self.ctx.get("bracket_schedule")
        prof = self.ctx.tax.default_profile()
        ents = self.ctx.entities.list()
        eid = self.ctx.current_entity_id
        filing = next((e["filing_status"] for e in ents if e["id"] == eid), "mfj")
        if saved:
            try:
                b = BracketSchedule.from_dict(saved)
                b.state_rate = prof.state_rate
                return b
            except Exception:  # noqa: BLE001
                pass
        return BracketSchedule.default(filing, state_rate=prof.state_rate)

    def save_brackets(self, b: BracketSchedule) -> None:
        self.ctx.set("bracket_schedule", b.to_dict())

    def other_income(self) -> float:
        return float(self.ctx.get("other_taxable_income", 300_000.0))

    def set_other_income(self, v: float) -> None:
        self.ctx.set("other_taxable_income", float(v))

    # ------------------------------------------------------------------ overview
    def overview(self, entity_id: int | None = None) -> dict:
        eid = entity_id or self.ctx.current_entity_id
        snap = self.data.latest_snapshot()
        lots = self.portfolio.lots_view(eid, snap=snap)
        if lots.empty:
            return {"positions": pd.DataFrame(), "stats": {}, "total_value": 0.0}
        sched = self.brackets()
        inc = self.other_income()
        tax_lots = lots[lots["account_type"] == "taxable"]
        g = lots.groupby("symbol").agg(market_value=("market_value", "sum"), cost_basis=("cost_basis", "sum"), unrealized=("unrealized", "sum"),
                                       n_lots=("lot_id", "count"))
        g["weight"] = g["market_value"] / g["market_value"].sum()
        g["embedded_gain_pct"] = g["unrealized"] / g["market_value"].replace(0, np.nan)
        st = lots[lots["term"] == "ST"].groupby("symbol")["unrealized"].sum()
        lt = lots[lots["term"] == "LT"].groupby("symbol")["unrealized"].sum()
        g["unrealized_st"] = st.reindex(g.index).fillna(0.0)
        g["unrealized_lt"] = lt.reindex(g.index).fillna(0.0)
        g["taxable_value"] = tax_lots.groupby("symbol")["market_value"].sum().reindex(g.index).fillna(0.0)
        # tax if liquidated today (LT at LTCG stacked on other income; ST at ordinary), taxable accounts only
        tax_if_sold = {}
        for sym in g.index:
            sub = tax_lots[tax_lots["symbol"] == sym]
            glt = float(sub.loc[sub["term"] == "LT", "unrealized"].clip(lower=0).sum())
            gst = float(sub.loc[sub["term"] == "ST", "unrealized"].clip(lower=0).sum())
            from ..tax.concentration import ordinary_tax
            tax_if_sold[sym] = ltcg_tax(glt, inc, sched)["total"] + (ordinary_tax(gst, inc + glt, sched)["total"] if gst > 0 else 0.0)
        g["tax_if_sold"] = pd.Series(tax_if_sold)
        g["after_tax_value"] = g["taxable_value"] - g["tax_if_sold"] + (g["market_value"] - g["taxable_value"])
        g["tax_drag_pct"] = g["tax_if_sold"] / g["market_value"].replace(0, np.nan)
        act = self.risk.active()
        if act is not None:
            model = act[1]
            w = g["weight"][g.index.isin(model.symbols)]
            if w.sum() > 0:
                from ..risk.analytics import risk_decomposition
                dec = risk_decomposition(model, w / w.sum())
                h = dec["holdings"]
                g["pct_of_risk"] = h["pct_of_risk"].reindex(g.index)
                g["mctr"] = h["mctr"].reindex(g.index)
                g["specific_vol"] = np.sqrt(model.specific_var.reindex(g.index))
                g["beta"] = model.exposures["market"].reindex(g.index) if "market" in model.exposures else np.nan
                g["idio_share_of_own_risk"] = (model.specific_var.reindex(g.index) /
                                               pd.Series({s: model.covariance([s]).iloc[0, 0] for s in g.index if s in model.symbols}).reindex(g.index))
                # lock-in ratio: tax cost of liquidating the name per 1% of tracking error it removes (vs benchmark)
                try:
                    bench = self.risk.benchmark_weights(snap, model)
                    wfull = w / w.sum()
                    te0 = model.tracking_error(wfull.reindex(model.symbols).fillna(0.0), bench.reindex(model.symbols).fillna(0.0))
                    red = {}
                    for sym in wfull.index:
                        w2 = wfull.drop(sym)
                        w2 = w2 / w2.sum() if w2.sum() > 0 else w2
                        te1 = model.tracking_error(w2.reindex(model.symbols).fillna(0.0), bench.reindex(model.symbols).fillna(0.0))
                        red[sym] = te0 - te1
                    g["te_reduction_if_sold"] = pd.Series(red).reindex(g.index)
                    g["lock_in_ratio"] = g["tax_if_sold"] / (g["te_reduction_if_sold"].clip(lower=1e-4) * 100)
                    g.loc[g["te_reduction_if_sold"] <= 0, "lock_in_ratio"] = np.nan
                except Exception as e:  # noqa: BLE001
                    log.debug("lock-in ratio skipped: %s", e)
        g = g.sort_values("weight", ascending=False)
        stats = concentration_stats(g["weight"].values)
        stats["gain_weighted_concentration"] = float((g["weight"] * g["embedded_gain_pct"].fillna(0).clip(lower=0)).sum())
        stats["total_embedded_gain"] = float(g["unrealized"].clip(lower=0).sum())
        stats["total_tax_if_liquidated"] = float(g["tax_if_sold"].sum())
        stats["locked_in_pct"] = stats["total_tax_if_liquidated"] / float(g["market_value"].sum()) if g["market_value"].sum() else 0.0
        stats["other_taxable_income"] = inc
        stats["brackets"] = sched.to_dict()
        return {"positions": g.reset_index(), "stats": stats, "total_value": float(g["market_value"].sum())}

    # ------------------------------------------------------------------ position facts
    def position_facts(self, symbol: str, entity_id: int | None = None) -> PositionFacts:
        eid = entity_id or self.ctx.current_entity_id
        snap = self.data.latest_snapshot()
        lots = self.portfolio.lots_view(eid, snap=snap)
        sub = lots[(lots["symbol"] == symbol.upper()) & (lots["account_type"] == "taxable")]
        if sub.empty:
            raise KeyError(f"no taxable lots of {symbol}")
        st = sub[sub["term"] == "ST"]
        years_to_lt = float(st["days_to_lt"].max() / 365.25) if len(st) else 0.0
        spec_vol, beta, tot = 0.30, 1.0, 0.35
        act = self.risk.active()
        if act is not None and symbol.upper() in act[1].symbols:
            m = act[1]
            spec_vol = float(np.sqrt(m.specific_var[symbol.upper()]))
            beta = float(m.exposures.loc[symbol.upper(), "market"]) if "market" in m.exposures else 1.0
            tot = float(np.sqrt(m.covariance([symbol.upper()]).iloc[0, 0]))
        total_wealth = float(lots["market_value"].sum())
        return PositionFacts(symbol.upper(), value=float(sub["market_value"].sum()), basis=float(sub["cost_basis"].sum()),
                             st_value=float(st["market_value"].sum()), st_basis=float(st["cost_basis"].sum()), years_to_lt=years_to_lt,
                             specific_vol=spec_vol, beta=beta, total_vol=tot, total_wealth=total_wealth,
                             weight=float(sub["market_value"].sum() / total_wealth) if total_wealth else None)

    # ------------------------------------------------------------------ planner
    def expected_losses_by_year(self, entity_id: int | None = None, years: int = 5) -> dict[int, float]:
        """Year-1 = today's harvestable loss in taxable accounts; later years use the historical tax-alpha proxy (25% decay)."""
        eid = entity_id or self.ctx.current_entity_id
        lots = self.portfolio.lots_view(eid, snap=self.data.latest_snapshot())
        base = float(-lots.loc[(lots["unrealized"] < 0) & (lots["account_type"] == "taxable") & (lots["wash_status"] == "SAFE"), "unrealized"].sum()) if not lots.empty else 0.0
        return {y: base * (0.75 ** (y - 1)) for y in range(1, years + 1)}

    def plan(self, symbol: str, spec: GlidePathSpec, entity_id: int | None = None, use_expected_losses: bool = True) -> dict:
        eid = entity_id or self.ctx.current_entity_id
        pos = self.position_facts(symbol, eid)
        sched = self.brackets()
        if use_expected_losses and not spec.losses_by_year:
            spec.losses_by_year = self.expected_losses_by_year(eid, spec.horizon_years)
        if not spec.carryforward:
            st, lt = self.ctx.tax.carryforward(eid, date.today().year - 1)
            spec.carryforward = float(st + lt)
        res = solve_glidepath(pos, spec, sched)
        return {"symbol": pos.symbol, "position": asdict(pos), "spec": asdict(spec), "brackets": sched.to_dict(), "summary": res.summary,
                "schedule": res.schedule, "comparison": res.comparison, "status": res.status,
                "tax_curve": tax_curve(np.linspace(0, max(pos.value - pos.basis, 1.0), 25), spec.other_taxable_income, sched)}

    def monte_carlo(self, symbol: str, spec: GlidePathSpec, mc: MonteCarloSpec, entity_id: int | None = None, optimised: np.ndarray | None = None) -> dict:
        pos = self.position_facts(symbol, entity_id)
        return monte_carlo(pos, spec, mc, self.brackets(), optimised)

    # ------------------------------------------------------------------ hedging & alternatives
    def hedge(self, symbol: str, T: float = 1.0, put_strike_pct: float = 0.90, call_strike_pct: float | None = None, sigma: float | None = None,
              r: float = 0.04, q: float = 0.0, entity_id: int | None = None) -> dict:
        pos = self.position_facts(symbol, entity_id)
        snap = self.data.latest_snapshot()
        px = float(self.data.prices_for(snap, [pos.symbol])[pos.symbol])
        shares = pos.value / px
        basis_ps = pos.basis / shares
        sig = sigma if sigma is not None else pos.total_vol
        sched = self.brackets()
        inc = self.other_income()
        lt_rate = ltcg_tax(max(pos.value - pos.basis, 1.0), inc, sched)["marginal_rate"]
        from ..tax.concentration import ordinary_tax
        ord_rate = ordinary_tax(max(pos.value - pos.basis, 1.0), inc, sched)["effective_rate"]
        is_lt = pos.st_value <= 0
        an = collar_analysis(px, shares, basis_ps, T, sig, r, q, put_strike_pct, call_strike_pct, is_lt, lt_rate, ord_rate)
        an["payoff"] = payoff_table(px, an["put_strike"], an["call_strike"], an["net_cost_per_share"], shares)
        an["symbol"] = pos.symbol
        an["sell_now_after_tax"] = pos.value - an["tax_if_sold_now"]
        return an

    def alternatives(self, symbol: str, agi: float | None = None, p_stepup: float = 0.3, horizon_years: float = 10, entity_id: int | None = None) -> dict:
        pos = self.position_facts(symbol, entity_id)
        sched = self.brackets()
        inc = self.other_income()
        gain = max(pos.value - pos.basis, 0.0)
        lt_rate = ltcg_tax(max(gain, 1.0), inc, sched)["marginal_rate"]
        from ..tax.concentration import marginal_ltcg_rate
        top_ord = next(r for hi, r in sched.ordinary if inc <= hi) + sched.state_rate
        return {
            "symbol": pos.symbol, "value": pos.value, "basis": pos.basis, "embedded_gain": gain, "ltcg_marginal_rate": lt_rate,
            "sell_now": {"tax": ltcg_tax(gain, inc, sched)["total"], "after_tax": pos.value - ltcg_tax(gain, inc, sched)["total"]},
            "charitable": charitable_comparison(pos.value, pos.basis, lt_rate, top_ord, agi or inc, is_long_term=pos.st_value <= 0),
            "gift": gift_to_lower_bracket(gain, lt_rate, marginal_ltcg_rate(gain, 30_000, sched)),
            "exchange_fund": exchange_fund_breakeven(pos.value, pos.basis, lt_rate),
            "stepup": stepup_value(gain, lt_rate, horizon_years, 0.04, p_stepup),
            "notes": ["Rates use the saved bracket schedule stacked on 'other taxable income'; edit both in the Concentration screen.",
                      f"Ordinary marginal rate used for deductions: {top_ord:.1%}."],
        }

    # ------------------------------------------------------------------ completion portfolio
    def completion(self, locked_symbols: list[str], n_max: int = 60, max_weight: float = 0.05, sector_band: float | None = 0.03,
                   save_as: str | None = None, entity_id: int | None = None, universe: list[str] | None = None) -> dict:
        """Keep the named positions at their current weights; optimise the remainder of the book to minimise TE."""
        from ..optim.glidepath import completion_portfolio
        eid = entity_id or self.ctx.current_entity_id
        act = self.risk.active()
        snap = self.data.latest_snapshot()
        if act is None or snap is None:
            raise RuntimeError("need an active risk model and a snapshot")
        model = act[1]
        w = self.risk.holdings_weights(eid, snap, model)
        if w is None:
            raise RuntimeError("no holdings in the model universe")
        locked = w.reindex([s.upper() for s in locked_symbols]).dropna()
        if locked.empty:
            raise ValueError("none of the locked symbols are held")
        bench = self.risk.benchmark_weights(snap, model)
        sec = snap.securities().set_index("symbol")
        etp = sec["subtype1"].fillna("").str.lower().str.startswith("exchange traded") if "subtype1" in sec else pd.Series(False, index=sec.index)
        uni = universe or [s for s in model.symbols if s in sec.index and not bool(etp.get(s, False))]
        res = completion_portfolio(model, locked, bench, uni, n_max=n_max, max_weight=max_weight, sector_band=sector_band)
        te_now = model.tracking_error(w.reindex(model.symbols).fillna(0.0), bench.reindex(model.symbols).fillna(0.0))
        out = {"locked": res["locked"].round(5).to_dict(), "free_budget": res["free_budget"], "n_free_names": res["n_free_names"], "status": res["status"],
               "te_current_book": float(te_now), "te_completion": res["te_after"], "te_locked_alone": res["te_locked_only"],
               "free_weights": res["free_weights"].round(5).to_dict(), "full_weights": res["full_weights"]}
        if save_as:
            self.ctx.baskets.save(save_as, res["full_weights"], f"completion portfolio around {', '.join(res['locked'].index)}", source="completion",
                                  benchmark_name=self.risk.benchmark_name(), params={"locked": list(res["locked"].index), "n_max": n_max, "max_weight": max_weight},
                                  metrics={"te": res["te_after"], "n_free": res["n_free_names"]}, resolve=self.ctx.resolve_assetid)
            out["saved_basket"] = save_as
        return out

    # ------------------------------------------------------------------ gain-offset plan
    def gain_offset_plan(self, symbol: str, sell_value: float, name: str | None = None, entity_id: int | None = None,
                         offset_with_losses: bool = True, replacement: str | None = None) -> tuple[int, dict]:
        """Sell `sell_value` of the concentrated name from highest-basis lots and pair it with wash-safe loss sales from the latest
        harvest recommendation so net realised gain is minimised; evaluate as an ai_plan run."""
        eid = entity_id or self.ctx.current_entity_id
        snap = self.data.latest_snapshot()
        lots = self.portfolio.lots_view(eid, snap=snap)
        sub = lots[(lots["symbol"] == symbol.upper()) & (lots["account_type"] == "taxable")].sort_values("cost_per_share", ascending=False)
        if sub.empty:
            raise KeyError(f"no taxable lots of {symbol}")
        px = float(sub["price"].iloc[0])
        trades = []
        remaining = sell_value
        gain_total = 0.0
        for _, lot in sub.iterrows():
            if remaining <= 0:
                break
            qty = min(lot["quantity"], remaining / px)
            trades.append({"side": "SELL", "symbol": symbol.upper(), "quantity": float(np.floor(qty)) if lot["quantity"] == int(lot["quantity"]) else float(qty),
                           "lot_id": int(lot["lot_id"]), "account_id": int(lot["account_id"])})
            gain_total += (px - lot["cost_per_share"]) * qty
            remaining -= qty * px
        offsets = []
        if offset_with_losses:
            losers = lots[(lots["unrealized"] < 0) & (lots["account_type"] == "taxable") & (lots["wash_status"] == "SAFE") & (lots["symbol"] != symbol.upper())]
            losers = losers.sort_values("unrealized")
            need = gain_total
            for _, lot in losers.iterrows():
                if need <= 0:
                    break
                offsets.append({"side": "SELL", "symbol": lot["symbol"], "quantity": float(lot["quantity"]), "lot_id": int(lot["lot_id"]), "account_id": int(lot["account_id"])})
                need += float(lot["unrealized"])
        buys = []
        if replacement:
            buys.append({"side": "BUY", "symbol": replacement.upper(), "quantity": float(np.floor(sell_value / float(self.data.prices_for(snap, [replacement.upper()])[replacement.upper()])))})
        rid, out = self.harvest.evaluate_trade_list(eid, name or f"Diversify {symbol.upper()} ({sell_value:,.0f})", trades + offsets + buys,
                                                    rationale=f"Concentration reduction: sell ${sell_value:,.0f} of {symbol.upper()} from highest-basis lots"
                                                              + (", offset with wash-safe harvest losses" if offsets else "") + (f", replace with {replacement}" if replacement else ""),
                                                    source="concentration")
        out["gain_from_concentrated_sale"] = gain_total
        out["n_offset_lots"] = len(offsets)
        return rid, out
