"""Harvest runs: assemble inputs from live state, run the optimizer, persist an immutable run snapshot."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..optim.frontier import frontier as _frontier
from ..optim.frontier import priority_comparison as _priority
from ..optim.harvest import HarvestConfig, HarvestInputs, HarvestResult, run_harvest
from .context import AppContext
from .data_service import DataService
from .portfolio_service import PortfolioService
from .risk_service import RiskService

log = logging.getLogger(__name__)


class HarvestService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.data = DataService(ctx)
        self.portfolio = PortfolioService(ctx)
        self.risk = RiskService(ctx)

    # ------------------------------------------------------------------ config persistence
    def load_config(self) -> HarvestConfig:
        saved = self.ctx.get("harvest_config")
        if saved:
            try:
                saved["priority"] = tuple(saved.get("priority", HarvestConfig().priority))
                saved["priority_weights"] = tuple(saved.get("priority_weights", HarvestConfig().priority_weights))
                return HarvestConfig(**saved)
            except TypeError:
                pass
        return HarvestConfig()

    def save_config(self, cfg: HarvestConfig) -> None:
        self.ctx.set("harvest_config", asdict(cfg))

    # ------------------------------------------------------------------ inputs
    def build_inputs(self, entity_id: int, as_of: date | None = None, snap=None, model=None,
                     benchmark_name: str | None = None) -> tuple[HarvestInputs, dict]:
        as_of = as_of or date.today()
        snap = snap or self.data.latest_snapshot()
        if snap is None:
            raise RuntimeError("no data snapshot; refresh data first")
        if model is None:
            act = self.risk.active()
            if act is None:
                raise RuntimeError("no active risk model; fit one first")
            model_id, model = act
        else:
            model_id = None
        book = self.portfolio.book(entity_id)
        held = [lot.symbol for lot in book.open_lots()]
        px = self.data.prices_for(snap, held)
        bench = self.risk.benchmark_weights(snap, model, benchmark_name)
        sec = snap.securities().set_index("symbol")
        rets = self.data.returns_matrix(snap, 300)
        inp = HarvestInputs(
            as_of=as_of, lots=book.open_lots(), prices=px, model=model, benchmark=bench,
            tax=self.ctx.tax.default_profile(), groups=book.groups, substitutes=self.ctx.substitutes,
            acquisitions=book.acquisitions(include_scheduled=True),
            recent_loss_sales=book.loss_sales(since=as_of - timedelta(days=31)),
            returns=rets, securities=sec, universe=list(model.symbols),
        )
        meta = {"snapshot_id": snap.id, "model_version_id": model_id, "benchmark": benchmark_name or self.risk.benchmark_name()}
        return inp, meta

    # ------------------------------------------------------------------ runs
    def run(self, entity_id: int, cfg: HarvestConfig | None = None, as_of: date | None = None, persist: bool = True,
            notes: str | None = None, benchmark_name: str | None = None) -> tuple[int | None, HarvestResult]:
        cfg = cfg or self.load_config()
        inp, meta = self.build_inputs(entity_id, as_of, benchmark_name=benchmark_name)
        res = run_harvest(inp, cfg)
        run_id = None
        if persist:
            run_id = self._persist(entity_id, inp, res, meta, "harvest", notes)
        return run_id, res

    def _persist(self, entity_id: int, inp: HarvestInputs, res: HarvestResult, meta: dict, run_type: str,
                 notes: str | None) -> int:
        params = {**asdict(res.config), **meta}
        rid = self.ctx.runs.create(run_type, inp.as_of, entity_id, meta["snapshot_id"], meta["model_version_id"],
                                   params, res.summary, notes=notes)
        folder = self.ctx.settings.runs_dir / f"run_{rid:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        res.trades.to_parquet(folder / "trades.parquet", index=False) if not res.trades.empty else None
        res.blocked.to_parquet(folder / "blocked.parquet", index=False) if not res.blocked.empty else None
        res.replacements.to_parquet(folder / "replacements.parquet", index=False) if not res.replacements.empty else None
        pd.DataFrame({"before": res.exposures_before, "after": res.exposures_after}).to_parquet(folder / "exposures.parquet")
        res.te_decomp_before.to_parquet(folder / "te_before.parquet")
        res.te_decomp_after.to_parquet(folder / "te_after.parquet")
        pd.DataFrame({"before": res.sector_before, "after": res.sector_after}).to_parquet(folder / "sectors.parquet")
        pd.DataFrame({"before": res.weights_before, "after": res.weights_after}).to_parquet(folder / "weights.parquet")
        (folder / "summary.json").write_text(json.dumps(res.to_dict(), indent=2, default=str), encoding="utf-8")
        self.ctx.db.update("runs", "id = ?", (rid,), artifact_path=str(folder))
        if not res.trades.empty:
            cols = ["account_id", "assetid", "symbol", "side", "quantity", "est_price", "est_value", "lot_id",
                    "realized_gain", "term", "tax_benefit", "wash_status", "wash_explanation", "replacement_for"]
            recs = []
            for r in res.trades[cols].to_dict("records"):
                recs.append({k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()})
            self.ctx.runs.add_trades(rid, recs)
        return rid

    def load_run(self, run_id: int) -> dict | None:
        row = self.ctx.runs.get(run_id)
        if not row:
            return None
        folder = Path(row["artifact_path"]) if row.get("artifact_path") else None
        out = dict(row)
        out["trades"] = self.ctx.runs.trades(run_id)

        def rd(name):
            p = folder / name if folder else None
            return pd.read_parquet(p) if p and p.exists() else pd.DataFrame()

        out["blocked"] = rd("blocked.parquet")
        out["replacements"] = rd("replacements.parquet")
        out["exposures"] = rd("exposures.parquet")
        out["te_before"] = rd("te_before.parquet")
        out["te_after"] = rd("te_after.parquet")
        out["sectors"] = rd("sectors.parquet")
        return out

    # ------------------------------------------------------------------ sweeps
    def frontier(self, entity_id: int, cfg: HarvestConfig | None = None, as_of: date | None = None,
                 te_grid: list[float] | None = None, benchmark_name: str | None = None) -> pd.DataFrame:
        cfg = cfg or self.load_config()
        inp, meta = self.build_inputs(entity_id, as_of, benchmark_name=benchmark_name)
        df = _frontier(inp, cfg, te_grid)
        self.ctx.runs.create("frontier", inp.as_of, entity_id, meta["snapshot_id"], meta["model_version_id"],
                             {**asdict(cfg), **meta}, {"rows": df.to_dict("records")})
        return df

    def priority_comparison(self, entity_id: int, cfg: HarvestConfig | None = None, as_of: date | None = None):
        cfg = cfg or self.load_config()
        inp, _ = self.build_inputs(entity_id, as_of)
        return _priority(inp, cfg)

    # ------------------------------------------------------------------ acting on a run
    def mark_acted(self, trade_ids: list[int], acted: bool = True) -> None:
        self.ctx.runs.mark_acted(trade_ids, acted)

    def book_trades(self, run_id: int, trade_ids: list[int], trade_date: date | None = None) -> list[str]:
        """Record selected recommended trades as executed transactions (paper booking, no order routing)."""
        trade_date = trade_date or date.today()
        trades = self.ctx.runs.trades(run_id)
        done = []
        for _, t in trades[trades["id"].isin(trade_ids)].iterrows():
            if t["side"] == "SELL":
                from ..tax.lots import LotMethod
                self.ctx.portfolio.record_sale(int(t["account_id"]), t["symbol"], int(t["assetid"]), trade_date,
                                               float(t["quantity"]), float(t["est_price"]), method=LotMethod.SPECIFIC,
                                               specific_ids=[int(t["lot_id"])], groups=self.ctx.groups(), source=f"run:{run_id}")
            else:
                self.ctx.portfolio.record_purchase(int(t["account_id"]), t["symbol"], int(t["assetid"]), trade_date,
                                                   float(t["quantity"]), float(t["est_price"]), groups=self.ctx.groups(),
                                                   notes=f"run:{run_id}")
            done.append(f"{t['side']} {t['quantity']:g} {t['symbol']}")
        self.mark_acted(trade_ids, True)
        return done


    # ------------------------------------------------------------------ AI / manual trade plans
    def evaluate_trade_list(self, entity_id: int, name: str, trades: list[dict], benchmark_name: str | None = None,
                            rationale: str | None = None, source: str = "manual", as_of: date | None = None) -> tuple[int, dict]:
        """Wash-screen and risk-evaluate an arbitrary trade plan, then persist it as an 'ai_plan' run.

        Each trade: {side: BUY|SELL, symbol, quantity, lot_id?, account_id?}. Sells without lot_id use the
        highest-basis open lots of that symbol. Nothing is executed.
        """
        import numpy as np

        from ..tax.lots import LotMethod, select_lots
        from ..tax.washsale import Acquisition, LossSale, screen_proposed_buy, screen_proposed_sale

        inp, meta = self.build_inputs(entity_id, as_of, benchmark_name=benchmark_name)
        model, px, tax = inp.model, inp.prices, inp.tax
        accounts = {a.id: a for a in self.ctx.entities.accounts(entity_id)}
        taxable = [a.id for a in accounts.values() if a.account_type == "taxable"]
        rows: list[dict] = []
        plan_buys: list[Acquisition] = []
        plan_loss_sales: list[LossSale] = []
        for t in trades:                                   # sells first: they define what buys would wash
            if str(t.get("side", "")).upper() != "SELL":
                continue
            sym = str(t["symbol"]).upper()
            qty = float(t["quantity"])
            lots = [lot for lot in inp.lots if lot.symbol == sym and lot.quantity_open > 0
                    and (t.get("lot_id") is None or lot.id == int(t["lot_id"]))
                    and (t.get("account_id") is None or lot.account_id == int(t["account_id"]))]
            if not lots:
                rows.append({"side": "SELL", "symbol": sym, "quantity": qty, "wash_status": "ERROR", "wash_explanation": "no matching open lot"})
                continue
            price = float(px.get(sym, np.nan))
            slices = select_lots(lots, min(qty, sum(lot.quantity_open for lot in lots)), LotMethod.HIFO)
            for sl in slices:
                lot = sl.lot
                gain = (price - lot.basis_per_share) * sl.quantity if price == price else float("nan")
                term = lot.term_at(inp.as_of)
                det = None
                if gain < 0:
                    det = screen_proposed_sale(lot.assetid, sym, lot.account_id, inp.as_of, sl.quantity, -gain, inp.acquisitions, inp.groups, lot_id=lot.id)
                    plan_loss_sales.append(LossSale(lot.assetid, sym, lot.account_id, inp.as_of, sl.quantity, -gain, lot_id=lot.id))
                rows.append({"side": "SELL", "account_id": lot.account_id, "symbol": sym, "assetid": lot.assetid, "lot_id": lot.id,
                             "quantity": sl.quantity, "est_price": price, "est_value": price * sl.quantity, "realized_gain": gain, "term": term,
                             "tax_benefit": tax.benefit_of_loss(-gain, term) if gain < 0 else -tax.tax_on_gain(gain, term),
                             "tax_alpha": tax.tax_alpha(-gain, term) if gain < 0 else None,
                             "wash_status": det.status if det else "N/A (gain)",
                             "wash_explanation": det.explanation if det else "Sale at a gain; wash-sale rules do not apply.",
                             "replacement_for": None, "holding_start": lot.holding_start_date, "basis_per_share": lot.basis_per_share})
        for t in trades:
            if str(t.get("side", "")).upper() != "BUY":
                continue
            sym = str(t["symbol"]).upper()
            qty = float(t["quantity"])
            aid = self.ctx.resolve_assetid(sym)
            acct = int(t.get("account_id") or (taxable[0] if taxable else next(iter(accounts))))
            price = float(px.get(sym, np.nan))
            if aid is None or price != price:
                rows.append({"side": "BUY", "symbol": sym, "quantity": qty, "account_id": acct, "wash_status": "ERROR",
                             "wash_explanation": "unknown symbol or no price in snapshot"})
                continue
            scr = screen_proposed_buy(aid, sym, inp.as_of, inp.recent_loss_sales + plan_loss_sales, inp.groups)
            plan_buys.append(Acquisition(aid, sym, acct, accounts[acct].account_type if acct in accounts else "taxable", inp.as_of, qty, kind="scheduled_buy"))
            rows.append({"side": "BUY", "account_id": acct, "symbol": sym, "assetid": aid, "lot_id": None, "quantity": qty, "est_price": price,
                         "est_value": price * qty, "realized_gain": None, "term": None, "tax_benefit": None, "tax_alpha": None,
                         "wash_status": scr.status, "wash_explanation": scr.explanation, "replacement_for": t.get("replacement_for"),
                         "holding_start": None, "basis_per_share": None})
        if plan_buys:                                        # a buy in the plan can wash the plan's own loss sells
            acqs = inp.acquisitions + plan_buys
            for r in rows:
                if r["side"] == "SELL" and r.get("realized_gain") is not None and r["realized_gain"] < 0:
                    det = screen_proposed_sale(r["assetid"], r["symbol"], r["account_id"], inp.as_of, r["quantity"], -r["realized_gain"], acqs, inp.groups, lot_id=r["lot_id"])
                    if det.status != "SAFE":
                        r["wash_status"], r["wash_explanation"] = det.status, det.explanation
        trades_df = pd.DataFrame(rows)
        hold = pd.Series(0.0, index=sorted({lot.symbol for lot in inp.lots if lot.symbol in px.index}))
        for lot in inp.lots:
            if lot.symbol in hold.index:
                hold[lot.symbol] += lot.quantity_open * px[lot.symbol]
        after = hold.copy()
        for r in rows:
            if r.get("est_value") is None or r["wash_status"] == "ERROR":
                continue
            after[r["symbol"]] = after.get(r["symbol"], 0.0) + (r["est_value"] if r["side"] == "BUY" else -r["est_value"])
        known = sorted(s for s in set(hold.index) | set(after.index) | set(inp.benchmark.index) if s in model.symbols)
        w0 = hold.reindex(known).fillna(0.0)
        w0 = w0 / w0.sum() if w0.sum() else w0
        w1 = after.reindex(known).fillna(0.0).clip(lower=0)
        w1 = w1 / w1.sum() if w1.sum() else w1
        wb = inp.benchmark.reindex(known).fillna(0.0)
        wb = wb / wb.sum() if wb.sum() else wb
        dec0, dec1 = model.te_decomposition(w0, wb), model.te_decomposition(w1, wb)
        exp0, exp1 = model.portfolio_exposures(w0), model.portfolio_exposures(w1)
        sector_cols = [c for c in model.factors if c.startswith(("sec:", "ind:"))]
        Xs = model.exposures.loc[known, sector_cols]
        sec0 = pd.Series((Xs.T @ w0).values, index=[c.split(':', 1)[1] for c in sector_cols])
        sec1 = pd.Series((Xs.T @ w1).values, index=[c.split(':', 1)[1] for c in sector_cols])
        empty = pd.DataFrame()
        sells = trades_df[trades_df["side"] == "SELL"] if not trades_df.empty else empty
        buys = trades_df[trades_df["side"] == "BUY"] if not trades_df.empty else empty
        unsafe = trades_df[~trades_df["wash_status"].isin(["SAFE", "N/A (gain)"])] if not trades_df.empty else empty
        loss_sells = sells[sells["realized_gain"] < 0] if len(sells) else empty
        style_cols = [c for c in model.factors if c in model.spec.styles]
        summary = {
            "as_of": inp.as_of.isoformat(), "mode": "ai_plan", "name": name, "priority": [], "portfolio_value": float(hold.sum()),
            "n_sells": int(len(sells)), "n_buys": int(len(buys)),
            "sell_value": float(sells["est_value"].sum()) if len(sells) else 0.0, "buy_value": float(buys["est_value"].sum()) if len(buys) else 0.0,
            "harvested_loss": float(-loss_sells["realized_gain"].sum()) if len(loss_sells) else 0.0,
            "harvested_loss_st": float(-loss_sells.loc[loss_sells["term"] == "ST", "realized_gain"].sum()) if len(loss_sells) else 0.0,
            "harvested_loss_lt": float(-loss_sells.loc[loss_sells["term"] == "LT", "realized_gain"].sum()) if len(loss_sells) else 0.0,
            "realized_gains": float(sells.loc[sells["realized_gain"] > 0, "realized_gain"].sum()) if len(sells) else 0.0,
            "tax_benefit": float(sells["tax_benefit"].fillna(0).sum()) if len(sells) else 0.0,
            "tax_alpha": float(loss_sells["tax_alpha"].fillna(0).sum()) if len(loss_sells) else 0.0,
            "tax_alpha_bps": float(loss_sells["tax_alpha"].fillna(0).sum() / hold.sum() * 1e4) if len(loss_sells) and hold.sum() else 0.0,
            "te_before": dec0.attrs["tracking_error"], "te_after": dec1.attrs["tracking_error"],
            "turnover": float(((sells["est_value"].sum() if len(sells) else 0.0) + (buys["est_value"].sum() if len(buys) else 0.0)) / hold.sum()) if hold.sum() else 0.0,
            "n_unsafe_trades": int(len(unsafe)), "n_blocked_lots": int(len(unsafe)), "n_candidate_lots": 0, "n_buy_candidates": 0,
            "total_harvestable_loss": None,
            "max_style_drift": float(np.abs((exp1 - exp0).reindex(style_cols)).max()) if style_cols else 0.0,
            "max_sector_drift": float(np.abs(sec1 - sec0).max()) if len(sec0) else 0.0,
            "solver_status": "ai_plan" if len(unsafe) == 0 else f"ai_plan ({len(unsafe)} wash-unsafe trades)",
            "rationale": rationale, "source": source,
        }
        res = HarvestResult(trades=trades_df, blocked=unsafe, replacements=pd.DataFrame(), summary=summary,
                            exposures_before=exp0, exposures_after=exp1, te_decomp_before=dec0, te_decomp_after=dec1,
                            sector_before=sec0, sector_after=sec1, config=self.load_config(), solver_status=summary["solver_status"],
                            weights_before=w0, weights_after=w1)
        rid = self._persist(entity_id, inp, res, meta, "ai_plan", notes=f"{name}: {rationale or ''}"[:500])
        out_trades = json.loads(trades_df.drop(columns=["wash_explanation"], errors="ignore").to_json(orient="records", date_format="iso")) if not trades_df.empty else []
        out_unsafe = json.loads(unsafe.to_json(orient="records", date_format="iso")) if len(unsafe) else []
        return rid, {"summary": summary, "trades": out_trades, "unsafe": out_unsafe}
