"""Walk-forward backtester for construction strategies.

At each rebalance date the strategy sees only trailing data: prices up to t (returns, shrunk sample covariance,
price-based style signals via `build_exposures`), current fundamentals (documented look-ahead for value/quality/
growth signals), and optionally point-in-time index membership. Weights drift with returns between rebalances;
one-way turnover is charged at `cost_bps`.

Caveats surfaced in the result: survivorship (universe = today's constituents unless membership is supplied) and
fundamental look-ahead. This module is AI-editable (ai/registry.py).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..risk.factors import FactorInputs, build_exposures
from .strategies import StrategyInputs, StrategySpec, run_strategy, shrunk_sample_cov

log = logging.getLogger(__name__)


@dataclass
class BacktestSpec:
    start: str | None = None                # ISO date; default = lookback after first price
    end: str | None = None
    rebalance: str = "M"                    # M | Q | W
    lookback_days: int = 252
    cost_bps: float = 5.0
    benchmark_symbol: str | None = None     # ETF in the price panel; None -> cap-weighted proxy from shares x price
    use_membership: bool = True
    styles: list[str] = field(default_factory=lambda: ["momentum", "lowvol", "size", "value", "quality", "growth"])
    max_universe: int = 600


@dataclass
class BacktestResult:
    equity: pd.Series
    bench_equity: pd.Series
    weights: pd.DataFrame                   # rebalance date x symbol
    turnover: pd.Series
    metrics: dict
    spec: BacktestSpec
    strategy: StrategySpec
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"metrics": self.metrics, "spec": asdict(self.spec), "strategy": asdict(self.strategy), "warnings": self.warnings,
                "n_rebalances": int(len(self.weights))}


def _rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    s = pd.Series(1, index=idx)
    rule = {"M": "ME", "Q": "QE", "W": "W-FRI"}.get(freq.upper(), "ME")
    return list(s.resample(rule).last().dropna().index.map(lambda d: idx[idx <= d][-1]).unique())


def run_backtest(prices: pd.DataFrame, securities: pd.DataFrame, fundamentals: pd.DataFrame, strategy: StrategySpec,
                 spec: BacktestSpec, membership: pd.DataFrame | None = None, rf_daily: pd.Series | None = None,
                 progress=None) -> BacktestResult:
    say = progress or (lambda m: None)
    warns: list[str] = []
    px = prices.sort_index()
    if spec.end:
        px = px.loc[: pd.Timestamp(spec.end)]
    rets = px.pct_change()
    first = px.index[min(spec.lookback_days, len(px) - 2)]
    start = max(pd.Timestamp(spec.start), first) if spec.start else first
    dates = px.index[px.index >= start]
    if len(dates) < 40:
        raise ValueError("not enough history after lookback for a backtest")
    rebal = [d for d in _rebalance_dates(px.index, spec.rebalance) if d >= start and d < dates[-1]]
    if not rebal or rebal[0] > start:
        rebal = [start] + rebal
    is_etp = securities["subtype1"].fillna("").str.lower().str.startswith("exchange traded") if "subtype1" in securities else pd.Series(False, index=securities.index)
    stocks = [s for s in px.columns if s in securities.index and not bool(is_etp.get(s, False))]
    if membership is None or membership.empty:
        warns.append("survivorship: universe is today's constituents (no point-in-time membership in snapshot)")
    if any(s in ("value", "quality", "growth") for s in spec.styles):
        warns.append("look-ahead: value/quality/growth signals use current fundamentals throughout the history")
    shares = pd.to_numeric(securities.get("shares_outstanding"), errors="coerce").reindex(stocks) if "shares_outstanding" in securities else pd.Series(np.nan, index=stocks)
    sectors = securities["gics_sector"].reindex(stocks) if "gics_sector" in securities else None

    # benchmark daily returns
    if spec.benchmark_symbol and spec.benchmark_symbol in rets.columns:
        bench_r = rets[spec.benchmark_symbol]
        bench_name = spec.benchmark_symbol
    else:
        capw = (shares.values[None, :] * px[stocks].values)
        capw = np.nan_to_num(capw)
        w_prev = capw[:-1] / np.clip(capw[:-1].sum(axis=1, keepdims=True), 1e-12, None)
        bench_r = pd.Series((w_prev * np.nan_to_num(rets[stocks].values[1:])).sum(axis=1), index=px.index[1:])
        bench_name = "cap-weighted proxy (today's shares x price)"
        warns.append("benchmark is a cap-weighted proxy built from today's share counts")

    weights_hist: dict[pd.Timestamp, pd.Series] = {}
    turnover: dict[pd.Timestamp, float] = {}
    port_r = pd.Series(0.0, index=dates)
    w_cur: pd.Series | None = None
    say(f"Backtesting {strategy.kind}: {len(rebal)} rebalances from {rebal[0].date()} to {dates[-1].date()}")
    rebal_set = set(rebal)
    for i, d in enumerate(dates):
        if d in rebal_set:
            try:
                w_new = _weights_at(d, px, rets, stocks, securities, fundamentals, shares, sectors, membership, strategy, spec)
                to = float(np.abs(w_new.reindex(w_new.index.union(w_cur.index if w_cur is not None else w_new.index)).fillna(0.0)
                                  - (w_cur.reindex(w_new.index.union(w_cur.index)).fillna(0.0) if w_cur is not None else 0.0)).sum() / 2) if w_cur is not None else 1.0
                turnover[d] = to
                weights_hist[d] = w_new
                w_cur = w_new
                cost = to * spec.cost_bps / 1e4
            except Exception as e:  # keep the backtest alive, record the failure
                log.warning("rebalance %s failed: %s", d.date(), e)
                warns.append(f"rebalance {d.date()} failed: {e}")
                cost = 0.0
            if (i % 6) == 0:
                say(f"  {d.date()} · {len(w_cur) if w_cur is not None else 0} names · turnover {turnover.get(d, 0):.0%}")
        else:
            cost = 0.0
        if w_cur is None:
            continue
        if i == 0:
            port_r[d] = -cost
            continue
        r = rets.loc[d, w_cur.index].fillna(0.0)
        port_r[d] = float((w_cur * r).sum()) - cost
        w_cur = w_cur * (1 + r)
        w_cur = w_cur / w_cur.sum()
    port_r = port_r.loc[rebal[0]:]
    bench_r = bench_r.reindex(port_r.index).fillna(0.0)
    equity = (1 + port_r).cumprod()
    bench_eq = (1 + bench_r).cumprod()
    rf = (rf_daily.reindex(port_r.index).fillna(0.0) if rf_daily is not None else pd.Series(0.0, index=port_r.index))
    metrics = _metrics(port_r, bench_r, rf, pd.Series(turnover), weights_hist)
    metrics["benchmark"] = bench_name
    W = pd.DataFrame(weights_hist).T.fillna(0.0)
    return BacktestResult(equity=equity, bench_equity=bench_eq, weights=W, turnover=pd.Series(turnover), metrics=metrics,
                          spec=spec, strategy=strategy, warnings=warns)


def _weights_at(d, px, rets, stocks, securities, fundamentals, shares, sectors, membership, strategy: StrategySpec,
                spec: BacktestSpec) -> pd.Series:
    hist = px.loc[:d]
    look = hist.iloc[-(spec.lookback_days + 1):]
    ok = look.notna().sum() >= int(0.9 * len(look))
    uni = [s for s in stocks if ok.get(s, False)]
    if membership is not None and not membership.empty and spec.use_membership:
        m = membership.reindex(columns=uni)
        row = m.loc[: d].iloc[-1] if (m.index <= d).any() else None
        if row is not None:
            uni = [s for s in uni if bool(row.get(s, False))]
    if len(uni) > spec.max_universe:
        mc = (shares.reindex(uni) * hist[uni].iloc[-1]).sort_values(ascending=False)
        uni = list(mc.index[: spec.max_universe])
    R = look[uni].pct_change().iloc[1:]
    cov = shrunk_sample_cov(R)
    uni = [s for s in uni if s in cov.index]
    mktcap = (shares.reindex(uni) * hist[uni].iloc[-1]).fillna(0.0)
    bench = mktcap / mktcap.sum() if mktcap.sum() > 0 else pd.Series(1.0 / len(uni), index=uni)
    signals = None
    exposures = None
    if strategy.kind in ("mean_variance", "black_litterman", "factor_tilt") or strategy.kind == "stratified_index":
        fi = FactorInputs(prices=hist[uni], t=d, fundamentals=fundamentals.reindex(uni), securities=securities.reindex(uni))
        styles = [s for s in spec.styles if s in ("momentum", "lowvol", "size", "value", "quality", "growth")]
        b = build_exposures(fi, styles, use_sectors=True)
        exposures = b.exposures
        signals = b.exposures[styles]
    inp = StrategyInputs(symbols=uni, cov=cov, benchmark=bench, returns=R, signals=signals, exposures=exposures,
                         sectors=sectors.reindex(uni) if sectors is not None else None, mktcap=mktcap)
    return run_strategy(strategy, inp).weights


def _metrics(port_r: pd.Series, bench_r: pd.Series, rf: pd.Series, turnover: pd.Series, weights_hist: dict) -> dict:
    n = len(port_r)
    years = n / 252
    ex = port_r - rf
    active = port_r - bench_r
    eq = (1 + port_r).cumprod()
    beq = (1 + bench_r).cumprod()
    dd = eq / eq.cummax() - 1
    bdd = beq / beq.cummax() - 1

    def ann(r):
        return float((1 + r).prod() ** (1 / max(years, 1e-9)) - 1)

    return {
        "start": str(port_r.index[0].date()), "end": str(port_r.index[-1].date()), "years": round(years, 2),
        "total_return": float(eq.iloc[-1] - 1), "bench_total_return": float(beq.iloc[-1] - 1),
        "cagr": ann(port_r), "bench_cagr": ann(bench_r),
        "ann_vol": float(port_r.std() * np.sqrt(252)), "bench_ann_vol": float(bench_r.std() * np.sqrt(252)),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0,
        "bench_sharpe": float((bench_r - rf).mean() / (bench_r - rf).std() * np.sqrt(252)) if bench_r.std() > 0 else 0.0,
        "max_drawdown": float(dd.min()), "bench_max_drawdown": float(bdd.min()),
        "tracking_error": float(active.std() * np.sqrt(252)),
        "information_ratio": float(active.mean() / active.std() * np.sqrt(252)) if active.std() > 0 else 0.0,
        "active_return_ann": float(ann(port_r) - ann(bench_r)),
        "avg_turnover_per_rebalance": float(turnover.iloc[1:].mean()) if len(turnover) > 1 else 0.0,
        "annual_turnover": float(turnover.iloc[1:].sum() / max(years, 1e-9)) if len(turnover) > 1 else 0.0,
        "avg_names": float(np.mean([int((w > 0).sum()) for w in weights_hist.values()])) if weights_hist else 0.0,
        "hit_rate_monthly": float((active.resample("ME").sum() > 0).mean()) if n > 40 else 0.0,
        "n_rebalances": int(len(weights_hist)),
    }
