"""Live smoke of every co-pilot tool against the real state (no API calls). Run: python scripts/smoke_tools.py"""
import json
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from tlh.ai.tools import TOOLS, ToolExecutor  # noqa: E402
from tlh.services.context import AppContext  # noqa: E402

ctx = AppContext()
ex = ToolExecutor(ctx, None)
failures = []


def call(tool, **kw):
    t = time.time()
    try:
        out, _cid = ex.execute(tool, kw)
    except Exception as e:  # noqa: BLE001
        failures.append((tool, repr(e)))
        print(f"-- {tool}: EXCEPTION {e!r}")
        return None
    d = json.loads(out) if out[:1] in "{[" else out
    txt = json.dumps(d) if not isinstance(d, str) else d
    print(f"-- {tool} ({time.time() - t:.1f}s): {txt[:420]}")
    if isinstance(d, dict) and "error" in d:
        failures.append((tool, d["error"]))
    return d


call("get_portfolio_context")
call("get_risk_model_summary")
call("list_models")
call("list_runs", limit=5)
call("get_run")
call("search_securities", sector="Information Technology", min_mktcap_musd=200000, limit=5)
call("get_price_stats", symbols=["AAPL", "MSFT", "XLK", "SPY"], lookback_days=252)
call("get_substitutes", symbol="SPY")
call("screen_trade", side="BUY", symbol="ADBE")
call("screen_trade", side="SELL", symbol="PYPL")
call("optimize_basket", name="Core 40", n_max=40, sector_band=0.02, tilts={"quality": 0.3, "lowvol": 0.2}, exclude_held=True, description="smoke")
call("optimize_basket", name="Tight 12", n_max=12, sector_band=0.02, description="tight name cap: exercises relaxation")
call("create_basket", name="Two ETF", weights={"SPY": 0.6, "QQQ": 0.4}, description="explicit weights")
call("analyze_basket", name="Core 40", benchmark="SPY")
call("list_baskets")
call("get_basket", name="Tight 12")
call("run_harvest", overrides={"mode": "full_rebalance", "benchmark": "basket:Core 40", "te_budget": 0.02}, notes="toward basket")
plan = call("evaluate_trade_list", name="manual plan",
            trades=[{"side": "SELL", "symbol": "NKE", "quantity": 90}, {"side": "SELL", "symbol": "ADBE", "quantity": 10},
                    {"side": "BUY", "symbol": "ADBE", "quantity": 5}, {"side": "BUY", "symbol": "CRM", "quantity": 10}],
            rationale="ADBE buy should be flagged (loss sale 12 days ago) and it washes the ADBE sell; NKE sell safe")
if plan:
    print("   plan unsafe:", [(u["side"], u["symbol"], u["wash_status"]) for u in plan["unsafe"]])
call("list_strategies")
call("build_strategy_basket", name="S:min_variance", strategy="min_variance", params={"n_max": 30}, description="smoke")
call("build_strategy_basket", name="S:transition", strategy="tax_aware_transition", params={"gain_budget": 0.005}, target_basket="Core 40")
bt = call("backtest_strategy", strategy="hrp", params={"n_max": 25}, start="2024-01-01", rebalance="Q", benchmark_symbol="SPY", name="smoke bt")
if bt:
    call("get_backtest", run_id=int(bt["run_id"]))
ov = call("concentration_overview")
top = ov["positions"][0]["symbol"] if ov and ov.get("positions") else "AAPL"
call("diversification_plan", symbol=top, horizon_years=4, annual_gain_budget=50000)
call("hedge_analysis", symbol=top, tenor_years=1.0, put_strike_pct=0.9)
call("concentration_alternatives", symbol=top, agi=400000)
call("stress_test", preset="Market -2σ")
call("risk_decomposition")
call("parametric_var")
m = call("fit_risk_model", overrides={"lookback_days": 756, "use_macro": True}, name="3y macro variant")
if m:
    call("compare_models", model_a=int(m["model_version_id"]), model_b=2)
call("list_style_factors")
call("list_editable_modules")
call("read_module", path="tlh/risk/custom/__init__.py")
call("set_benchmark", benchmark="S&P 500")
call("run_frontier", te_grid=[0.01, 0.03])
print(f"\n{len(TOOLS)} tools defined; failures: {failures if failures else 'none'}")
