"""Sweep design, multi-core execution and the result store for the harvesting research.

A study = base parameters + the sweeps requested + rolling windows. Each (parameter set, window) is one `run_window`
call; they are independent, so they run across all cores with a ProcessPoolExecutor (each worker memory-maps the store
once). Results land in var/research/studies/<study>/results.parquet as one row per run with the flat metrics, plus the
monthly series in monthly.parquet for the charts. Summaries aggregate across windows (median and interquartile range),
which is what a due-diligence reviewer wants: the *distribution* of outcomes over start years, not one lucky decade.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import APPROACHES, ResearchSpec, StudySpec

log = logging.getLogger(__name__)

SWEEP_LABELS = {
    "account_size": "Account size", "basket_size": "Basket size (names)", "trigger": "Harvest trigger", "approach": "Harvesting approach",
    "concentrated": "Concentrated start (size x embedded gain)", "horizon": "Window length",
}
SUMMARY_METRICS = ["harvested_per_year_pct", "harvested_pct_of_start", "tax_value_pct_of_start", "harvest_life_months", "harvest_half_life_months",
                   "te_realised", "te_forecast_avg", "excess_return_annual", "turnover_annual", "names_avg", "wash_blocked", "unrealised_gain_end_pct",
                   "conc_months_to_diversify", "trades"]


# ---------------------------------------------------------------------- design
def window_starts(study: StudySpec, horizon: int, last_data_year: int) -> list[int]:
    last = study.last_start_year if study.last_start_year is not None else last_data_year - horizon
    return list(range(study.first_start_year, last + 1, max(study.every_n_years, 1)))


def design(study: StudySpec, last_data_year: int) -> list[dict]:
    """Every run to execute: {'sweep', 'level', 'label', 'spec'}. The base case is included once per window under sweep='base'."""
    runs: list[dict] = []
    seen: set[str] = set()

    def add(sweep: str, level, label: str, spec: ResearchSpec):
        key = json.dumps(spec.to_dict(), sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        runs.append({"sweep": sweep, "level": level, "label": label, "spec": spec})

    for horizon in study.horizons:
        for y in window_starts(study, horizon, last_data_year):
            base = study.base.with_(horizon_years=horizon, start_year=y)
            add("base", "base", "base case", base)
            for sw in study.sweeps:
                if sw == "account_size":
                    for v in study.account_sizes:
                        add(sw, v, f"${v:,.0f}", base.with_(account_size=float(v)))
                elif sw == "basket_size":
                    for v in study.basket_sizes:
                        add(sw, v, f"{v} names", base.with_(basket_size=int(v)))
                elif sw == "trigger":
                    for v in study.triggers:
                        add(sw, v, f"{v:.2%}", base.with_(trigger=float(v)))
                elif sw == "approach":
                    for v in study.approaches:
                        if v in APPROACHES:
                            add(sw, v, v, base.with_(approach=v))
                elif sw == "concentrated":
                    for cs in study.concentrated_sizes:
                        for g in study.concentrated_gains:
                            add(sw, f"{cs:.0%} @ {g:.0%}", f"{cs:.0%} position, {g:.0%} gain", base.with_(concentrated_pct=float(cs), concentrated_gain=float(g)))
                elif sw == "horizon":
                    pass   # covered by the horizons loop
    return runs


def estimate(study: StudySpec, last_data_year: int) -> dict:
    runs = design(study, last_data_year)
    n_opt = sum(1 for r in runs if r["spec"].approach == "optimizer")
    secs = n_opt * 32 * (study.base.horizon_years / 10) + (len(runs) - n_opt) * 4 * (study.base.horizon_years / 10)
    return {"runs": len(runs), "optimizer_runs": n_opt, "approx_seconds_single_core": int(secs),
            "approx_minutes_with_workers": round(secs / max(os.cpu_count() or 1, 1) / 60, 1)}


# ---------------------------------------------------------------------- execution
_STORE = None


def _worker_init(store_root: str) -> None:
    global _STORE
    from .data import load_store
    _STORE = load_store(Path(store_root))


def _run_one(payload: dict) -> dict:
    from .engine import run_window
    spec = ResearchSpec(**payload["spec"])
    t0 = time.time()
    try:
        res = run_window(_STORE, spec)
        m = res.metrics
        monthly = res.monthly.reset_index()
        monthly["run_id"] = payload["run_id"]
        return {"run_id": payload["run_id"], "ok": True, "metrics": m, "monthly": monthly.to_dict("list"), "seconds": time.time() - t0,
                "warnings": res.warnings}
    except Exception as e:  # noqa: BLE001
        return {"run_id": payload["run_id"], "ok": False, "error": repr(e), "seconds": time.time() - t0}


def run_study(study: StudySpec, store_root: Path, out_root: Path, last_data_year: int, workers: int | None = None, progress=None,
              cancel=None) -> Path:
    """Execute the whole design; returns the study folder. Re-running skips runs already in results.parquet (resumable)."""
    say = progress or (lambda m: None)
    out = Path(out_root) / _slug(study.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "study.json").write_text(json.dumps(asdict(study), default=str, indent=1), encoding="utf-8")
    runs = design(study, last_data_year)
    done_ids: set[str] = set()
    rows: list[dict] = []
    monthly_frames: list[pd.DataFrame] = []
    if (out / "results.parquet").exists():
        prev = pd.read_parquet(out / "results.parquet")
        done_ids = set(prev["run_id"])
        rows = prev.to_dict("records")
        if (out / "monthly.parquet").exists():
            monthly_frames.append(pd.read_parquet(out / "monthly.parquet"))
    todo = []
    for r in runs:
        rid = _run_id(r)
        if rid in done_ids:
            continue
        todo.append({"run_id": rid, "sweep": r["sweep"], "level": str(r["level"]), "label": r["label"], "spec": r["spec"].to_dict()})
    say(f"{len(runs)} runs in design, {len(todo)} to execute on {workers or os.cpu_count()} workers")
    if not todo:
        _write(out, rows, monthly_frames)
        return out
    meta = {p["run_id"]: p for p in todo}
    t0 = time.time()
    n_done = 0
    workers = workers or max((os.cpu_count() or 2) - 1, 1)
    import multiprocessing as mp
    # spawn context: workers re-import the caller's __main__, which must therefore be guarded by if __name__ == '__main__'
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(str(store_root),), mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_run_one, p): p["run_id"] for p in todo}
        for fut in as_completed(futs):
            r = fut.result()
            p = meta[r["run_id"]]
            n_done += 1
            if r["ok"]:
                flat = {k: (np.nan if v is None else v) for k, v in r["metrics"].items() if not isinstance(v, dict)}
                rows.append({"run_id": r["run_id"], "sweep": p["sweep"], "level": p["level"], "label": p["label"], **_spec_cols(p["spec"]), **flat,
                             "harvested_by_year": json.dumps(r["metrics"].get("harvested_by_year", {})), "te_by_year": json.dumps(r["metrics"].get("te_by_year", {})),
                             "seconds": r["seconds"], "warnings": "; ".join(r.get("warnings", []))})
                monthly_frames.append(pd.DataFrame(r["monthly"]))
            else:
                rows.append({"run_id": r["run_id"], "sweep": p["sweep"], "level": p["level"], "label": p["label"], **_spec_cols(p["spec"]), "error": r["error"], "seconds": r["seconds"]})
                log.warning("run %s failed: %s", r["run_id"], r["error"])
            if n_done % 5 == 0 or n_done == len(todo):
                el = time.time() - t0
                say(f"{n_done}/{len(todo)} runs · {el / 60:.1f} min elapsed · ~{el / n_done * (len(todo) - n_done) / 60:.1f} min left")
                _write(out, rows, monthly_frames)
            if cancel is not None and cancel():
                say("cancelled; partial results saved (re-run to resume)")
                for f in futs:
                    f.cancel()
                break
    _write(out, rows, monthly_frames)
    return out


def _write(out: Path, rows: list[dict], monthly_frames: list[pd.DataFrame]) -> None:
    if rows:
        pd.DataFrame(rows).to_parquet(out / "results.parquet", index=False)
    if monthly_frames:
        m = pd.concat(monthly_frames, ignore_index=True)
        m = m.drop_duplicates(subset=["run_id", "date"], keep="last")
        m.to_parquet(out / "monthly.parquet", index=False)


def _spec_cols(spec: dict) -> dict:
    keep = ("horizon_years", "start_year", "account_size", "basket_size", "trigger", "approach", "concentrated_pct", "concentrated_gain", "te_limit",
            "sector_band", "factor_alignment", "whole_shares")
    return {k: spec[k] for k in keep if k in spec}


def _run_id(r: dict) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(r["spec"].to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)[:60] or "study"


# ---------------------------------------------------------------------- results
def load_results(study_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    study_dir = Path(study_dir)
    res = pd.read_parquet(study_dir / "results.parquet") if (study_dir / "results.parquet").exists() else pd.DataFrame()
    mon = pd.read_parquet(study_dir / "monthly.parquet") if (study_dir / "monthly.parquet").exists() else pd.DataFrame()
    return res, mon


def summarise(results: pd.DataFrame, sweep: str, metrics: list[str] | None = None) -> pd.DataFrame:
    """Median and interquartile range across windows for each level of one sweep (plus the base case)."""
    metrics = metrics or SUMMARY_METRICS
    if results.empty:
        return pd.DataFrame()
    ok = results[results.get("error", pd.Series(index=results.index, dtype=object)).isna()] if "error" in results else results
    df = ok[ok["sweep"].isin([sweep, "base"])] if sweep != "base" else ok[ok["sweep"] == "base"]
    if df.empty:
        return pd.DataFrame()
    col = {"account_size": "account_size", "basket_size": "basket_size", "trigger": "trigger", "approach": "approach",
           "concentrated": "level", "horizon": "horizon_years"}.get(sweep, "level")
    if sweep == "base":
        col = "label"
    keep = [m for m in metrics if m in df.columns]
    num = df[[col] + keep].copy()
    for c in keep:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    g = num.groupby(col)
    med = g[keep].median()
    q1 = g[keep].quantile(0.25)
    q3 = g[keep].quantile(0.75)
    out = med.copy()
    for c in keep:
        out[f"{c}_iqr"] = q3[c] - q1[c]
    out["windows"] = g.size()
    if col != "label":
        out = out.sort_index()
    return out.reset_index().rename(columns={col: "level"})


def concentrated_grid(results: pd.DataFrame, metric: str = "conc_months_to_diversify") -> pd.DataFrame:
    """Median metric over windows on the (position size x embedded gain) grid."""
    if results.empty or "concentrated_pct" not in results:
        return pd.DataFrame()
    df = results[(results["sweep"] == "concentrated") & results.get("error", pd.Series(index=results.index)).isna()] if "error" in results else results[results["sweep"] == "concentrated"]
    if df.empty or metric not in df:
        return pd.DataFrame()
    return df.pivot_table(index="concentrated_pct", columns="concentrated_gain", values=metric, aggfunc="median")


def harvest_curves(monthly: pd.DataFrame, results: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """Median cumulative harvested (% of start value) by month-since-start, one column per level of the sweep."""
    if monthly.empty or results.empty:
        return pd.DataFrame()
    df = results[results["sweep"].isin([sweep, "base"])][["run_id", "level", "account_size", "sweep", "label", "approach", "basket_size", "trigger"]]
    m = monthly.merge(df, on="run_id")
    if m.empty:
        return pd.DataFrame()
    m = m.sort_values(["run_id", "date"])
    m["month"] = m.groupby("run_id").cumcount()
    m["cum_pct"] = m.groupby("run_id")["harvested"].cumsum() / m["account_size"]
    key = {"account_size": "account_size", "basket_size": "basket_size", "trigger": "trigger", "approach": "approach"}.get(sweep, "level")
    if sweep == "base":
        key = "label"
    return m.pivot_table(index="month", columns=key, values="cum_pct", aggfunc="median")


def yearly_table(results: pd.DataFrame, field: str = "harvested_by_year") -> pd.DataFrame:
    """Base-case harvested (or TE) by calendar year across all windows: which years harvest, which do not."""
    if results.empty or field not in results:
        return pd.DataFrame()
    base = results[results["sweep"] == "base"]
    rows = {}
    for _, r in base.iterrows():
        try:
            d = json.loads(r[field]) if isinstance(r[field], str) else (r[field] or {})
        except Exception:
            continue
        for y, v in d.items():
            rows.setdefault(int(y), []).append(float(v) / (r["account_size"] if field == "harvested_by_year" else 1.0))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame({"year": sorted(rows), "median": [float(np.median(rows[y])) for y in sorted(rows)],
                         "min": [float(np.min(rows[y])) for y in sorted(rows)], "max": [float(np.max(rows[y])) for y in sorted(rows)], "windows": [len(rows[y]) for y in sorted(rows)]})
