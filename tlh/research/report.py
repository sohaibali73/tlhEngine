"""Due-diligence write-up of a study: what was tested, how, and what the numbers say, in plain language with the tables.
Markdown (for the screen and YANG) and an Excel workbook (one sheet per sweep) for the reviewer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .grid import SUMMARY_METRICS, concentrated_grid, summarise, yearly_table
from .spec import APPROACHES, StudySpec

METHOD = """
**Universe.** Point-in-time S&P 500: every name that was a member on the day (Norgate "S&P 500 Current & Past", delisted
names included, sold at their last close when they disappear). Benchmark return is the S&P 500 total-return index.
Capitalisation weights for construction are a proxy (today's shares scaled by the split factor); sectors are today's GICS.

**Testing method.** Rolling windows starting on the first trading day of every calendar year from {first} (every
{every} year(s)), each {horizons} years long, reviewed monthly at month-end. Reported numbers are medians across the
windows with the interquartile range, so no single decade drives a conclusion.

**Account mechanics.** Whole shares, {cost} bps per trade, cash from dividends and sales reinvested at the next review.
A lot is harvested when its loss clears the trigger ({trigger_basis} basis) and is at least ${min_harvest:,.0f}; a
name bought inside the {wash}-day window is not sold at a loss, and a name sold at a loss is not bought back inside
the window, so no harvested loss is disallowed as a wash sale. A concentrated starting position is sold only as far
as realised losses cover its gain (plus any gain budget), so the unwind is tax-neutral by construction.

**Construction.** Basket of {basket} names, tracking-error target {te:.1%}, sector band {sector:.0%}, factor alignment
{factors} (size, momentum, volatility, beta within ±{fband:.2f} z of the index). Risk model for TE forecasts and the
optimizer: the calibration study's winner, a {cov}-day equal-weighted Ledoit-Wolf covariance.

**Approaches.**
{approaches}

**Metrics.** *Losses harvested* (per year as % of starting value, split short/long term, and their tax value at
{st:.1%}/{lt:.1%}); *harvest life* (months until the trailing-12-month harvest yield falls below {oss:.1%} of value for
good, plus the half-life of cumulative harvesting); *tracking error* (realised daily active return vs the index,
annualised, and the ex-ante forecast); turnover, trades, wash-window blocks, names held, ending embedded gain.
"""


def method_text(study: StudySpec) -> str:
    b = study.base
    appr = "\n".join(f"- `{k}`: {v}" for k, v in APPROACHES.items() if k in study.approaches or k == b.approach)
    return METHOD.format(first=study.first_start_year, every=study.every_n_years, horizons="/".join(str(h) for h in study.horizons), cost=b.cost_bps,
                         trigger_basis=b.trigger_basis, min_harvest=b.min_harvest, wash=b.wash_days, basket=b.basket_size, te=b.te_limit, sector=b.sector_band,
                         factors="on" if b.factor_alignment else "off", fband=b.factor_band, cov=b.cov_lookback, approaches=appr, st=b.st_rate, lt=b.lt_rate,
                         oss=b.ossification_yield)


def label_levels(df: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """Human labels for the sweep level column (0.0025 -> 0.25%, 500000 -> $500,000)."""
    if df is None or df.empty or "level" not in df:
        return df
    d = df.copy()
    if sweep == "trigger":
        d["level"] = d["level"].map(lambda v: f"{float(v):.2%}" if isinstance(v, (int, float)) else v)
    elif sweep == "account_size":
        d["level"] = d["level"].map(lambda v: f"${float(v):,.0f}" if isinstance(v, (int, float)) else v)
    elif sweep == "basket_size":
        d["level"] = d["level"].map(lambda v: f"{int(v)} names" if isinstance(v, (int, float)) else v)
    else:
        d["level"] = d["level"].astype(str)
    return d


def _fmt_table(df: pd.DataFrame, pct_cols=(), int_cols=()) -> str:
    if df is None or df.empty:
        return "_no results yet_"
    d = df.copy()
    for c in d.columns:
        if c == "level":
            continue
        if c in pct_cols or "_pct" in c or c.startswith("te_") or c in ("excess_return_annual", "turnover_annual"):
            d[c] = d[c].map(lambda v: f"{v:.2%}" if pd.notna(v) and isinstance(v, (int, float)) else v)
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:,.1f}" if pd.notna(v) else "")
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def findings(results: pd.DataFrame) -> list[str]:
    """Plain-language findings a reviewer can check against the tables."""
    out: list[str] = []
    if results.empty:
        return out
    ok = results[results["error"].isna()] if "error" in results else results
    base = ok[ok["sweep"] == "base"]
    if len(base):
        out.append(f"Base case over {len(base)} windows: median harvest {base['harvested_per_year_pct'].median():.2%} of starting value per year, "
                   f"median realised tracking error {base['te_realised'].median():.2%}, harvest life {base['harvest_life_months'].median():.0f} months "
                   f"(half of all losses inside {base['harvest_half_life_months'].median():.0f} months).")
    for sw, _col, fmt in (("trigger", "trigger", lambda v: f"{v:.2%}"), ("basket_size", "basket_size", lambda v: f"{int(v)} names"),
                         ("account_size", "account_size", lambda v: f"${v:,.0f}"), ("approach", "approach", str)):
        s = summarise(ok, sw)
        if s.empty or len(s) < 2:
            continue
        best = s.loc[s["harvested_per_year_pct"].idxmax()]
        tight = s.loc[s["te_realised"].idxmin()]
        out.append(f"{sw.replace('_', ' ').capitalize()}: most harvest at {fmt(best['level'])} ({best['harvested_per_year_pct']:.2%}/yr, TE {best['te_realised']:.2%}); "
                   f"tightest tracking at {fmt(tight['level'])} (TE {tight['te_realised']:.2%}, harvest {tight['harvested_per_year_pct']:.2%}/yr).")
    cg = concentrated_grid(ok)
    if not cg.empty:
        out.append(f"Concentrated starts: a {cg.index.min():.0%} position with a {cg.columns.min():.0%} gain diversifies in a median {cg.iloc[0, 0]:.0f} months; "
                   f"a {cg.index.max():.0%} position with a {cg.columns.max():.0%} gain takes {cg.iloc[-1, -1]:.0f} months (blank = not inside the window).")
    small = ok[(ok["sweep"] == "account_size")]
    if len(small):
        s = small.groupby("account_size")["names_avg"].median()
        under = s[s < 0.8 * ok["basket_size"].median()]
        if len(under):
            out.append(f"Whole-share rounding: accounts of {', '.join(f'${a:,.0f}' for a in under.index)} cannot hold the target basket "
                       f"(median {under.min():.0f} names); their tracking error and harvest reflect that.")
    return out


def markdown_report(study: StudySpec, results: pd.DataFrame, monthly: pd.DataFrame | None = None) -> str:
    parts = [f"# TLH research study: {study.name}", method_text(study), "## Findings"]
    f = findings(results)
    parts += [f"- {x}" for x in f] or ["_run the study to populate_"]
    for sw in ["base"] + [s for s in study.sweeps if s != "concentrated"]:
        parts.append(f"## {sw.replace('_', ' ').capitalize()}")
        parts.append(_fmt_table(label_levels(summarise(results, sw, ["harvested_per_year_pct", "tax_value_pct_of_start", "harvest_life_months", "harvest_half_life_months", "te_realised", "turnover_annual", "names_avg", "wash_blocked"]), sw)))
    if "concentrated" in study.sweeps:
        parts.append("## Concentrated start: months to diversify (median)")
        cg = concentrated_grid(results)
        if not cg.empty:
            cg = cg.copy()
            cg.index = [f"{i:.0%}" for i in cg.index]
            cg.columns = [f"gain {c:.0%}" for c in cg.columns]
            parts.append(_fmt_table(cg.reset_index().rename(columns={"index": "position"})))
        parts.append("## Concentrated start: harvest per year (median)")
        cg2 = concentrated_grid(results, "harvested_per_year_pct")
        if not cg2.empty:
            cg2 = cg2.copy()
            cg2.index = [f"{i:.0%}" for i in cg2.index]
            cg2.columns = [f"gain {c:.0%}" for c in cg2.columns]
            parts.append(_fmt_table(cg2.reset_index().rename(columns={"index": "position"})))
    yt = yearly_table(results)
    if not yt.empty:
        parts.append("## Harvest by calendar year (base case, % of start value)")
        parts.append(_fmt_table(yt.assign(median=yt["median"], min=yt["min"], max=yt["max"]).rename(columns={"median": "median_pct", "min": "min_pct", "max": "max_pct"})))
    parts.append("## Caveats\n- Capitalisation weights are a proxy (today's shares x split factor); the benchmark return is the real S&P 500 TR index.\n"
                 "- Sectors are today's GICS. Fundamentals are not point-in-time in Norgate, so the twin pairing uses price-based descriptors.\n"
                 "- Trading at month-end closes with a flat cost per trade; no market impact. Dividends reinvested at the next review.\n"
                 "- Nothing here is a forecast; it is the historical distribution of outcomes for the rule as specified.")
    return "\n\n".join(parts)


def excel_report(study: StudySpec, results: pd.DataFrame, monthly: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        pd.DataFrame({"item": ["study", "generated", "runs", "method"], "value": [study.name, pd.Timestamp.now().isoformat(timespec="seconds"), len(results), method_text(study)]}).to_excel(xw, sheet_name="Method", index=False)
        pd.DataFrame({"finding": findings(results)}).to_excel(xw, sheet_name="Findings", index=False)
        for sw in ["base"] + list(study.sweeps):
            if sw == "concentrated":
                for metric in ("conc_months_to_diversify", "harvested_per_year_pct", "te_realised"):
                    cg = concentrated_grid(results, metric)
                    if not cg.empty:
                        cg.to_excel(xw, sheet_name=f"Conc {metric[:20]}")
                continue
            s = label_levels(summarise(results, sw, SUMMARY_METRICS), sw)
            if not s.empty:
                s.to_excel(xw, sheet_name=sw[:31], index=False)
        yt = yearly_table(results)
        if not yt.empty:
            yt.to_excel(xw, sheet_name="By year", index=False)
        cols = [c for c in results.columns if c not in ("harvested_by_year", "te_by_year")]
        results[cols].to_excel(xw, sheet_name="All runs", index=False)
        if monthly is not None and not monthly.empty and len(monthly) < 900_000:
            monthly.to_excel(xw, sheet_name="Monthly", index=False)
    return path


def study_from_json(path: Path) -> StudySpec:
    from .spec import ResearchSpec
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    base = ResearchSpec(**d.pop("base"))
    return StudySpec(base=base, **d)
