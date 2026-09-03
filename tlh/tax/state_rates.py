"""State capital-gains tax treatment for all 50 states + DC (data in state_rates.yaml).

Every number here is an approximate 2026 planning figure and is labelled as such in the GUI and exports. The
module answers two questions the engine needs everywhere:

* `state_rates(state, income)` -> the state's marginal rate on short-term and on long-term gains for a taxpayer
  with `income` of other taxable income (progressive brackets, exclusions, flat capital-gains rates, surtaxes).
* `combined_marginal(state, filing, income)` -> federal + NIIT + state marginal rates on ST and LT gains, which is
  what a harvested loss is worth and what a realised gain costs.

`table()` returns the whole country as a DataFrame for the Tax rates screen and Excel export.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from ..config import RESOURCE_ROOT

YAML_PATH = Path(__file__).with_name("state_rates.yaml")
if not YAML_PATH.exists():                      # frozen build: resources live under sys._MEIPASS
    YAML_PATH = RESOURCE_ROOT / "tlh" / "tax" / "state_rates.yaml"

# Federal figures used for the combined view (approximate 2026; the concentration module holds the full schedules).
FED_ST_TOP = 0.37
FED_LT_TOP = 0.20
NIIT = 0.038


@dataclass
class StateTax:
    abbrev: str
    name: str
    treatment: str                                   # ordinary | none | exclusion | flat_cg | excise
    brackets: list[tuple[float, float]] = field(default_factory=list)   # (threshold, marginal rate), ascending
    lt_rate: float | None = None                     # flat_cg / excise
    st_rate: float | None = None                     # flat_cg override for short-term
    exclusion_pct: float = 0.0
    applies_to: str = "lt"                           # exclusion applies to lt | both
    exclusion_cap: float | None = None
    lt_addon: float = 0.0
    st_addon: float = 0.0
    addon_threshold: float = math.inf
    cg_threshold: float = 0.0                        # excise: gains below this are untaxed
    note: str = ""
    local_note: str = ""

    # ------------------------------------------------------------------ core
    @property
    def top_ordinary(self) -> float:
        return max((r for _, r in self.brackets), default=0.0)

    def marginal_ordinary(self, income: float) -> float:
        rate = 0.0
        for thr, r in self.brackets:
            if income >= thr:
                rate = r
        return rate

    def _addon(self, income: float, which: str) -> float:
        add = self.lt_addon if which == "lt" else self.st_addon
        return add if (add and income >= self.addon_threshold) else 0.0

    def rate_lt(self, income: float = 1e12, gain: float = 0.0) -> float:
        """Marginal state rate on an additional dollar of long-term gain for a taxpayer with `income`."""
        if self.treatment == "none":
            return 0.0
        if self.treatment == "excise":
            if gain and gain < self.cg_threshold:
                return 0.0
            return (self.lt_rate or 0.0) + (self.lt_addon if gain >= self.addon_threshold else 0.0)
        if self.treatment == "flat_cg":
            return (self.lt_rate if self.lt_rate is not None else self.marginal_ordinary(income)) + self._addon(income, "lt")
        base = self.marginal_ordinary(income)
        if self.treatment == "exclusion":
            base *= 1.0 - self.exclusion_pct
        return base + self._addon(income, "lt")

    def rate_st(self, income: float = 1e12) -> float:
        if self.treatment in ("none", "excise"):
            return 0.0
        if self.treatment == "flat_cg":
            return (self.st_rate if self.st_rate is not None else self.marginal_ordinary(income)) + self._addon(income, "st")
        base = self.marginal_ordinary(income)
        if self.treatment == "exclusion" and self.applies_to == "both":
            base *= 1.0 - self.exclusion_pct
        return base + self._addon(income, "st")

    def describe(self) -> str:
        if self.treatment == "none":
            t = "no state tax on capital gains"
        elif self.treatment == "exclusion":
            t = f"{self.exclusion_pct:.0%} of {'all' if self.applies_to == 'both' else 'long-term'} gains excluded"
        elif self.treatment == "flat_cg":
            t = f"long-term gains at a flat {self.lt_rate:.2%}" if self.lt_rate is not None else "flat capital-gains rate"
        elif self.treatment == "excise":
            t = f"{(self.lt_rate or 0.0):.0%} excise on long-term gains above ${self.cg_threshold:,.0f}"
        else:
            t = "gains taxed as ordinary income"
        return t + (f". {self.note}" if self.note else "")


def _parse(d: dict) -> StateTax:
    return StateTax(
        abbrev=d["abbrev"], name=d["name"], treatment=d["treatment"],
        brackets=[(float(t), float(r)) for t, r in d.get("brackets", [[0, 0.0]])],
        lt_rate=d.get("lt_rate"), st_rate=d.get("st_rate"), exclusion_pct=float(d.get("exclusion_pct", 0.0)),
        applies_to=d.get("applies_to", "lt"), exclusion_cap=d.get("exclusion_cap"),
        lt_addon=float(d.get("lt_addon", 0.0)), st_addon=float(d.get("st_addon", 0.0)),
        addon_threshold=float(d.get("addon_threshold", math.inf)), cg_threshold=float(d.get("cg_threshold", 0.0)),
        note=d.get("note", ""), local_note=d.get("local_note", ""),
    )


@lru_cache(maxsize=1)
def _load() -> tuple[int, dict[str, StateTax]]:
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    states = {s["abbrev"]: _parse(s) for s in raw["states"]}
    return int(raw.get("year", 2026)), states


def data_year() -> int:
    return _load()[0]


def all_states() -> dict[str, StateTax]:
    return _load()[1]


def get_state(state: str) -> StateTax:
    states = all_states()
    key = state.strip().upper()
    if key in states:
        return states[key]
    for s in states.values():
        if s.name.lower() == state.strip().lower():
            return s
    raise KeyError(f"unknown state '{state}'")


def state_rates(state: str, income: float = 1e12, gain: float = 0.0) -> dict:
    s = get_state(state)
    return {"state": s.abbrev, "name": s.name, "st_rate": s.rate_st(income), "lt_rate": s.rate_lt(income, gain),
            "ordinary_marginal": s.marginal_ordinary(income), "treatment": s.treatment, "description": s.describe(),
            "local_note": s.local_note, "year": data_year()}


def combined_marginal(state: str, filing_status: str = "single", income: float = 1e12, gain: float = 0.0,
                      fed_st: float | None = None, fed_lt: float | None = None, apply_niit: bool | None = None) -> dict:
    """Federal + NIIT + state marginal rates on short-term and long-term gains for a taxpayer with `income`.

    Federal rates default to the bracket schedules in tax/concentration.py stacked on `income`; pass `fed_st`/`fed_lt`
    to override. NIIT applies above the statutory MAGI threshold for the filing status."""
    from .concentration import DEFAULT_BRACKETS, NIIT_RATE

    sched = DEFAULT_BRACKETS.get(filing_status, DEFAULT_BRACKETS["single"])
    if fed_st is None:
        fed_st = _marginal(sched["ordinary"], income)
    if fed_lt is None:
        fed_lt = _marginal(sched["ltcg"], income)
    niit_on = (income >= sched["niit_threshold"]) if apply_niit is None else apply_niit
    niit = NIIT_RATE if niit_on else 0.0
    st = state_rates(state, income, gain)
    return {"state": st["state"], "income": income, "filing_status": filing_status,
            "fed_st": fed_st, "fed_lt": fed_lt, "niit": niit, "state_st": st["st_rate"], "state_lt": st["lt_rate"],
            "total_st": fed_st + niit + st["st_rate"], "total_lt": fed_lt + niit + st["lt_rate"],
            "description": st["description"], "local_note": st["local_note"], "year": data_year()}


def _marginal(brackets: list[tuple[float, float]], income: float) -> float:
    """`brackets` as (upper_bound, rate) rows (concentration.py convention): the rate of the bracket containing income."""
    for upper, rate in brackets:
        if income < upper:
            return rate
    return brackets[-1][1]


def table(income: float = 1e12) -> pd.DataFrame:
    """One row per state: treatment, ordinary top rate, ST/LT marginal state rates at `income`, combined with federal top rates."""
    rows = []
    for s in all_states().values():
        st, lt = s.rate_st(income), s.rate_lt(income, gain=max(income, 1.0))
        rows.append({"abbrev": s.abbrev, "name": s.name, "treatment": s.treatment, "ordinary_top_rate": s.top_ordinary,
                     "st_rate": st, "lt_rate": lt, "lt_top_rate": s.rate_lt(1e12, 1e12), "st_top_rate": s.rate_st(1e12),
                     "combined_st_top": FED_ST_TOP + NIIT + s.rate_st(1e12), "combined_lt_top": FED_LT_TOP + NIIT + s.rate_lt(1e12, 1e12),
                     "n_brackets": len([b for b in s.brackets if b[1] > 0]) or 0, "flat": len(s.brackets) <= 2 and s.treatment != "exclusion",
                     "description": s.describe(), "local_note": s.local_note})
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


def rank_for_harvesting(income: float = 1e12) -> pd.DataFrame:
    """Where a harvested short-term loss is worth the most: combined ST marginal rate, descending."""
    df = table(income)
    df["tlh_value_per_10k_st_loss"] = df["combined_st_top"] * 10_000
    return df.sort_values("combined_st_top", ascending=False).reset_index(drop=True)
