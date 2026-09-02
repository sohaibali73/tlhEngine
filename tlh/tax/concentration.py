"""Embedded gains & concentrated positions: bracket-aware tax engine, option hedging maths, charitable / gifting /
exchange-fund comparisons.

Tax brackets are editable defaults (2026 approximate, inflation-indexed figures; verify against the IRS tables) and
are always shown to the user next to results. Nothing here is tax advice; the engine quantifies conventions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

# ====================================================================================== brackets
# (upper bound of bracket, rate); the last bracket is open-ended (inf). Approximate 2026 figures; editable in Settings.
DEFAULT_BRACKETS: dict[str, dict] = {
    "single": {
        "ordinary": [(12_400, 0.10), (50_400, 0.12), (105_700, 0.22), (201_775, 0.24), (256_225, 0.32), (640_600, 0.35), (math.inf, 0.37)],
        "ltcg": [(49_450, 0.0), (545_500, 0.15), (math.inf, 0.20)],
        "niit_threshold": 200_000, "std_deduction": 16_100,
    },
    "mfj": {
        "ordinary": [(24_800, 0.10), (100_800, 0.12), (211_400, 0.22), (403_550, 0.24), (512_450, 0.32), (768_700, 0.35), (math.inf, 0.37)],
        "ltcg": [(98_900, 0.0), (613_700, 0.15), (math.inf, 0.20)],
        "niit_threshold": 250_000, "std_deduction": 32_200,
    },
    "mfs": {
        "ordinary": [(12_400, 0.10), (50_400, 0.12), (105_700, 0.22), (201_775, 0.24), (256_225, 0.32), (384_350, 0.35), (math.inf, 0.37)],
        "ltcg": [(49_450, 0.0), (306_850, 0.15), (math.inf, 0.20)],
        "niit_threshold": 125_000, "std_deduction": 16_100,
    },
    "hoh": {
        "ordinary": [(17_700, 0.10), (67_450, 0.12), (105_700, 0.22), (201_775, 0.24), (256_200, 0.32), (640_600, 0.35), (math.inf, 0.37)],
        "ltcg": [(66_200, 0.0), (579_600, 0.15), (math.inf, 0.20)],
        "niit_threshold": 200_000, "std_deduction": 24_150,
    },
}
NIIT_RATE = 0.038


@dataclass
class BracketSchedule:
    filing_status: str = "mfj"
    ordinary: list[tuple[float, float]] = field(default_factory=lambda: list(DEFAULT_BRACKETS["mfj"]["ordinary"]))
    ltcg: list[tuple[float, float]] = field(default_factory=lambda: list(DEFAULT_BRACKETS["mfj"]["ltcg"]))
    niit_threshold: float = 250_000
    niit_rate: float = NIIT_RATE
    state_rate: float = 0.0
    year: int = 2026

    @classmethod
    def default(cls, filing_status: str = "mfj", state_rate: float = 0.0) -> BracketSchedule:
        d = DEFAULT_BRACKETS.get(filing_status, DEFAULT_BRACKETS["mfj"])
        return cls(filing_status, list(d["ordinary"]), list(d["ltcg"]), d["niit_threshold"], NIIT_RATE, state_rate)

    def to_dict(self) -> dict:
        return {"filing_status": self.filing_status, "ordinary": [[u if math.isfinite(u) else None, r] for u, r in self.ordinary],
                "ltcg": [[u if math.isfinite(u) else None, r] for u, r in self.ltcg], "niit_threshold": self.niit_threshold,
                "niit_rate": self.niit_rate, "state_rate": self.state_rate, "year": self.year}

    @classmethod
    def from_dict(cls, d: dict) -> BracketSchedule:
        fix = lambda rows: [(math.inf if u is None else float(u), float(r)) for u, r in rows]  # noqa: E731
        return cls(d.get("filing_status", "mfj"), fix(d["ordinary"]), fix(d["ltcg"]), float(d.get("niit_threshold", 250_000)),
                   float(d.get("niit_rate", NIIT_RATE)), float(d.get("state_rate", 0.0)), int(d.get("year", 2026)))


def _stacked_tax(amount: float, base: float, brackets: list[tuple[float, float]]) -> float:
    """Tax on `amount` stacked on top of `base` income using progressive brackets."""
    if amount <= 0:
        return 0.0
    tax = 0.0
    lo = 0.0
    lower, upper = base, base + amount
    for hi, rate in brackets:
        seg_lo, seg_hi = max(lo, lower), min(hi, upper)
        if seg_hi > seg_lo:
            tax += (seg_hi - seg_lo) * rate
        lo = hi
        if lo >= upper:
            break
    return tax


def ltcg_tax(gain: float, other_taxable_income: float, sched: BracketSchedule, include_state: bool = True) -> dict:
    """Federal LTCG tax stacked on ordinary taxable income, NIIT on the portion of gain above the MAGI threshold, state."""
    fed = _stacked_tax(gain, max(other_taxable_income, 0.0), sched.ltcg)
    headroom = max(sched.niit_threshold - other_taxable_income, 0.0)
    niit = sched.niit_rate * max(gain - headroom, 0.0)
    state = sched.state_rate * gain if include_state else 0.0
    total = fed + niit + state
    marg = marginal_ltcg_rate(gain, other_taxable_income, sched, include_state)
    return {"federal": fed, "niit": niit, "state": state, "total": total, "effective_rate": total / gain if gain > 0 else 0.0,
            "marginal_rate": marg}


def ordinary_tax(gain: float, other_taxable_income: float, sched: BracketSchedule, include_state: bool = True) -> dict:
    fed = _stacked_tax(gain, max(other_taxable_income, 0.0), sched.ordinary)
    headroom = max(sched.niit_threshold - other_taxable_income, 0.0)
    niit = sched.niit_rate * max(gain - headroom, 0.0)
    state = sched.state_rate * gain if include_state else 0.0
    total = fed + niit + state
    return {"federal": fed, "niit": niit, "state": state, "total": total, "effective_rate": total / gain if gain > 0 else 0.0}


def marginal_ltcg_rate(gain: float, other_taxable_income: float, sched: BracketSchedule, include_state: bool = True) -> float:
    top = other_taxable_income + max(gain, 0.0)
    rate = next(r for hi, r in sched.ltcg if top <= hi)
    if top > sched.niit_threshold:
        rate += sched.niit_rate
    return rate + (sched.state_rate if include_state else 0.0)


def convex_pieces(sched: BracketSchedule, other_income: float, kind: str = "ltcg", include_state: bool = True) -> list[tuple[float, float]]:
    """Represent tax(g) = sum_k slope_k * max(0, g - kink_k) (convex piecewise-linear) for use in cvxpy.

    Returns [(kink_k, incremental_slope_k)], kinks in gain space given `other_income` already stacked below."""
    brackets = sched.ltcg if kind == "ltcg" else sched.ordinary
    out = []
    lower = 0.0
    prev_rate = 0.0
    for hi, rate in brackets:
        kink = max(lower - other_income, 0.0)
        if rate - prev_rate != 0:
            out.append((kink, rate - prev_rate))
        prev_rate = rate
        lower = hi
    niit_kink = max(sched.niit_threshold - other_income, 0.0)
    out.append((niit_kink, sched.niit_rate))
    if include_state and sched.state_rate:
        out.append((0.0, sched.state_rate))
    return out


def tax_from_pieces(gain: float, pieces: list[tuple[float, float]]) -> float:
    return float(sum(slope * max(0.0, gain - kink) for kink, slope in pieces))


# ====================================================================================== option maths
def bs_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        return intrinsic
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        return float((S > K) if kind == "call" else -(S < K))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * (norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0)


def zero_cost_collar(S: float, put_strike: float, T: float, r: float, sigma: float, q: float = 0.0) -> dict:
    """Find the call strike whose premium equals the put premium (bisection)."""
    put = bs_price(S, put_strike, T, r, sigma, q, "put")
    lo, hi = S, S * 3.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        c = bs_price(S, mid, T, r, sigma, q, "call")
        if c > put:
            lo = mid
        else:
            hi = mid
    K_call = 0.5 * (lo + hi)
    return {"put_strike": put_strike, "call_strike": K_call, "put_premium": put, "call_premium": bs_price(S, K_call, T, r, sigma, q, "call"),
            "floor_pct": put_strike / S - 1, "cap_pct": K_call / S - 1, "band_pct": (K_call - put_strike) / S}


def collar_analysis(S: float, shares: float, basis_per_share: float, T: float, sigma: float, r: float = 0.04, q: float = 0.0,
                    put_strike_pct: float = 0.90, call_strike_pct: float | None = None, is_long_term: bool = True,
                    ltcg_rate: float = 0.238, ordinary_rate: float = 0.408, constructive_band: float = 0.15) -> dict:
    """Collar economics + tax flags for a concentrated position.

    Flags: constructive sale risk (§1259) when the collar band is narrower than `constructive_band` of spot; straddle
    rules (§1092) when the lot is still short-term (holding period suspended, losses deferred); qualified covered call
    considerations for the short call."""
    K_put = S * put_strike_pct
    if call_strike_pct is None:
        zc = zero_cost_collar(S, K_put, T, r, sigma, q)
        K_call = zc["call_strike"]
    else:
        K_call = S * call_strike_pct
    put_prem = bs_price(S, K_put, T, r, sigma, q, "put")
    call_prem = bs_price(S, K_call, T, r, sigma, q, "call")
    net_cost_ps = put_prem - call_prem
    value = S * shares
    gain = (S - basis_per_share) * shares
    tax_now = gain * (ltcg_rate if is_long_term else ordinary_rate) if gain > 0 else 0.0
    band = (K_call - K_put) / S
    flags = []
    if band < constructive_band:
        flags.append(f"CONSTRUCTIVE SALE RISK (§1259): collar band {band:.0%} of spot is narrower than the {constructive_band:.0%} practitioner threshold; "
                     "the IRS may treat the hedge as a sale of the position.")
    if not is_long_term:
        flags.append("STRADDLE RULES (§1092): lot is short-term; entering an offsetting position suspends the holding period and defers losses on the position.")
    if K_call < S:
        flags.append("Short call is in the money: a non-qualified covered call may terminate the holding period of the underlying.")
    flags.append("Dividends received while hedged with a deep-in-the-money put may lose qualified-dividend treatment (holding-period rules).")
    return {
        "spot": S, "shares": shares, "position_value": value, "embedded_gain": gain, "tax_if_sold_now": tax_now,
        "put_strike": K_put, "call_strike": K_call, "put_premium": put_prem, "call_premium": call_prem, "net_cost_per_share": net_cost_ps,
        "net_cost_total": net_cost_ps * shares, "net_cost_pct": net_cost_ps / S, "annualised_cost_pct": (net_cost_ps / S) / max(T, 1e-9),
        "floor_value": K_put * shares, "cap_value": K_call * shares, "floor_pct": K_put / S - 1, "cap_pct": K_call / S - 1, "band_pct": band,
        "max_loss_vs_now": (K_put - S) * shares - net_cost_ps * shares, "delta_hedged": 1 + bs_delta(S, K_put, T, r, sigma, q, "put") - bs_delta(S, K_call, T, r, sigma, q, "call"),
        "tax_deferred_by_hedging": tax_now, "flags": flags, "sigma": sigma, "T": T,
    }


def payoff_table(S: float, K_put: float, K_call: float, net_cost_ps: float, shares: float, grid: np.ndarray | None = None) -> list[dict]:
    grid = grid if grid is not None else np.linspace(0.5 * S, 1.5 * S, 21)
    rows = []
    for s in grid:
        unhedged = (s - S) * shares
        hedged = (min(max(s, K_put), K_call) - S) * shares - net_cost_ps * shares
        rows.append({"price": float(s), "pct_move": float(s / S - 1), "unhedged_pnl": float(unhedged), "collar_pnl": float(hedged)})
    return rows


# ====================================================================================== charitable / gifting / exchange fund
def charitable_comparison(value: float, basis: float, ltcg_rate: float, ordinary_marginal_rate: float, agi: float | None = None,
                          is_long_term: bool = True) -> dict:
    """Donate appreciated shares vs sell then donate cash vs keep. Deduction of appreciated LT stock is limited to 30% of AGI
    (50%/60% for cash); excess carries forward five years — flagged, not modelled."""
    gain = max(value - basis, 0.0)
    tax_on_sale = gain * ltcg_rate if is_long_term else gain * ordinary_marginal_rate
    donate_shares = {"charity_receives": value, "deduction": value if is_long_term else basis,
                     "tax_saved_by_deduction": (value if is_long_term else basis) * ordinary_marginal_rate, "cap_gains_tax_paid": 0.0}
    donate_shares["net_cost_to_donor"] = value - donate_shares["tax_saved_by_deduction"]
    sell_then_donate = {"charity_receives": value - tax_on_sale, "deduction": value - tax_on_sale,
                        "tax_saved_by_deduction": (value - tax_on_sale) * ordinary_marginal_rate, "cap_gains_tax_paid": tax_on_sale}
    sell_then_donate["net_cost_to_donor"] = value - sell_then_donate["tax_saved_by_deduction"]
    flags = []
    if agi and is_long_term and value > 0.30 * agi:
        flags.append(f"Deduction for appreciated stock is limited to 30% of AGI (${0.30 * agi:,.0f}); excess carries forward up to 5 years.")
    if not is_long_term:
        flags.append("Short-term appreciated stock: deduction is limited to basis; wait for long-term status before donating.")
    return {"donate_shares": donate_shares, "sell_then_donate": sell_then_donate,
            "advantage_of_donating_shares": sell_then_donate["net_cost_to_donor"] - donate_shares["net_cost_to_donor"],
            "extra_to_charity": donate_shares["charity_receives"] - sell_then_donate["charity_receives"], "flags": flags}


def gift_to_lower_bracket(gain: float, donor_rate: float, donee_rate: float, annual_exclusion: float = 19_000) -> dict:
    return {"tax_saved": gain * max(donor_rate - donee_rate, 0.0), "donor_rate": donor_rate, "donee_rate": donee_rate,
            "notes": [f"Carryover basis: the donee inherits the donor's basis and holding period; annual gift-tax exclusion ${annual_exclusion:,.0f} per donee.",
                      "Kiddie tax: unearned income of a dependent child above the threshold is taxed at the parents' rate.",
                      "Donee must actually be in the lower bracket after the gain is stacked on their income."]}


def exchange_fund_breakeven(value: float, basis: float, ltcg_rate: float, fee_rate: float = 0.01, lockup_years: int = 7,
                            discount_rate: float = 0.04, expected_return: float = 0.06) -> dict:
    """Section 721 exchange fund: contribute shares, receive a diversified partnership interest, no gain recognised; basis
    carries over; 7-year lockup; fees. Compare PV of fees + illiquidity vs PV of tax deferred."""
    gain = max(value - basis, 0.0)
    tax_now = gain * ltcg_rate
    pv_fees = sum(value * ((1 + expected_return) ** t) * fee_rate / (1 + discount_rate) ** t for t in range(1, lockup_years + 1))
    deferral_value = tax_now - tax_now / (1 + discount_rate) ** lockup_years
    return {"tax_deferred": tax_now, "pv_of_fees": pv_fees, "pv_value_of_deferral": deferral_value,
            "net_benefit_vs_selling": deferral_value - pv_fees, "lockup_years": lockup_years,
            "notes": ["Requires ~20% of fund assets in qualifying assets (real estate) to avoid investment-company treatment.",
                      "Basis carries over; gain is realised when the diversified interest is eventually sold.",
                      "Accredited/qualified purchaser minimums typically apply."]}


def stepup_value(gain: float, ltcg_rate: float, horizon_years: float, discount_rate: float, p_stepup: float) -> dict:
    """Expected present value of the embedded tax liability given a probability of a basis step-up at death within the horizon."""
    tax = gain * ltcg_rate
    pv_if_sold_at_horizon = tax / (1 + discount_rate) ** horizon_years
    return {"tax_liability": tax, "pv_if_deferred_to_horizon": pv_if_sold_at_horizon, "expected_pv_with_stepup": (1 - p_stepup) * pv_if_sold_at_horizon,
            "value_of_stepup_option": p_stepup * pv_if_sold_at_horizon}


# ====================================================================================== concentration statistics
def concentration_stats(weights: np.ndarray) -> dict:
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    w = w / w.sum() if w.sum() else w
    hhi = float((w ** 2).sum()) if len(w) else 0.0
    return {"hhi": hhi, "effective_n": 1.0 / hhi if hhi > 0 else 0.0, "top1": float(w.max()) if len(w) else 0.0,
            "top5": float(np.sort(w)[::-1][:5].sum()) if len(w) else 0.0, "n_positions": int(len(w))}
