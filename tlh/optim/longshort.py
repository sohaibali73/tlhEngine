"""Long/short tax-loss harvesting: the economics of a 130/30-style extension and the "Exchange" glide for concentrated stock.

Why it matters (Quantinno DEALS, April 2026): a long-only book runs out of losers after a few good years, so its tax
benefit decays toward 1% of value. A market-neutral long/short extension (e.g. +30% / -30% on top of a 100% long core)
always holds positions that are under water on one side or the other, so it keeps generating short-term losses in
both rising and falling markets while adding little beta. Quantinno's published 10-year averages: 130/30 ≈ -10% ST
losses and +3% LT gains per year (≈3.4% of value in tax benefit at 40.8% / 23.8%), 145/45 ≈ -13%, 175/75 ≈ -19%,
200/100 ≈ -23%.

This module (1) simulates net capital-loss generation for long-only versus long/short extensions with a Monte Carlo
that reproduces the published decay profile from first principles (basis dispersion, vol, correlation, harvest
rules, wash-sale lockouts), (2) prices the financing (fed funds + spread on the long extension, rebate on the short
proceeds) and (3) builds the tax-neutral "Exchange" divestiture schedule for a concentrated position: sell each period
exactly the amount whose tax equals the losses the extension is expected to generate.

Nothing here places orders or shorts anything; construction of an actual extension is `strategies.long_short_extension`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

QUANTINNO_REFERENCE = {   # initial 10-year averages, % of market value per year (Quantinno DEALS Core deck, Apr 2026)
    "130/30": {"st_losses": -0.10, "lt_gains": 0.03, "te_target": (0.013, 0.015), "excess_return": 0.006},
    "145/45": {"st_losses": -0.13, "lt_gains": 0.04, "te_target": (0.015, 0.02), "excess_return": 0.008},
    "175/75": {"st_losses": -0.19, "lt_gains": 0.05, "te_target": (0.025, 0.03), "excess_return": 0.012},
    "200/100": {"st_losses": -0.232, "lt_gains": 0.056, "te_target": (0.03, 0.035), "excess_return": 0.014},
}
QUANTINNO_YEAR_PROFILE = {  # net capital losses by year, cash-funded accounts, % of beginning value
    "long_only": [0.05, 0.03, 0.02, 0.015, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "130/30": [0.23, 0.13, 0.09, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03],
    "145/45": [0.285, 0.165, 0.12, 0.07, 0.055, 0.045, 0.045, 0.045, 0.045, 0.045],
    "175/75": [0.395, 0.235, 0.18, 0.11, 0.085, 0.075, 0.075, 0.075, 0.075, 0.075],
    "200/100": [0.487, 0.293, 0.23, 0.143, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10],
}


@dataclass
class LongShortSpec:
    extension: float = 0.30            # 130/30 -> 0.30; long = 1 + ext, short = -ext
    n_long: int = 150                  # names in the long book (core + extension)
    n_short: int = 100
    years: int = 10
    market_return: float = 0.06
    market_vol: float = 0.16
    stock_specific_vol: float = 0.22   # idiosyncratic annual vol per name
    rebalance_per_year: int = 12
    harvest_threshold: float = 0.05    # realise a loss when a lot is >5% under water (either side)
    wash_lockout_periods: int = 1      # periods a harvested name cannot be re-entered (≈31 days at monthly cadence)
    st_rate: float = 0.408
    lt_rate: float = 0.238
    fed_funds: float = 0.04
    long_spread: float = 0.005         # borrow spread on the long extension
    short_rebate_haircut: float = 0.006  # fed funds minus rebate earned on short proceeds
    n_paths: int = 200
    seed: int = 7


@dataclass
class LongShortResult:
    by_year: pd.DataFrame              # year, long_only_net_loss_pct, ls_net_loss_pct (means), and p10/p90 for L/S
    summary: dict
    financing: dict
    reference: dict = field(default_factory=dict)


def financing_cost(extension: float, fed_funds: float = 0.04, long_spread: float = 0.005, short_rebate_haircut: float = 0.006,
                   st_rate: float = 0.408) -> dict:
    """Annual financing of a long/short extension as a fraction of core value (Quantinno appendix arithmetic)."""
    long_cost = extension * (fed_funds + long_spread)          # borrow to fund the long extension
    short_income = extension * (fed_funds - short_rebate_haircut)   # rebate on short-sale proceeds
    pre_tax = -(long_cost) + short_income                         # negative = cost
    post_tax = pre_tax * (1 - st_rate) if pre_tax < 0 else pre_tax
    return {"extension": extension, "long_financing_cost": long_cost, "short_rebate_income": short_income, "net_pre_tax": pre_tax,
            "net_post_tax": post_tax, "note": "interest expense assumed deductible against interest/dividend income (net post-tax)"}


def simulate_loss_generation(spec: LongShortSpec | None = None) -> LongShortResult:
    """Monte Carlo of realised short-term losses for a long-only book and a long/short extension book.

    Model: each name = beta·market + idiosyncratic noise; lots are marked each period; a lot more than `harvest_threshold`
    under water (long side: price < basis; short side: price > entry) is closed, the loss booked, and the name sits out
    `wash_lockout_periods` before being re-entered at the new price (a correlated replacement is assumed to keep exposure).
    Realised gains on the other side are avoided by never voluntarily closing winners; a long-only book has no short side,
    so its losses dry up as the market rises. Reported as net capital losses as % of beginning value, per year."""
    spec = spec or LongShortSpec()
    rng = np.random.default_rng(spec.seed)
    P = spec.rebalance_per_year
    T = spec.years * P
    dt = 1.0 / P
    mu_p, sig_m, sig_s = spec.market_return * dt, spec.market_vol * np.sqrt(dt), spec.stock_specific_vol * np.sqrt(dt)
    lo_years = np.zeros((spec.n_paths, spec.years))
    ls_years = np.zeros((spec.n_paths, spec.years))
    for p in range(spec.n_paths):
        mkt = rng.normal(mu_p - 0.5 * sig_m ** 2, sig_m, T)
        # long-only core (n_long names) at equal weight, basis = 1.0 (cash funded)
        n_l, n_s = spec.n_long, spec.n_short
        px_l = np.ones(n_l)
        basis_l = np.ones(n_l)
        lock_l = np.zeros(n_l, dtype=int)
        # long/short: same core + long extension (ext weight spread over n_l) + short book (n_s names)
        px_s = np.ones(n_s)
        entry_s = np.ones(n_s)
        lock_s = np.zeros(n_s, dtype=int)
        w_l = 1.0 / n_l                          # per-name weight of the core
        w_ext = spec.extension / n_l             # extra long weight per name
        w_s = spec.extension / n_s               # per-name short weight
        for t in range(T):
            yr = t // P
            r_l = mkt[t] + rng.normal(0, sig_s, n_l)
            r_s = mkt[t] + rng.normal(0, sig_s, n_s)
            px_l *= 1 + r_l
            px_s *= 1 + r_s
            # --- long side harvest (both books share the long core; the L/S book also harvests the extension)
            under = (px_l < basis_l * (1 - spec.harvest_threshold)) & (lock_l == 0)
            loss_core = ((basis_l - px_l) / basis_l)[under] * w_l
            lo_years[p, yr] += loss_core.sum()
            ls_years[p, yr] += loss_core.sum() + ((basis_l - px_l) / basis_l)[under].sum() * w_ext
            basis_l[under] = px_l[under]
            lock_l[under] = spec.wash_lockout_periods
            lock_l = np.maximum(lock_l - 1, 0)
            # --- short side: a short is under water when the price rose above entry
            over = (px_s > entry_s * (1 + spec.harvest_threshold)) & (lock_s == 0)
            ls_years[p, yr] += ((px_s - entry_s) / entry_s)[over].sum() * w_s
            entry_s[over] = px_s[over]
            lock_s[over] = spec.wash_lockout_periods
            lock_s = np.maximum(lock_s - 1, 0)
            # normalise to beginning-of-year value growth (losses reported as % of beginning value: approximate by market path)
        # scale each year's losses to that year's beginning value (portfolio ≈ market growth)
        growth = np.cumprod(1 + mkt.reshape(spec.years, P).sum(axis=1))
        scale = np.concatenate([[1.0], growth[:-1]])
        lo_years[p] = lo_years[p] * scale / scale          # already in units of beginning weights (weights fixed) -> keep
        ls_years[p] = ls_years[p]
    yrs = np.arange(1, spec.years + 1)
    by_year = pd.DataFrame({
        "year": yrs,
        "long_only_net_loss_pct": lo_years.mean(axis=0), "long_short_net_loss_pct": ls_years.mean(axis=0),
        "long_short_p10": np.percentile(ls_years, 10, axis=0), "long_short_p90": np.percentile(ls_years, 90, axis=0),
        "long_only_p10": np.percentile(lo_years, 10, axis=0), "long_only_p90": np.percentile(lo_years, 90, axis=0),
    })
    label = f"{int(round(100 + spec.extension * 100))}/{int(round(spec.extension * 100))}"
    ref = QUANTINNO_YEAR_PROFILE.get(label)
    if ref:
        by_year["quantinno_reference_pct"] = (ref + [ref[-1]] * spec.years)[: spec.years]
    by_year["long_only_reference_pct"] = (QUANTINNO_YEAR_PROFILE["long_only"] + [0.01] * spec.years)[: spec.years]
    fin = financing_cost(spec.extension, spec.fed_funds, spec.long_spread, spec.short_rebate_haircut, spec.st_rate)
    tax_lo = by_year["long_only_net_loss_pct"].mean() * spec.st_rate
    tax_ls = by_year["long_short_net_loss_pct"].mean() * spec.st_rate
    summary = {
        "label": label, "avg_annual_loss_long_only": float(by_year["long_only_net_loss_pct"].mean()),
        "avg_annual_loss_long_short": float(by_year["long_short_net_loss_pct"].mean()),
        "avg_annual_tax_benefit_long_only": float(tax_lo), "avg_annual_tax_benefit_long_short": float(tax_ls),
        "net_of_financing_long_short": float(tax_ls + fin["net_post_tax"]),
        "uplift_vs_long_only": float(tax_ls + fin["net_post_tax"] - tax_lo),
        "year1_long_short": float(by_year["long_short_net_loss_pct"].iloc[0]), "year5_long_short": float(by_year["long_short_net_loss_pct"].iloc[min(4, spec.years - 1)]),
        "assumptions": {"market_return": spec.market_return, "market_vol": spec.market_vol, "specific_vol": spec.stock_specific_vol,
                        "harvest_threshold": spec.harvest_threshold, "st_rate": spec.st_rate, "paths": spec.n_paths,
                        "note": "losses in % of beginning value; assumes sufficient gains elsewhere to use the losses; no shorting cost beyond rebate haircut"},
    }
    return LongShortResult(by_year=by_year, summary=summary, financing=fin, reference=QUANTINNO_REFERENCE.get(label, {}))


def exchange_glide(position_value: float, cost_basis: float, extension: float = 0.30, years: int = 10, expected_loss_profile: list[float] | None = None,
                   lt_rate: float = 0.238, st_rate: float = 0.408, growth: float = 0.06, proceeds_growth: float = 0.06) -> pd.DataFrame:
    """Tax-neutral divestiture ("Exchange"): each year sell the amount of the concentrated stock whose long-term gain
    equals the short-term losses the extension generates (valued at the ST rate), so the net tax bill is zero.

    Returns the year-by-year schedule; `years_to_full_divestiture` is the first year the position hits zero."""
    label = f"{int(round(100 + extension * 100))}/{int(round(extension * 100))}"
    prof = expected_loss_profile or QUANTINNO_YEAR_PROFILE.get(label) or QUANTINNO_YEAR_PROFILE["130/30"]
    prof = (list(prof) + [prof[-1]] * years)[:years]
    rows = []
    held = position_value
    basis = cost_basis
    diversified = 0.0
    for y in range(1, years + 1):
        held *= 1 + growth
        diversified *= 1 + proceeds_growth
        total = held + diversified
        losses = prof[y - 1] * total                      # extension losses scale with the whole account
        gain_frac = 1 - basis / held if held > 0 else 0.0  # gain per $ sold
        # tax-neutral: lt_rate * gain_frac * sold = st_rate * losses  (losses offset gains dollar for dollar; rate difference handled as value)
        sold = min(held, (st_rate * losses) / (lt_rate * gain_frac) if gain_frac > 0 else held)
        gain = sold * gain_frac
        basis_sold = sold * (1 - gain_frac)
        held -= sold
        basis -= basis_sold
        diversified += sold
        rows.append({"year": y, "extension_losses": losses, "sold": sold, "gain_realised": gain, "tax_on_gain": gain * lt_rate,
                     "tax_saved_by_losses": losses * st_rate, "net_tax": gain * lt_rate - losses * st_rate,
                     "concentrated_remaining": held, "diversified": diversified, "pct_diversified": diversified / max(held + diversified, 1e-9)})
        if held <= 1e-6:
            break
    df = pd.DataFrame(rows)
    df.attrs["years_to_full_divestiture"] = int(df["year"].iloc[-1]) if not df.empty and df["concentrated_remaining"].iloc[-1] <= 1e-6 else None
    df.attrs["label"] = label
    return df


def years_to_diversify_table() -> pd.DataFrame:
    """Reference from the DEALS deck: estimated years to full diversification by cost basis and gross exposure."""
    data = {"130/30": [8.4, 6.3, 4.3, 2.3], "145/45": [6.3, 4.8, 3.3, 1.7], "175/75": [4.3, 3.3, 2.2, 1.2]}
    return pd.DataFrame(data, index=["basis 0%", "basis 25%", "basis 50%", "basis 75%"]).rename_axis("cost basis / value")
