"""Index-futures beta overlay for tax-loss-harvesting sleeves (TradeStation memo, Aug 2026).

During a harvest the sleeve is briefly under-invested (cash from sells, imperfect replacements) or carries unwanted
market exposure. A small listed index-futures position restores the target beta with a known cost of carry instead
of an idiosyncratic residual, and a short overlay lets a household cut net exposure without realising embedded gains.

This module sizes that overlay (contracts, notional, SPAN-style margin, carry) and quantifies the tax interactions
that must be reviewed: Section 1256 marked-to-market 60/40 treatment, the straddle rules (§1092) when the overlay
offsets appreciated positions, and the fact that year-end overlay gains can consume harvested losses. Nothing here
trades; the output is a ticket and a set of flags for counsel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONTRACTS: dict[str, dict] = {
    # multiplier = index points x $; margin_pct approximates CME SPAN initial margin as % of notional (2026 planning)
    "ES": {"name": "E-mini S&P 500", "multiplier": 50.0, "margin_pct": 0.055, "underlying": "SPX", "proxy_etf": "SPY", "etf_ratio": 10.0},
    "MES": {"name": "Micro E-mini S&P 500", "multiplier": 5.0, "margin_pct": 0.055, "underlying": "SPX", "proxy_etf": "SPY", "etf_ratio": 10.0},
    "NQ": {"name": "E-mini Nasdaq-100", "multiplier": 20.0, "margin_pct": 0.075, "underlying": "NDX", "proxy_etf": "QQQ", "etf_ratio": 41.0},
    "MNQ": {"name": "Micro E-mini Nasdaq-100", "multiplier": 2.0, "margin_pct": 0.075, "underlying": "NDX", "proxy_etf": "QQQ", "etf_ratio": 41.0},
    "RTY": {"name": "E-mini Russell 2000", "multiplier": 50.0, "margin_pct": 0.07, "underlying": "RUT", "proxy_etf": "IWM", "etf_ratio": 10.0},
    "M2K": {"name": "Micro E-mini Russell 2000", "multiplier": 5.0, "margin_pct": 0.07, "underlying": "RUT", "proxy_etf": "IWM", "etf_ratio": 10.0},
}


@dataclass
class OverlayInputs:
    portfolio_value: float
    portfolio_beta: float                      # current beta to the overlay's index (from the risk model)
    target_beta: float = 1.0
    cash: float = 0.0                          # uninvested cash (adds zero beta)
    index_level: float | None = None           # e.g. SPX level; if None, derived from proxy ETF price x etf_ratio
    proxy_etf_price: float | None = None
    contract: str = "MES"
    days: int = 31                             # expected life of the overlay (a wash window by default)
    financing_rate: float = 0.045              # implied carry ≈ (r - dividend yield) x notional x days/365
    dividend_yield: float = 0.013
    st_rate: float = 0.408
    lt_rate: float = 0.238
    embedded_gain_hedged: float = 0.0          # $ of unrealised gain in appreciated positions the short overlay offsets (straddle risk)
    harvested_losses: float = 0.0              # $ of harvested losses this year (overlay gains can consume them)


@dataclass
class OverlayPlan:
    contract: str
    contracts: int                             # signed: + long, - short
    notional: float
    beta_gap: float
    beta_after: float
    residual_beta_gap: float
    margin_required: float
    margin_pct_of_portfolio: float
    carry_cost: float
    ticket: str
    flags: list[str] = field(default_factory=list)
    tax: dict = field(default_factory=dict)
    payoff: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "payoff"}
        d["payoff"] = self.payoff.to_dict("records") if not self.payoff.empty else []
        return d


def contract_notional(contract: str, index_level: float) -> float:
    return CONTRACTS[contract]["multiplier"] * index_level


def plan_overlay(inp: OverlayInputs) -> OverlayPlan:
    c = CONTRACTS[inp.contract]
    level = inp.index_level
    if level is None:
        if inp.proxy_etf_price is None:
            raise ValueError("need index_level or proxy_etf_price")
        level = inp.proxy_etf_price * c["etf_ratio"]
    invested = inp.portfolio_value - inp.cash
    beta_now = inp.portfolio_beta * (invested / inp.portfolio_value) if inp.portfolio_value > 0 else 0.0
    gap = inp.target_beta - beta_now
    target_notional = gap * inp.portfolio_value
    per = contract_notional(inp.contract, level)
    n = int(round(target_notional / per)) if per > 0 else 0
    notional = n * per
    beta_after = beta_now + notional / inp.portfolio_value if inp.portfolio_value > 0 else beta_now
    margin = abs(notional) * c["margin_pct"]
    carry = notional * (inp.financing_rate - inp.dividend_yield) * inp.days / 365.0     # long pays carry, short earns it
    flags: list[str] = []
    if n == 0:
        flags.append(f"beta gap {gap:+.3f} is below one {inp.contract} contract (${per:,.0f} notional); consider a micro contract or accept the gap.")
    if abs(beta_after - inp.target_beta) > 0.02:
        flags.append(f"contract granularity leaves a residual beta gap of {beta_after - inp.target_beta:+.3f}.")
    if margin > inp.cash and n != 0:
        flags.append(f"initial margin ${margin:,.0f} exceeds available cash ${inp.cash:,.0f}; the custodian will require funding or Reg-T borrowing.")
    if n < 0 and inp.embedded_gain_hedged > 0:
        flags.append("SHORT OVERLAY AGAINST APPRECIATED POSITIONS: §1092 straddle rules may suspend losses and terminate holding periods on the "
                     "offset positions; broad-based index futures are generally not 'substantially identical' to single names, but written tax "
                     "counsel is required before running this.")
    if inp.harvested_losses > 0:
        flags.append("§1256: overlay P&L is marked to market at year end, 60% long-term / 40% short-term. Overlay gains would consume harvested "
                     "losses; size and account for the overlay jointly with the harvest schedule.")
    # payoff / tax table across index moves
    moves = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
    rows = []
    for m in moves:
        pnl = notional * m
        tax = pnl * (0.6 * inp.lt_rate + 0.4 * inp.st_rate) if pnl > 0 else pnl * (0.6 * inp.lt_rate + 0.4 * inp.st_rate)
        port = invested * inp.portfolio_beta * m
        rows.append({"index_move": m, "overlay_pnl": pnl, "portfolio_pnl_beta_part": port, "combined": pnl + port,
                     "overlay_tax_1256": tax, "after_tax_overlay": pnl - tax})
    payoff = pd.DataFrame(rows)
    blended = 0.6 * inp.lt_rate + 0.4 * inp.st_rate
    ticket = (f"{'BUY' if n > 0 else 'SELL'} {abs(n)} {inp.contract} ({c['name']}) ≈ ${abs(notional):,.0f} notional at index {level:,.0f}; "
              f"margin ≈ ${margin:,.0f}; carry ≈ ${carry:,.0f} over {inp.days} days") if n else "no overlay"
    return OverlayPlan(contract=inp.contract, contracts=n, notional=notional, beta_gap=gap, beta_after=beta_after,
                       residual_beta_gap=beta_after - inp.target_beta, margin_required=margin,
                       margin_pct_of_portfolio=margin / inp.portfolio_value if inp.portfolio_value else 0.0, carry_cost=carry, ticket=ticket,
                       flags=flags, tax={"section_1256_blended_rate": blended, "vs_short_term_rate": inp.st_rate,
                                         "rate_advantage_vs_st": inp.st_rate - blended, "note": "60/40 treatment applies to §1256 contracts regardless of holding period."},
                       payoff=payoff)


def harvest_transition_overlay(portfolio_value: float, sell_value: float, buy_value: float, sold_beta: float, bought_beta: float,
                               portfolio_beta: float, target_beta: float | None = None, **kw) -> OverlayPlan:
    """Overlay that neutralises the beta change of a harvest trade list during the transition/settlement window."""
    target_beta = portfolio_beta if target_beta is None else target_beta
    beta_after = portfolio_beta - sold_beta * sell_value / portfolio_value + bought_beta * buy_value / portfolio_value
    cash = max(sell_value - buy_value, 0.0)
    return plan_overlay(OverlayInputs(portfolio_value=portfolio_value, portfolio_beta=beta_after * portfolio_value / max(portfolio_value - cash, 1e-9),
                                      target_beta=target_beta, cash=cash, **kw))


def micro_vs_mini(notional: float, index_level: float, family: str = "S&P 500") -> pd.DataFrame:
    fam = {"S&P 500": ("ES", "MES"), "Nasdaq-100": ("NQ", "MNQ"), "Russell 2000": ("RTY", "M2K")}[family]
    rows = []
    for code in fam:
        per = contract_notional(code, index_level)
        n = int(round(notional / per))
        rows.append({"contract": code, "name": CONTRACTS[code]["name"], "notional_per_contract": per, "contracts": n,
                     "achieved_notional": n * per, "rounding_error": n * per - notional, "margin": abs(n * per) * CONTRACTS[code]["margin_pct"]})
    return pd.DataFrame(rows)


def custodian_capabilities() -> pd.DataFrame:
    """Reference table from the broker/custodian research notes (Aug 2026); verify in diligence."""
    rows = [
        {"custodian": "TradeStation", "futures_in_account": True, "self_clearing": True, "margin": "Reg-T + SPAN", "direct_shorts": True, "options": True,
         "note": "Dual BD/FCM registration: overlay lives in the same account; block-and-allocate; FIX/REST API."},
        {"custodian": "Schwab (Advisor Services)", "futures_in_account": False, "self_clearing": True, "margin": "Reg-T 50/25", "direct_shorts": False, "options": True,
         "note": "No futures on the advisor platform; inverse/levered ETFs permitted; SIRP enrolment for short rebates."},
        {"custodian": "Fidelity", "futures_in_account": False, "self_clearing": True, "margin": "≈ Reg-T", "direct_shorts": False, "options": True,
         "note": "Default 50% margin; futures not permitted."},
        {"custodian": "Interactive Brokers", "futures_in_account": True, "self_clearing": True, "margin": "Reg-T / portfolio margin", "direct_shorts": True, "options": True,
         "note": "Futures and stocks in linked accounts with automatic margining."},
    ]
    return pd.DataFrame(rows)


def beta_from_model(model, weights: pd.Series, market_factor: str = "market") -> float:
    """Portfolio beta to the model's market factor (exposure-weighted)."""
    if market_factor not in model.factors:
        return float("nan")
    w = weights[weights.index.isin(model.symbols)]
    w = w / w.sum() if w.sum() else w
    return float(model.exposures.loc[w.index, market_factor] @ w)


def carry_breakeven_days(notional: float, financing_rate: float, dividend_yield: float, replacement_te: float, portfolio_value: float) -> float:
    """Days over which the overlay's carry equals the expected tracking-error cost (½·TE²·V·t/252) of an imperfect replacement."""
    daily_carry = abs(notional) * (financing_rate - dividend_yield) / 365.0
    te_cost_daily = 0.5 * replacement_te ** 2 * portfolio_value / 252.0
    return math.inf if te_cost_daily <= 0 else daily_carry / te_cost_daily
