"""Plain-English explanations of engine outputs, for the Start-here screen, exports and YANG.

Every sentence is generated from the numbers the engine produced; nothing here computes tax or risk.
"""
from __future__ import annotations


def money(v: float | None, decimals: int = 0) -> str:
    if v is None or v != v:
        return "n/a"
    s = f"{abs(v):,.{decimals}f}"
    return f"-${s}" if v < 0 else f"${s}"


def pct(v: float | None, decimals: int = 1) -> str:
    if v is None or v != v:
        return "n/a"
    return f"{v * 100:.{decimals}f}%"


def explain_harvest(summary: dict, st_rate: float, lt_rate: float, benchmark: str = "the benchmark") -> list[str]:
    """Sentences an advisor can read to a client, from a HarvestResult.summary dict."""
    out: list[str] = []
    loss = float(summary.get("harvested_loss") or 0.0)
    st = float(summary.get("harvested_loss_st") or 0.0)
    lt = float(summary.get("harvested_loss_lt") or 0.0)
    benefit = float(summary.get("tax_benefit") or 0.0)
    alpha = float(summary.get("tax_alpha") or 0.0)
    n_sells = int(summary.get("n_sells") or 0)
    n_buys = int(summary.get("n_buys") or 0)
    te0, te1 = summary.get("te_before"), summary.get("te_after")
    blocked = int(summary.get("n_blocked_lots") or 0)
    value = float(summary.get("portfolio_value") or 0.0)
    if loss <= 0:
        out.append("No wash-safe losses worth harvesting today. The book is either above water or every loser is inside a "
                   "wash-sale window; the engine will keep watching.")
        if blocked:
            out.append(f"{blocked} loss lot{'s are' if blocked != 1 else ' is'} blocked by the wash-sale rule right now; the Harvest screen "
                       "lists each one with the date it clears.")
        return out
    out.append(f"Selling {n_sells} lot{'s' if n_sells != 1 else ''} realises {money(loss)} of losses"
               + (f" ({money(st)} short-term, {money(lt)} long-term)" if st and lt else (" (all short-term)" if st else " (all long-term)")) + ".")
    out.append(f"At this household's marginal rates ({pct(st_rate, 1)} short-term, {pct(lt_rate, 1)} long-term) that is worth about "
               f"{money(benefit)} of tax this year.")
    if alpha and value:
        out.append(f"After allowing for the lower cost basis (a bigger gain to tax later), the true after-tax gain, called tax alpha, is about "
                   f"{money(alpha)}, or {alpha / value * 1e4:.0f} basis points of the portfolio.")
    if n_buys:
        out.append(f"The proceeds go into {n_buys} replacement{'s' if n_buys != 1 else ''} chosen to track the sold names closely, so the "
                   f"portfolio stays invested and is never out of the market.")
    if te0 is not None and te1 is not None:
        direction = "unchanged" if abs(te1 - te0) < 5e-4 else ("slightly higher" if te1 > te0 else "lower")
        out.append(f"Tracking error versus {benchmark} goes from {pct(te0, 2)} to {pct(te1, 2)} ({direction}); the portfolio still behaves like its target.")
    out.append("Every sell was checked against the wash-sale rule across all accounts, including IRAs, and every buy was checked against "
               "recent loss sales. Nothing here trades automatically: these are tickets for a human to approve.")
    if blocked:
        out.append(f"{blocked} additional loss lot{'s were' if blocked != 1 else ' was'} skipped because selling now would be a wash sale; each has a clear date.")
    return out


def explain_kpis(k: dict) -> str:
    parts = []
    if k.get("market_value"):
        parts.append(f"The portfolio is worth {money(k['market_value'])} across {k.get('n_positions', 0)} positions.")
    hl = k.get("harvestable_loss") or 0.0
    if hl > 0:
        parts.append(f"{money(hl)} of unrealised losses could be harvested today"
                     + (f", of which {money(k.get('blocked_loss'))} is currently wash-blocked" if k.get("blocked_loss") else "") + ".")
    else:
        parts.append("There are no unrealised losses to harvest right now.")
    if k.get("ytd_harvested"):
        parts.append(f"So far this year {money(k['ytd_harvested'])} of losses have been realised, worth about {money(k.get('ytd_tax_value'))} of tax.")
    if k.get("tracking_error") is not None:
        parts.append(f"Tracking error versus {k.get('benchmark', 'the benchmark')} is {pct(k['tracking_error'], 2)}.")
    return " ".join(parts)


def explain_state(c: dict) -> str:
    return (f"In {c['state']} a short-term gain (or loss) is taxed at about {pct(c['total_st'])} all-in "
            f"({pct(c['fed_st'])} federal, {pct(c['niit'])} NIIT, {pct(c['state_st'])} state) and a long-term one at {pct(c['total_lt'])} "
            f"({pct(c['fed_lt'])} federal, {pct(c['niit'])} NIIT, {pct(c['state_lt'])} state). {c['description']}"
            + (f" {c['local_note']}" if c.get("local_note") else "")
            + f" State figures are approximate {c['year']} planning values; verify before filing.")
