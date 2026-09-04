"""TLH research: a due-diligence backtesting laboratory for the harvesting parameters.

Answers, with numbers over rolling 5- and 10-year windows since 2000 on the point-in-time S&P 500:
    how much loss a direct-indexed account harvests, how long it keeps harvesting before it ossifies, and what tracking
    error it pays, as a function of account size, basket size, harvest trigger, harvesting approach (pairs on the fly
    within sector or index, twin baskets paired by SARD, tracking-error optimizer), and a concentrated starting position
    with an embedded gain.

Modules: data.py (deep Norgate store: prices, dividends, membership, sectors, index), spec.py (ResearchSpec), engine.py
(lot-level monthly simulator with wash-sale windows, whole shares, three approaches), grid.py (sweep design, multi-core
runner, result store, summaries), report.py (due-diligence write-up).
"""
