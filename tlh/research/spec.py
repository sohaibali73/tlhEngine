"""Parameters of one research run and the sweep grids requested for due diligence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

APPROACHES = {
    "pairs_sector": "Pairs on the fly, same sector: each harvested name is replaced by its most correlated (trailing 252-day) index member in the same GICS sector that is not held and not inside a wash window.",
    "pairs_index": "Pairs on the fly, whole index: the most correlated index member anywhere in the S&P 500 replaces the harvested name.",
    "twin_baskets": "Twin baskets: before trading starts, two baskets are built so every held name has a pre-assigned twin (same sector, minimum sum-absolute-rank-difference over size, momentum, volatility, beta and dividend yield); harvesting sells a name and buys its twin, later swaps back. Re-paired annually.",
    "optimizer": "Tracking-error optimizer: after the loss sales, the buy side is solved as a convex minimum-TE problem against the index (calibrated 126-day Ledoit-Wolf covariance) with a sector band, factor-alignment bands, the basket-size cap and wash-window exclusions.",
}

# the grids from the research brief
ACCOUNT_SIZES = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
BASKET_SIZES = [50, 100, 150, 200, 250, 300]
TRIGGERS = [0.0001, 0.0005, 0.0010, 0.0015, 0.0025, 0.0050, 0.0075, 0.0100]
CONCENTRATED_SIZES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONCENTRATED_GAINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class ResearchSpec:
    # universe / window
    horizon_years: int = 10                  # 5 or 10
    start_year: int = 2010                   # the window starts on the first trading day of this year
    rebalance: str = "M"                     # monthly review
    # account
    account_size: float = 500_000.0
    whole_shares: bool = True
    min_trade: float = 250.0                 # dollars; smaller buys are skipped (fractional accounts: set 0 and whole_shares False)
    cost_bps: float = 5.0
    # construction
    basket_size: int = 150
    te_limit: float = 0.02                   # ex-ante TE budget (optimizer) / reported against for all approaches
    sector_band: float = 0.02                # |sector active weight| <= band (optimizer); twin/pairs keep sector by construction
    factor_alignment: bool = True            # optimizer: |active exposure| <= factor_band in z-units for size / momentum / vol / beta
    factor_band: float = 0.10
    approach: str = "optimizer"
    cov_lookback: int = 126                  # the calibration study's winner: 126 days, equal weights, Ledoit-Wolf
    # harvesting
    trigger: float = 0.0025                  # lot loss as a fraction of account value (trigger_basis="account") or of lot cost ("lot")
    trigger_basis: str = "account"
    wash_days: int = 30
    min_harvest: float = 100.0               # dollars of loss below which a lot is not worth selling
    # concentrated start
    concentrated_pct: float = 0.0            # fraction of the account in one stock at the start (rest is cash)
    concentrated_gain: float = 0.0           # unrealised gain on that stock as a fraction of its value (basis = value / (1 + gain))
    concentrated_symbol: str | None = None   # default: the largest index member at the start date
    gain_budget: float = 0.0                 # extra realised gains allowed per year while unwinding, fraction of account (0 = losses only)
    # tax value of losses (reporting only)
    st_rate: float = 0.408
    lt_rate: float = 0.238
    # ossification definition: trailing-12-month harvested losses below this fraction of account value
    ossification_yield: float = 0.002
    seed: int = 0

    def with_(self, **kw) -> ResearchSpec:
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def end_year(self) -> int:
        return self.start_year + self.horizon_years


@dataclass
class StudySpec:
    """A named study: a base case plus the sweeps to run over rolling windows."""
    name: str = "MVP"
    base: ResearchSpec = field(default_factory=ResearchSpec)
    sweeps: list[str] = field(default_factory=lambda: ["account_size", "basket_size", "trigger", "approach"])
    horizons: list[int] = field(default_factory=lambda: [10])
    first_start_year: int = 2000
    last_start_year: int | None = None       # default: the last year with a full horizon
    every_n_years: int = 1                   # 1 = every calendar year start; 3 = quick mode
    account_sizes: list[float] = field(default_factory=lambda: list(ACCOUNT_SIZES))
    basket_sizes: list[int] = field(default_factory=lambda: list(BASKET_SIZES))
    triggers: list[float] = field(default_factory=lambda: list(TRIGGERS))
    approaches: list[str] = field(default_factory=lambda: list(APPROACHES))
    concentrated_sizes: list[float] = field(default_factory=lambda: [x for x in CONCENTRATED_SIZES if x > 0])
    concentrated_gains: list[float] = field(default_factory=lambda: list(CONCENTRATED_GAINS))

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
