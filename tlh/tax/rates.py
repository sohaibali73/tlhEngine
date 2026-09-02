"""Marginal tax assumptions and after-tax value of a harvested loss / realized gain.

The after-tax *benefit* of a realized loss depends on what it offsets. We use the standard planning
convention: a short-term loss is valued at the short-term marginal rate (it offsets short-term gains, taxed
as ordinary income, first), a long-term loss at the long-term rate. Both include state tax and NIIT when
applicable. `TaxProfile.benefit_of_loss` exposes an optional override for the case where the operator
knows the loss will only offset long-term gains (e.g. no short-term gains this year), which values a
short-term loss at the long-term rate instead.

Carryforward netting is in `ledger.py`; this module is pure rate arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .holding import LT, ST, Term


@dataclass(frozen=True)
class TaxProfile:
    name: str = "default"
    fed_st_rate: float = 0.37
    fed_lt_rate: float = 0.20
    state_rate: float = 0.0
    niit_rate: float = 0.038
    ordinary_offset: float = 3000.0          # annual cap on net capital loss vs ordinary income
    filing_status: str = "single"            # single | mfj | mfs | hoh
    apply_niit: bool = True
    id: int | None = field(default=None, compare=False)

    @property
    def st_rate(self) -> float:
        return self.fed_st_rate + self.state_rate + (self.niit_rate if self.apply_niit else 0.0)

    @property
    def lt_rate(self) -> float:
        return self.fed_lt_rate + self.state_rate + (self.niit_rate if self.apply_niit else 0.0)

    def rate_for(self, term: Term) -> float:
        return self.st_rate if term == ST else self.lt_rate

    @property
    def effective_ordinary_offset(self) -> float:
        return self.ordinary_offset / 2 if self.filing_status == "mfs" else self.ordinary_offset

    def benefit_of_loss(self, loss_amount: float, term: Term, offsets_term: Term | None = None) -> float:
        """Tax saved by realizing `loss_amount` (positive number) of a loss with character `term`.

        `offsets_term` overrides the character of the gains being offset (ST loss offsetting only LT gains
        is worth the LT rate). Returns a positive number of dollars.
        """
        if loss_amount < 0:
            raise ValueError("loss_amount must be non-negative")
        return loss_amount * self.rate_for(offsets_term or term)

    def tax_on_gain(self, gain_amount: float, term: Term) -> float:
        if gain_amount < 0:
            raise ValueError("gain_amount must be non-negative")
        return gain_amount * self.rate_for(term)

    def deferral_haircut(self, years: float, discount_rate: float = 0.04) -> float:
        """Present-value factor for a tax liability deferred `years` years.

        Harvesting lowers basis, so the loss saved today is (partly) a gain deferred to the eventual sale.
        Tax alpha = benefit today - PV(tax later). If the position is held to death (step-up) or donated,
        the deferred tax is never paid; callers pass years=inf for that assumption.
        """
        if years == float("inf"):
            return 0.0
        return (1.0 + discount_rate) ** (-max(years, 0.0))

    def tax_alpha(self, loss_amount: float, term: Term, horizon_years: float = 10.0,
                  discount_rate: float = 0.04, eventual_term: Term = LT) -> float:
        """Net present value of harvesting a loss: benefit now minus PV of the higher tax at liquidation.

        The eventual gain is taxed at `eventual_term` (long-term by default because the replacement is
        typically held > 1 year). With horizon_years=inf the deferred tax vanishes (step-up at death).
        """
        now = self.benefit_of_loss(loss_amount, term)
        later = self.tax_on_gain(loss_amount, eventual_term) * self.deferral_haircut(horizon_years, discount_rate)
        return now - later


DEFAULT_PROFILE = TaxProfile()
