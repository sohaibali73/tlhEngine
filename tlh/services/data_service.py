"""Universe assembly and snapshot lifecycle."""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from ..data.cache import Snapshot
from .context import AppContext

log = logging.getLogger(__name__)


class DataService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    # ------------------------------------------------------------------ status
    def norgate_ok(self) -> bool:
        return self.ctx.norgate.status()

    def universe_name(self) -> str:
        return self.ctx.get("universe_name", self.ctx.settings.default_universe)

    def universe_symbols(self, entity_id: int | None = None) -> list[str]:
        """Fit universe = index watchlist + held names + every symbol in the substitute mapping."""
        syms: set[str] = set()
        try:
            syms |= set(self.ctx.norgate.watchlist_symbols(self.universe_name()))
        except Exception as e:
            log.warning("watchlist %s unavailable: %s", self.universe_name(), e)
        if entity_id is not None:
            syms |= set(self.ctx.portfolio.held_symbols(entity_id))
        syms |= self.ctx.substitutes.all_symbols()
        return sorted(syms)

    # ------------------------------------------------------------------ snapshots
    def latest_snapshot(self) -> Snapshot | None:
        return self.ctx.store.latest()

    def snapshot_is_current(self, snap: Snapshot, entity_id: int | None, max_age_days: int = 1) -> bool:
        if (date.today() - snap.as_of_date).days > max_age_days + 2:      # allow weekends
            return False
        if entity_id is not None:
            held = set(self.ctx.portfolio.held_symbols(entity_id))
            have = set(snap.manifest().get("returned_symbols", snap.symbols()))
            if held - have:
                return False
        return True

    def ensure_snapshot(self, entity_id: int | None = None, force: bool = False, progress=None) -> Snapshot:
        snap = self.latest_snapshot()
        if snap is not None and not force and self.snapshot_is_current(snap, entity_id):
            return snap
        self.ctx.norgate.require()
        symbols = self.universe_symbols(entity_id)
        snap = self.ctx.store.create(self.universe_name(), symbols, self.ctx.settings.price_history_start,
                                     progress=progress)
        sec = snap.securities()
        if not sec.empty:
            self.ctx.securities.upsert_frame(sec.rename(columns={"first_quoted": "first_quoted", "last_quoted": "last_quoted"}))
        return snap

    def prices_for(self, snap: Snapshot, symbols: list[str]) -> pd.Series:
        px = snap.last_prices()
        missing = [s for s in symbols if s not in px.index]
        if missing:
            # fall back to Norgate for names outside the snapshot (e.g. freshly added holdings)
            for s in missing:
                lc = self.ctx.norgate.last_close(s) if self.ctx.norgate.status() else None
                if lc:
                    px[s] = lc[1]
        return px

    def returns_matrix(self, snap: Snapshot, lookback: int = 300) -> pd.DataFrame:
        close = snap.close_matrix("close")
        return close.pct_change().iloc[-lookback:]
