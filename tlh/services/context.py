"""AppContext: one object that owns settings, the database, repositories and data clients.

Services and the GUI receive an AppContext; nothing else constructs repositories directly.
"""
from __future__ import annotations

import logging
from datetime import date

from ..config import Settings, get_settings
from ..data.cache import SnapshotStore
from ..data.norgate import NorgateClient
from ..data.substitutes import SubstituteMap
from ..db.database import Database
from ..db.repos import (
    AgentRepo,
    BasketRepo,
    CodeRepo,
    ConversationRepo,
    EntityRepo,
    ModelRepo,
    PipelineRepo,
    PortfolioRepo,
    RunRepo,
    SecurityRepo,
    TaxRepo,
)
from ..tax.washsale import SubstantiallyIdentical

log = logging.getLogger(__name__)


class AppContext:
    def __init__(self, settings: Settings | None = None, db: Database | None = None,
                 norgate: NorgateClient | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.db = db or Database(self.settings.db_path)
        self.norgate = norgate or NorgateClient()
        self.store = SnapshotStore(self.db, self.settings.snapshots_dir, self.norgate)
        self.entities = EntityRepo(self.db)
        self.securities = SecurityRepo(self.db)
        self.portfolio = PortfolioRepo(self.db)
        self.tax = TaxRepo(self.db)
        self.runs = RunRepo(self.db)
        self.models = ModelRepo(self.db)
        self.code = CodeRepo(self.db)
        self.conversations = ConversationRepo(self.db)
        self.baskets = BasketRepo(self.db)
        self.agent = AgentRepo(self.db)
        self.pipelines = PipelineRepo(self.db)
        self._subs: SubstituteMap | None = None

    # ------------------------------------------------------------------ settings in DB
    def get(self, key: str, default=None):
        return self.db.get_setting(key, default)

    def set(self, key: str, value) -> None:
        self.db.set_setting(key, value)

    @property
    def treat_presumed_identical(self) -> bool:
        return bool(self.get("treat_presumed_identical_as_identical", True))

    @property
    def current_entity_id(self) -> int | None:
        eid = self.get("current_entity_id")
        if eid is None:
            ents = self.entities.list()
            eid = ents[0]["id"] if ents else None
        return eid

    @current_entity_id.setter
    def current_entity_id(self, eid: int) -> None:
        self.set("current_entity_id", int(eid))

    # ------------------------------------------------------------------ substitutes / groups
    @property
    def substitutes(self) -> SubstituteMap:
        if self._subs is None:
            self._subs = SubstituteMap.load(treat_presumed_as_identical=self.treat_presumed_identical)
        return self._subs

    def reload_substitutes(self) -> SubstituteMap:
        self._subs = None
        return self.substitutes

    def resolve_assetid(self, symbol: str) -> int | None:
        aid = self.securities.resolve(symbol)
        if aid is not None:
            return aid
        meta = None
        try:
            meta = self.norgate.security_meta(symbol)
        except Exception as e:  # NDU down or unknown symbol
            log.debug("resolve %s failed: %s", symbol, e)
        if meta is None:
            return None
        self.securities.upsert(meta.__dict__)
        return meta.assetid

    def groups(self) -> SubstantiallyIdentical:
        return self.substitutes.to_substantially_identical(self.resolve_assetid)

    def today(self) -> date:
        return date.today()

    def close(self) -> None:
        self.db.close()
