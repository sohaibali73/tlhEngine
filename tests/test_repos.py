from datetime import date

import pytest

from tlh.db.database import Database
from tlh.db.repos import CodeRepo, EntityRepo, PortfolioRepo, TaxRepo
from tlh.tax.lots import LotMethod
from tlh.tax.washsale import SubstantiallyIdentical

D = date


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "t.sqlite")


@pytest.fixture
def setup(db):
    ent = EntityRepo(db)
    eid = ent.get_or_create("Household")
    tax_id = ent.get_or_create_account(eid, "Brokerage", "taxable")
    ira_id = ent.get_or_create_account(eid, "IRA", "ira")
    return eid, tax_id, ira_id


def test_purchase_sale_roundtrip_persists_lots_and_closures(db, setup):
    eid, tax_id, _ = setup
    repo = PortfolioRepo(db)
    lot = repo.record_purchase(tax_id, "AAA", 100, D(2024, 1, 10), 100, 50.0)
    assert lot.id > 0
    closures = repo.record_sale(tax_id, "AAA", 100, D(2025, 6, 1), 40, 45.0, method=LotMethod.FIFO)
    assert len(closures) == 1 and closures[0].id > 0
    book = repo.load_book(eid)
    assert len(book.lots) == 1 and book.lots[0].quantity_open == 60
    assert len(book.closures) == 1 and book.closures[0].realized_gain == pytest.approx(-200)
    assert book.closures[0].term == "LT"
    lots = repo.lots_frame(eid)
    assert lots.iloc[0]["quantity_open"] == 60
    assert repo.transactions_frame(eid).shape[0] == 2
    assert repo.closures_frame(eid).iloc[0]["realized_gain"] == pytest.approx(-200)


def test_cross_account_ira_wash_persists(db, setup):
    eid, tax_id, ira_id = setup
    repo = PortfolioRepo(db)
    repo.record_purchase(tax_id, "AAA", 100, D(2024, 1, 10), 100, 100.0)
    cs = repo.record_sale(tax_id, "AAA", 100, D(2025, 3, 15), 100, 60.0)
    assert cs[0].wash_disallowed == 0
    repo.record_purchase(ira_id, "AAA", 100, D(2025, 3, 20), 50, 61.0)
    book = repo.load_book(eid)
    c = book.closures[0]
    assert c.wash_disallowed == pytest.approx(2000)
    assert c.wash_matched_quantity == pytest.approx(50)
    assert "no basis step-up" in c.wash_explanation
    # replacement lot's used quantity is rehydrated so further sales don't double-count it
    ira_lot = next(lot for lot in book.lots if lot.account_id == ira_id)
    assert ira_lot.extra["used_as_replacement"] == pytest.approx(50)
    assert ira_lot.basis_adjustment == 0


def test_groups_from_substitutes_apply_in_repo(db, setup):
    eid, tax_id, _ = setup
    repo = PortfolioRepo(db)
    groups = SubstantiallyIdentical({1: "spx", 2: "spx"})
    repo.record_purchase(tax_id, "SPY", 1, D(2024, 1, 10), 10, 500.0, groups=groups)
    repo.record_sale(tax_id, "SPY", 1, D(2025, 3, 15), 10, 450.0, groups=groups)
    ivv = repo.record_purchase(tax_id, "IVV", 2, D(2025, 3, 16), 10, 452.0, groups=groups)
    assert ivv.basis_adjustment == pytest.approx(500)
    assert ivv.holding_start_date == D(2024, 1, 10)
    book = repo.load_book(eid, groups)
    assert book.closures[0].wash_disallowed == pytest.approx(500)


def test_scheduled_events_hydrate(db, setup):
    eid, tax_id, _ = setup
    repo = PortfolioRepo(db)
    repo.add_scheduled_event(tax_id, "AAA", 100, D(2025, 4, 1), "drip", 3)
    book = repo.load_book(eid)
    assert len(book.scheduled) == 1 and book.scheduled[0].kind == "scheduled_drip"
    assert repo.scheduled_events(eid).shape[0] == 1


def test_tax_profile_default_and_save(db):
    t = TaxRepo(db)
    p = t.default_profile()
    assert p.name == "default" and p.id is not None
    from tlh.tax.rates import TaxProfile
    t.save(TaxProfile(name="CA", state_rate=0.133), make_default=True)
    assert t.default_profile().name == "CA"


def test_code_versions_and_changes(db):
    c = CodeRepo(db)
    v1 = c.add_version("tlh/risk/factors.py", "x = 1", "human")
    v2 = c.add_version("tlh/risk/factors.py", "x = 2", "ai")
    assert c.latest_version("tlh/risk/factors.py")["version_no"] == 2
    assert c.get_version(v1)["is_active"] == 0 and c.get_version(v2)["is_active"] == 1
    cid = c.create_change("tlh/risk/factors.py", "bump", "x = 3", "why", "diff", None)
    c.set_sandbox_result(cid, "ok", True)
    assert c.change(cid)["status"] == "tested"
    c.set_status(cid, "approved", approved_by="user")
    assert c.change(cid)["approved_by"] == "user"
    assert db.fetchone("SELECT COUNT(*) AS n FROM audit_log")["n"] >= 4
