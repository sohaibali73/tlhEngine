from datetime import datetime, timedelta

import pytest

from tlh.ai import schedule as sched
from tlh.ai.copilot import TurnResult
from tlh.db.database import Database
from tlh.services.agent_service import TEMPLATES, AgentService
from tlh.services.context import AppContext


# ------------------------------------------------------------------ schedule grammar
def test_parse_and_describe():
    assert sched.parse("manual") == {"kind": "manual"}
    assert sched.parse("every 30m")["minutes"] == 30
    assert sched.parse("every 2h")["minutes"] == 120
    assert sched.parse("daily 08:30")["time"].hour == 8
    assert sched.parse("weekly mon 09:00")["dow"] == 0
    assert sched.parse("monthly 15 07:00")["day"] == 15
    assert "weekdays" in sched.describe("weekdays 16:30")
    assert sched.describe("bogus").startswith("invalid")
    with pytest.raises(ValueError):
        sched.parse("every 0m")


def test_next_run():
    after = datetime(2026, 9, 2, 10, 0)              # a Wednesday
    assert sched.next_run("manual", after) is None
    assert sched.next_run("every 30m", after) == after + timedelta(minutes=30)
    assert sched.next_run("daily 08:30", after) == datetime(2026, 9, 3, 8, 30)
    assert sched.next_run("daily 16:30", after) == datetime(2026, 9, 2, 16, 30)
    fri = sched.next_run("weekdays 16:30", datetime(2026, 9, 4, 17, 0))     # Friday after 16:30 -> Monday
    assert fri == datetime(2026, 9, 7, 16, 30)
    assert sched.next_run("weekly mon 09:00", after) == datetime(2026, 9, 7, 9, 0)
    assert sched.next_run("monthly 1 09:00", after) == datetime(2026, 10, 1, 9, 0)


# ------------------------------------------------------------------ service with a fake co-pilot
class FakeCopilot:
    available = True
    effort = "high"
    model = "fake"

    def __init__(self, ctx):
        self.ctx = ctx
        self.calls = []

    def new_conversation(self, title=None):
        return self.ctx.conversations.create("fake", title)

    def chat(self, cid, text, cb=None, extra_system=None):
        self.calls.append((text, extra_system, self.effort))
        if cb and cb.on_text:
            cb.on_text("working… ")
        return TurnResult(text="I did the thing.\n\n## Report\nAll good.", stop_reason="end_turn", usage={"input_tokens": 10},
                          cost_usd=0.01, change_ids=[7], tool_calls=[{"name": "x"}])

    def cancel(self):
        pass


@pytest.fixture
def svc(tmp_path):
    from tlh.config import Settings
    settings = Settings(TLH_VAR_DIR=str(tmp_path))
    ctx = AppContext(settings, db=Database(tmp_path / "t.sqlite"))
    return AgentService(ctx, copilot=FakeCopilot(ctx))


def test_task_crud_and_schedule(svc):
    tid = svc.save_task("scan", "do a scan", "daily 08:00", effort="low")
    df = svc.tasks()
    assert len(df) == 1 and df.iloc[0]["schedule_desc"] == "daily at 08:00" and df.iloc[0]["next_run_at"]
    svc.set_enabled(tid, False)
    assert svc.tasks().iloc[0]["enabled"] == 0 and svc.tasks().iloc[0]["next_run_at"] is None
    svc.set_enabled(tid, True)
    assert svc.tasks().iloc[0]["next_run_at"]
    n = svc.install_templates()
    assert n == len(TEMPLATES) and svc.install_templates() == 0
    svc.delete_task(tid)
    assert "scan" not in set(svc.tasks()["name"])


def test_due_tasks_and_run(svc):
    tid = svc.save_task("scan", "do a scan", "every 30m", effort="low")
    svc.ctx.agent.set_task(tid, next_run_at=(datetime.now() - timedelta(minutes=1)).isoformat())
    due = svc.due_tasks()
    assert [d["id"] for d in due] == [tid]
    res = svc.run_task(tid, trigger="schedule")
    assert res.status == "done" and res.report.startswith("## Report") and res.change_ids == [7]
    t = svc.ctx.agent.task(tid)
    assert t["last_status"] == "done" and datetime.fromisoformat(t["next_run_at"]) > datetime.now()
    assert "running as an UNATTENDED task" in svc.copilot.calls[0][1]
    assert svc.copilot.calls[0][2] == "low" and svc.copilot.effort == "high"    # effort override restored
    assert svc.due_tasks() == []
    runs = svc.runs()
    assert len(runs) == 1 and runs.iloc[0]["cost_usd"] == pytest.approx(0.01) and runs.iloc[0]["tool_calls"] == 1
    assert svc.unread() == 1
    svc.mark_read()
    assert svc.unread() == 0


def test_adhoc_run_and_failure(svc):
    res = svc.run_adhoc("build me something", trigger="popup")
    assert res.status == "done" and svc.runs().iloc[0]["trigger"] == "popup" and svc.runs().iloc[0]["task_id"] is None

    def boom(*a, **k):
        raise RuntimeError("api down")

    svc.copilot.chat = boom
    res = svc.run_adhoc("again")
    assert res.status == "failed" and "api down" in res.error
    assert svc.runs().iloc[0]["status"] == "failed"
