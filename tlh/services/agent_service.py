"""Unattended agent: scheduled and ad-hoc co-pilot jobs that run in the background and file a report.

Every job is a normal co-pilot conversation with an agent instruction appended, so it has the same tools and the
same guardrails: no orders, code changes only as reviewable proposals. Runs are recorded in `agent_runs` and surface
as tray notifications / an unread badge in the GUI.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ..ai import schedule as sched
from ..ai.copilot import ChatCallbacks, Copilot, TurnResult
from .context import AppContext

log = logging.getLogger(__name__)

AGENT_SUFFIX = """You are YANG running as an UNATTENDED task named "{name}". Nobody is watching; do the work end to end with \
your tools, do not ask questions. Use the current portfolio/model/run state. If something is impossible, say so. \
Finish with a section titled "## Report" (concise markdown: what you found, what you created — run ids, basket names, \
change ids — and any action the operator should consider). Never claim a trade was executed; you cannot trade."""

TEMPLATES: list[dict] = [
    {"name": "Daily harvest scan", "schedule": "weekdays 16:30", "effort": "medium",
     "prompt": "Run the harvest optimizer with the saved default configuration (run_harvest with no overrides). Then report: total harvestable loss, "
               "harvested loss and tax alpha in the recommendation, TE before/after, each blocked lot with its reason, and the top 5 trades as a table. "
               "If tax alpha exceeds 10 bps of portfolio value, say prominently that a harvest is worth reviewing today."},
    {"name": "Wash-window watch", "schedule": "daily 08:00", "effort": "low",
     "prompt": "Using get_portfolio_context, list every open wash-sale window (loss sales blocking repurchases, recent purchases blocking loss sales, "
               "scheduled DRIPs) with the date it closes. Highlight windows closing within 5 days and any lot that becomes long-term within 10 days."},
    {"name": "Weekly model health", "schedule": "weekly mon 07:30", "effort": "medium",
     "prompt": "Fit a risk-model variant with the default spec but lookback_days=756 (do not activate). Compare it to the active model: factor vol changes, "
               "R², my portfolio TE under each. Report whether the active model looks stale and which factor drives most of today's TE."},
    {"name": "Month-end transition check", "schedule": "monthly 25 09:00", "effort": "medium",
     "prompt": "If a basket exists whose name starts with 'Core', run a tax-aware transition toward it with a 0.5% net realised-gain budget and 30% turnover cap "
               "(build_strategy_basket with strategy tax_aware_transition, save=false). Report TE-to-target before/after, turnover, realised gains vs losses, and "
               "the ten largest implied trades. If no Core basket exists, say so and suggest building one."},
    {"name": "Strategy leaderboard", "schedule": "monthly 1 07:00", "effort": "medium",
     "prompt": "Backtest min_variance, risk_parity and hrp (n_max 30, quarterly rebalances, since 2022, benchmark SPY). Rank by information ratio and "
               "present a table of CAGR, vol, Sharpe, max drawdown, TE, IR and turnover. State the backtester's caveats verbatim."},
]


@dataclass
class AgentRunResult:
    run_id: int
    status: str
    report: str
    turn: TurnResult | None = None
    error: str | None = None
    change_ids: list[int] = field(default_factory=list)


class AgentService:
    def __init__(self, ctx: AppContext, copilot: Copilot | None = None):
        self.ctx = ctx
        self.copilot = copilot or Copilot(ctx)
        self._lock = threading.Lock()
        self._running: int | None = None

    # ------------------------------------------------------------------ tasks
    def tasks(self) -> pd.DataFrame:
        df = self.ctx.agent.tasks()
        if not df.empty:
            df["schedule_desc"] = df["schedule"].apply(sched.describe)
        return df

    def save_task(self, name: str, prompt: str, schedule: str = "manual", enabled: bool = True, notify: bool = True,
                  effort: str | None = None) -> int:
        sched.parse(schedule)                                   # validate
        nxt = sched.next_run(schedule, datetime.now())
        return self.ctx.agent.upsert_task(name, prompt, schedule, enabled, notify, effort, nxt.isoformat() if nxt else None)

    def install_templates(self, enabled: bool = False) -> int:
        n = 0
        for t in TEMPLATES:
            if self.ctx.agent.task_by_name(t["name"]) is None:
                self.save_task(t["name"], t["prompt"], t["schedule"], enabled=enabled, effort=t.get("effort"))
                n += 1
        return n

    def set_enabled(self, task_id: int, enabled: bool) -> None:
        t = self.ctx.agent.task(task_id)
        if not t:
            return
        nxt = sched.next_run(t["schedule"], datetime.now()) if enabled else None
        self.ctx.agent.set_task(task_id, enabled=int(enabled), next_run_at=nxt.isoformat() if nxt else None)

    def delete_task(self, task_id: int) -> None:
        self.ctx.agent.delete_task(task_id)

    def due_tasks(self, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now()
        df = self.ctx.agent.tasks(enabled_only=True)
        if df.empty:
            return []
        out = []
        for _, r in df.iterrows():
            if r["schedule"] in ("manual", "startup"):
                continue
            nra = r["next_run_at"]
            if not nra:                                          # schedule saved without next run: compute lazily
                nxt = sched.next_run(r["schedule"], now)
                self.ctx.agent.set_task(int(r["id"]), next_run_at=nxt.isoformat() if nxt else None)
                continue
            if datetime.fromisoformat(nra) <= now:
                out.append(dict(r))
        return out

    def startup_tasks(self) -> list[dict]:
        df = self.ctx.agent.tasks(enabled_only=True)
        if df.empty:
            return []
        return [dict(r) for _, r in df[df["schedule"] == "startup"].iterrows()]

    @property
    def running_run_id(self) -> int | None:
        return self._running

    # ------------------------------------------------------------------ running
    def run_task(self, task_id: int, trigger: str = "manual", on_text: Callable[[str], None] | None = None,
                 on_status: Callable[[str], None] | None = None) -> AgentRunResult:
        t = self.ctx.agent.task(task_id)
        if not t:
            raise KeyError(task_id)
        res = self._run(t["name"], t["prompt"], trigger, task_id=task_id, effort=t.get("effort"), on_text=on_text, on_status=on_status)
        nxt = sched.next_run(t["schedule"], datetime.now())
        self.ctx.agent.set_task(task_id, last_run_at=pd.Timestamp.now().isoformat(), next_run_at=nxt.isoformat() if nxt else None,
                                last_status=res.status, last_summary=(res.report or res.error or "")[:400])
        return res

    def run_adhoc(self, prompt: str, name: str | None = None, trigger: str = "popup", effort: str | None = None,
                  on_text: Callable[[str], None] | None = None, on_status: Callable[[str], None] | None = None) -> AgentRunResult:
        return self._run(name or prompt[:60], prompt, trigger, task_id=None, effort=effort, on_text=on_text, on_status=on_status)

    def _run(self, name: str, prompt: str, trigger: str, task_id: int | None, effort: str | None,
             on_text=None, on_status=None) -> AgentRunResult:
        if not self.copilot.available:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; the agent cannot run.")
        with self._lock:                                          # one unattended job at a time
            cid = self.copilot.new_conversation(title=f"[agent] {name}")
            rid = self.ctx.agent.start_run(task_id, name, prompt, trigger, cid)
            self._running = rid
            t0 = time.time()
            prev_effort = self.copilot.effort
            if effort:
                self.copilot.effort = effort
            try:
                cb = ChatCallbacks(on_text=on_text, on_status=on_status)
                turn = self.copilot.chat(cid, prompt, cb, extra_system=AGENT_SUFFIX.format(name=name))
                report = _extract_report(turn.text)
                status = "done" if turn.stop_reason in ("end_turn", "max_tokens") else ("cancelled" if turn.stop_reason == "cancelled" else "failed")
                self.ctx.agent.finish_run(rid, status, report=report, change_ids=turn.change_ids, tool_calls=len(turn.tool_calls),
                                          cost_usd=turn.cost_usd, duration_s=time.time() - t0)
                self.ctx.db.audit("ai", "agent.run", name, run_id=rid, status=status, cost=round(turn.cost_usd, 4), trigger=trigger)
                return AgentRunResult(rid, status, report, turn, change_ids=turn.change_ids)
            except Exception as e:
                log.exception("agent run %s failed", name)
                self.ctx.agent.finish_run(rid, "failed", error=f"{type(e).__name__}: {e}", duration_s=time.time() - t0)
                self.ctx.db.audit("ai", "agent.run", name, run_id=rid, status="failed", error=str(e)[:300])
                return AgentRunResult(rid, "failed", "", None, error=str(e))
            finally:
                self.copilot.effort = prev_effort
                self._running = None

    # ------------------------------------------------------------------ results
    def runs(self, limit: int = 100) -> pd.DataFrame:
        return self.ctx.agent.runs(limit)

    def run_detail(self, run_id: int) -> dict | None:
        return self.ctx.agent.run(run_id)

    def unread(self) -> int:
        return self.ctx.agent.unread_count()

    def mark_read(self, run_ids: list[int] | None = None) -> None:
        self.ctx.agent.mark_read(run_ids)

    def pending_changes(self) -> int:
        df = self.ctx.code.changes(status="tested")
        return int(len(df))


def _extract_report(text: str) -> str:
    if not text:
        return ""
    i = text.rfind("## Report")
    return text[i:].strip() if i >= 0 else text.strip()
