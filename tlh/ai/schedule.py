"""Tiny schedule grammar for unattended agent tasks.

    manual                  never runs automatically
    startup                 once each time the app starts
    every 30m | every 2h    fixed interval
    daily 08:30             every day at a local time
    weekdays 16:30          Monday-Friday at a local time
    weekly mon 09:00        one weekday per week
    monthly 1 09:00         a day of month (1-28) at a local time
"""
from __future__ import annotations

import re
from datetime import datetime, time, timedelta

DOW = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_TIME = r"(\d{1,2}):(\d{2})"


def parse(schedule: str) -> dict:
    s = (schedule or "manual").strip().lower()
    if s in ("manual", ""):
        return {"kind": "manual"}
    if s == "startup":
        return {"kind": "startup"}
    m = re.fullmatch(r"every\s+(\d+)\s*(m|min|mins|minutes|h|hr|hrs|hours|d|day|days)", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)[0]
        minutes = n * {"m": 1, "h": 60, "d": 1440}[unit]
        if minutes < 1:
            raise ValueError("interval must be >= 1 minute")
        return {"kind": "interval", "minutes": minutes}
    m = re.fullmatch(rf"daily\s+{_TIME}", s)
    if m:
        return {"kind": "daily", "time": _t(m.group(1), m.group(2))}
    m = re.fullmatch(rf"weekdays\s+{_TIME}", s)
    if m:
        return {"kind": "weekdays", "time": _t(m.group(1), m.group(2))}
    m = re.fullmatch(rf"weekly\s+([a-z]{{3}})\s+{_TIME}", s)
    if m and m.group(1) in DOW:
        return {"kind": "weekly", "dow": DOW[m.group(1)], "time": _t(m.group(2), m.group(3))}
    m = re.fullmatch(rf"monthly\s+(\d{{1,2}})\s+{_TIME}", s)
    if m and 1 <= int(m.group(1)) <= 28:
        return {"kind": "monthly", "day": int(m.group(1)), "time": _t(m.group(2), m.group(3))}
    raise ValueError(f"unrecognised schedule '{schedule}' (try: manual, startup, every 30m, daily 08:30, weekdays 16:30, weekly mon 09:00, monthly 1 09:00)")


def _t(h: str, m: str) -> time:
    hh, mm = int(h), int(m)
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError("bad time")
    return time(hh, mm)


def next_run(schedule: str, after: datetime) -> datetime | None:
    """First run strictly after `after` (local time). None for manual/startup."""
    p = parse(schedule)
    k = p["kind"]
    if k in ("manual", "startup"):
        return None
    if k == "interval":
        return after + timedelta(minutes=p["minutes"])
    t = p["time"]
    cand = datetime.combine(after.date(), t)
    if k == "daily":
        return cand if cand > after else cand + timedelta(days=1)
    if k == "weekdays":
        while cand <= after or cand.weekday() > 4:
            cand += timedelta(days=1)
        return cand
    if k == "weekly":
        while cand <= after or cand.weekday() != p["dow"]:
            cand += timedelta(days=1)
        return cand
    if k == "monthly":
        y, mo = after.year, after.month
        for _ in range(3):
            cand = datetime.combine(datetime(y, mo, p["day"]).date(), t)
            if cand > after:
                return cand
            mo += 1
            if mo > 12:
                mo, y = 1, y + 1
    return None


def describe(schedule: str) -> str:
    try:
        p = parse(schedule)
    except ValueError as e:
        return f"invalid: {e}"
    k = p["kind"]
    if k == "manual":
        return "manual only"
    if k == "startup":
        return "at app startup"
    if k == "interval":
        m = p["minutes"]
        return f"every {m} min" if m < 60 else (f"every {m // 60} h" if m % 60 == 0 else f"every {m} min")
    t = p["time"].strftime("%H:%M")
    if k == "daily":
        return f"daily at {t}"
    if k == "weekdays":
        return f"weekdays at {t}"
    if k == "weekly":
        return f"{[d for d, i in DOW.items() if i == p['dow']][0].title()} at {t}"
    return f"day {p['day']} of each month at {t}"
