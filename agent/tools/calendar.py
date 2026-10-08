"""A bounded evaluator for Asterisk/FreePBX time-group expressions."""

import re
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ..context import get


_DAYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_TIME = re.compile(r"^(\d\d):(\d\d)$")


def _clock(value: str) -> int:
    match = _TIME.fullmatch(value)
    if not match:
        raise ValueError("unsupported time")
    hour, minute = map(int, match.groups())
    if hour > 23 or minute > 59:
        raise ValueError("unsupported time")
    return hour * 60 + minute


def _token(value: str, names: list[str] | None, maximum: int) -> int:
    value = value.strip().lower()
    if names and value in names:
        return names.index(value) + 1
    if value.isdecimal() and 1 <= int(value) <= maximum:
        return int(value)
    raise ValueError("unsupported range token")


def _matches(expression: str, value: int, names: list[str] | None, maximum: int) -> bool:
    if expression == "*":
        return True
    matched = False
    for segment in expression.lower().split("&"):
        parts = segment.split("-")
        if len(parts) > 2:
            raise ValueError("unsupported range")
        first = _token(parts[0], names, maximum)
        last = _token(parts[-1], names, maximum)
        if first <= last and first <= value <= last:
            matched = True
        if first > last and (value >= first or value <= last):
            matched = True
    return matched


def validate_rule(rule: str) -> None:
    parts = rule.split("|")
    if len(parts) != 4:
        raise ValueError("unsupported rule")
    times, weekdays, monthdays, months = parts
    if times != "*":
        for segment in times.split("&"):
            clocks = segment.split("-")
            if len(clocks) not in (1, 2):
                raise ValueError("unsupported time range")
            for clock in clocks:
                _clock(clock)
    _matches(weekdays, 1, _DAYS, 7)
    _matches(monthdays, 1, None, 31)
    _matches(months, 1, _MONTHS, 12)


def _parse_rule(rule: str, day: date):
    parts = rule.split("|")
    if len(parts) != 4:
        raise ValueError("unsupported rule")
    times, weekdays, monthdays, months = parts
    # Asterisk day numbering is Sunday=1; Python is Monday=0.
    weekday = (day.weekday() + 1) % 7 + 1
    if (_matches(weekdays, weekday, _DAYS, 7)
            and _matches(monthdays, day.day, None, 31)
            and _matches(months, day.month, _MONTHS, 12)):
        if times == "*":
            return [(0, 1439)]
        ranges = []
        for segment in times.split("&"):
            time_parts = segment.split("-")
            if len(time_parts) == 1:
                minute = _clock(time_parts[0])
                ranges.append((minute, minute))
            elif len(time_parts) == 2:
                ranges.append((_clock(time_parts[0]), _clock(time_parts[1])))
            else:
                raise ValueError("unsupported time range")
        return ranges
    return []


def _local(day: date, minute: int, tz: ZoneInfo) -> datetime:
    candidate = datetime(day.year, day.month, day.day, minute // 60, minute % 60, tzinfo=tz)
    # Reject nonexistent wall time during the spring DST jump. Ambiguous fall
    # times use the earlier occurrence (fold=0), consistently with Asterisk.
    back = candidate.astimezone(timezone.utc).astimezone(tz)
    if back.replace(tzinfo=None) != candidate.replace(tzinfo=None):
        raise ValueError("nonexistent local opening time")
    return candidate


def _intervals(rules: list[str], day: date, tz: ZoneInfo) -> list[tuple[datetime, datetime]]:
    intervals = []
    for rule in rules:
        for start, end in _parse_rule(rule, day):
            # Asterisk stores a minute bitmap and applies the weekday mask to
            # the current local day. An overnight range enables both pieces on
            # that weekday; it does not carry yesterday's weekday into today.
            pieces = [(start, end + 1)] if end >= start else [(0, end + 1), (start, 1440)]
            for first, after_last in pieces:
                start_dt = _local(day, first, tz)
                end_dt = (_local(day + timedelta(days=1), 0, tz)
                          if after_last == 1440 else _local(day, after_last, tz))
                intervals.append((start_dt, end_dt))
    return sorted(intervals)


def _unknown(service_id: str, cal_id: str | None = None, reason: str = "unavailable") -> dict:
    return {"status": "unknown", "is_open": None, "opens_at": None,
            "closes_at": None, "timezone": None, "next_opening": None,
            "service_id": service_id, "source_id": cal_id, "reason": reason}


def opening_hours(context, service_id: str, requested_date: str) -> dict:
    profile = get(context, "profile") or {}
    cal_id = profile.get("calendar_services", {}).get(service_id)
    calendars = get(context, "calendars") or {}
    cal = calendars.get(cal_id) if cal_id else None
    if not isinstance(cal, dict):
        return _unknown(service_id, cal_id)
    if not cal.get("supported") or cal.get("override") == "unknown":
        return _unknown(service_id, cal_id, "unsupported")
    now = int(time.time())
    max_age = cal.get("max_age_seconds", 300)
    if type(max_age) is not int or max_age < 0 or now - cal.get("observed_at", 0) > max_age:
        return _unknown(service_id, cal_id, "stale")
    try:
        day = date.fromisoformat(requested_date)
        tz = ZoneInfo(cal["timezone"])
        rules = cal["rules"]
        for rule in rules:
            validate_rule(rule)
        exceptions = cal.get("exceptions", {})
        if not isinstance(exceptions, dict):
            raise ValueError("unsupported exception format")
        exception = exceptions.get(requested_date)
        if exception not in (None, "closed", "open"):
            raise ValueError("unsupported exception")
        intervals = [] if exception == "closed" else _intervals(rules, day, tz)
        local_now = datetime.now(tz)
        today = day == local_now.date()
        if not today and cal["override"] != "auto":
            # A temporary override may reset at the next PBX boundary. Its
            # present value cannot establish the actual state on another date.
            return _unknown(service_id, cal_id, "override_date_unknown")
        if cal["override"] == "closed":
            is_open = False
        elif cal["override"] == "open":
            is_open = True
        elif exception == "open":
            is_open = True
        else:
            is_open = any(start <= local_now < end for start, end in intervals) if today else bool(intervals)
        opening = next(((start, end) for start, end in intervals
                        if today and start <= local_now < end), None)
        if opening is None:
            opening = intervals[0] if intervals else None
        opens_at = opening[0].strftime("%H:%M") if opening else None
        closes_at = (opening[1] - timedelta(minutes=1)).strftime("%H:%M") if opening else None
        if cal["override"] != "auto":
            opens_at = closes_at = None
        next_opening = None
        if not is_open and cal["override"] == "auto":
            for offset in range(0 if today else 1, 371):
                future = day + timedelta(days=offset)
                if exceptions.get(future.isoformat()) == "closed":
                    continue
                upcoming = _intervals(rules, future, tz)
                if today and offset == 0:
                    upcoming = [(start, end) for start, end in upcoming if start > local_now]
                if upcoming:
                    next_opening = upcoming[0][0].isoformat()
                    break
        return {"status": "known", "is_open": is_open, "opens_at": opens_at,
                "closes_at": closes_at, "timezone": str(tz),
                "next_opening": next_opening, "service_id": service_id,
                "source_id": cal_id, "observed_at": cal["observed_at"],
                "override": cal["override"],
                "evaluation": "current" if today else "scheduled_date"}
    except (ValueError, TypeError, KeyError, OverflowError):
        return _unknown(service_id, cal_id, "unsupported")
