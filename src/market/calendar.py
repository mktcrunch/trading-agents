"""US equity session calendar helpers (Alpaca)."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Optional, Tuple

import pytz

from src.logger import setup_logger

logger = setup_logger(__name__)

ET = pytz.timezone("US/Eastern")
SESSION_CLOSE_HOUR = 16


def today_et() -> date:
    return datetime.now(ET).date()


def is_trading_session_on_date(
    session_date: date,
    calendar_days: Optional[list],
) -> bool:
    """True when Alpaca calendar lists a session for the given date."""
    return bool(calendar_days)


def check_overnight_trading_session(
    system: str = "baseline",
    *,
    skip_calendar: bool = False,
) -> Tuple[bool, str]:
    """
    Whether the overnight workflow should run today.

    The EOD job is scheduled after the regular close. Skip when today has no
    equity session (weekend or exchange holiday per Alpaca calendar).

    ``skip_calendar=True`` bypasses this gate only when the caller explicitly opts in
    (``force`` / ``skip_calendar`` on streamQuery, or force/retry in the message).
    """
    if skip_calendar:
        return True, "calendar_bypassed (triggered run)"

    session_date = today_et()

    if session_date.weekday() >= 5:
        return False, f"Weekend ({session_date.strftime('%A %Y-%m-%d')}) — no equity session"

    try:
        from alpaca.trading.requests import GetCalendarRequest
        from src.apis.alpaca_client import AlpacaClient

        client = AlpacaClient(system=system)
        days = client.get_calendar(
            GetCalendarRequest(start=session_date, end=session_date)
        )
    except Exception as e:
        logger.warning(
            f"Alpaca calendar lookup failed ({e}); allowing overnight workflow"
        )
        return True, "calendar_unavailable"

    if not is_trading_session_on_date(session_date, days):
        return False, f"Market holiday — no equity session on {session_date.isoformat()}"

    return True, f"Equity session on {session_date.isoformat()}"


def check_chase_trading_session(
    system: str = "baseline",
    *,
    skip_calendar: bool = False,
) -> Tuple[bool, str]:
    """
    Whether post-open chase should run today.

    Same session gate as overnight: skip weekends and exchange holidays.
    """
    ok, reason = check_overnight_trading_session(
        system=system,
        skip_calendar=skip_calendar,
    )
    if not ok:
        return False, reason
    return True, reason


def _overnight_lookback_cutoff_fallback(now_et: datetime) -> datetime:
    """Weekday calendar fallback when Alpaca session history is unavailable."""
    if now_et.weekday() == 0:
        return (now_et - timedelta(days=3)).replace(
            hour=SESSION_CLOSE_HOUR, minute=0, second=0, microsecond=0
        )
    return (now_et - timedelta(days=1)).replace(
        hour=SESSION_CLOSE_HOUR, minute=0, second=0, microsecond=0
    )


def _calendar_entry_date(entry: Any) -> date:
    session_date = getattr(entry, "date", None)
    if isinstance(session_date, date):
        return session_date
    if isinstance(session_date, str):
        return date.fromisoformat(session_date)
    raise TypeError(f"Unexpected calendar entry date: {session_date!r}")


def prior_session_close_cutoff_et(
    now_et: datetime,
    system: str = "baseline",
) -> datetime:
    """
    4:00 PM ET on the most recent equity session strictly before ``now_et``'s date.

    Uses the Alpaca exchange calendar so holiday weeks (e.g. Thu close → Mon open)
    still include Thursday overnight orders.
    """
    today = now_et.date()
    window_start = today - timedelta(days=14)
    window_end = today - timedelta(days=1)

    try:
        from alpaca.trading.requests import GetCalendarRequest
        from src.apis.alpaca_client import AlpacaClient

        days = AlpacaClient(system=system).get_calendar(
            GetCalendarRequest(start=window_start, end=window_end)
        )
        if days:
            session_date = _calendar_entry_date(days[-1])
            return ET.localize(
                datetime.combine(session_date, time(SESSION_CLOSE_HOUR, 0))
            )
    except Exception as e:
        logger.warning(
            f"Alpaca prior-session lookup failed ({e}); using weekday fallback"
        )

    return _overnight_lookback_cutoff_fallback(now_et)
