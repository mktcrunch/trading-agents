"""US equity session calendar helpers (Alpaca)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Tuple

import pytz

from src.logger import setup_logger

logger = setup_logger(__name__)

ET = pytz.timezone("US/Eastern")


def today_et() -> date:
    return datetime.now(ET).date()


def is_trading_session_on_date(
    session_date: date,
    calendar_days: Optional[list],
) -> bool:
    """True when Alpaca calendar lists a session for the given date."""
    return bool(calendar_days)


def check_overnight_trading_session(system: str = "baseline") -> Tuple[bool, str]:
    """
    Whether the overnight workflow should run today.

    The EOD job is scheduled after the regular close. Skip when today has no
    equity session (weekend or exchange holiday per Alpaca calendar).
    """
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
