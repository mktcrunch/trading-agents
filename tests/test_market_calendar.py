"""Market session calendar gating for overnight workflow."""
from datetime import date
from unittest.mock import MagicMock, patch

from src.market.calendar import (
    check_chase_trading_session,
    check_overnight_trading_session,
    is_trading_session_on_date,
    prior_session_close_cutoff_et,
    today_et,
    ET,
)
from datetime import datetime


def test_is_trading_session_on_date():
    assert is_trading_session_on_date(date(2026, 6, 9), [{"date": "2026-06-09"}])
    assert not is_trading_session_on_date(date(2026, 7, 4), [])


@patch("src.market.calendar.today_et", return_value=date(2026, 6, 13))  # Saturday
def test_skip_weekend(_mock_today):
    allowed, reason = check_overnight_trading_session()
    assert allowed is False
    assert "Weekend" in reason


@patch("src.market.calendar.today_et", return_value=date(2026, 6, 13))  # Saturday
def test_skip_calendar_bypasses_weekend(_mock_today):
    allowed, reason = check_overnight_trading_session(skip_calendar=True)
    assert allowed is True
    assert "bypassed" in reason


@patch("src.market.calendar.today_et", return_value=date(2026, 6, 12))  # Friday
@patch("src.apis.alpaca_client.AlpacaClient")
def test_allow_regular_friday(mock_client_cls, _mock_today):
    mock_client_cls.return_value.get_calendar.return_value = [MagicMock()]
    allowed, reason = check_overnight_trading_session()
    assert allowed is True
    assert "2026-06-12" in reason


@patch("src.market.calendar.today_et", return_value=date(2026, 7, 3))  # Independence Day observed
@patch("src.apis.alpaca_client.AlpacaClient")
def test_skip_holiday(mock_client_cls, _mock_today):
    mock_client_cls.return_value.get_calendar.return_value = []
    allowed, reason = check_overnight_trading_session()
    assert allowed is False
    assert "holiday" in reason.lower()


@patch("src.market.calendar.today_et", return_value=date(2026, 6, 19))  # Juneteenth
@patch("src.apis.alpaca_client.AlpacaClient")
def test_chase_skips_holiday(mock_client_cls, _mock_today):
    mock_client_cls.return_value.get_calendar.return_value = []
    allowed, reason = check_chase_trading_session()
    assert allowed is False
    assert "holiday" in reason.lower()


@patch("src.apis.alpaca_client.AlpacaClient")
def test_prior_session_close_uses_last_trading_day_not_calendar_friday(mock_client_cls):
    """Mon Jun 22 open after Fri Jun 19 holiday → look back to Thu Jun 18 close."""
    thu = MagicMock()
    thu.date = date(2026, 6, 18)
    mock_client_cls.return_value.get_calendar.return_value = [thu]

    now_et = ET.localize(datetime(2026, 6, 22, 9, 35))
    cutoff = prior_session_close_cutoff_et(now_et, system="baseline")

    assert cutoff == ET.localize(datetime(2026, 6, 18, 16, 0))
