"""Market session calendar gating for overnight workflow."""
from datetime import date
from unittest.mock import MagicMock, patch

from src.market.calendar import (
    check_overnight_trading_session,
    is_trading_session_on_date,
    today_et,
)


def test_is_trading_session_on_date():
    assert is_trading_session_on_date(date(2026, 6, 9), [{"date": "2026-06-09"}])
    assert not is_trading_session_on_date(date(2026, 7, 4), [])


@patch("src.market.calendar.today_et", return_value=date(2026, 6, 13))  # Saturday
def test_skip_weekend(_mock_today):
    allowed, reason = check_overnight_trading_session()
    assert allowed is False
    assert "Weekend" in reason


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
