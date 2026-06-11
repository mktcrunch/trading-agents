"""Signal ledger response parsing."""
from src.agents.ledger_utils import (
    PORTFOLIO_TICKER,
    parse_signal_ledger_response,
)


def test_parse_wrapper_with_no_action_rationale():
    text = """{
      "decisions": [],
      "no_action_rationale": "Ahead on leaderboard; staying in cash."
    }"""
    result = parse_signal_ledger_response(text, ["SPY", "QQQ"])
    assert result.decisions == []
    assert "cash" in result.no_action_rationale


def test_parse_legacy_array():
    text = """[{"action": "BUY", "ticker": "SPY", "size_pct": 0.05, "confidence": 0.7,
      "rationale": "test", "invalidation": "x", "competitive_note": "y"}]"""
    result = parse_signal_ledger_response(text, ["SPY"])
    assert len(result.decisions) == 1
    assert result.decisions[0].ticker == "SPY"


def test_portfolio_ticker_constant():
    assert PORTFOLIO_TICKER == "PORTFOLIO"
