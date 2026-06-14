"""Audit logging for Gemini signal queries."""
from unittest.mock import patch

from src.agents.ledger_utils import (
    record_signal_gemini_query,
    signal_context_coverage,
)
from src import config


def test_signal_context_coverage_counts_sections():
    valid = ["SPY", "QQQ", "TLT"]
    payload = {
        "valid_tickers": valid,
        "technical_data": {"SPY": {}, "QQQ": {}},
        "mc_predictions": {"SPY": {}},
        "news_data": {},
        "signal_learning": "prior lesson",
    }
    cov = signal_context_coverage(payload)
    assert cov["technical_data"]["present"] == 2
    assert cov["technical_data"]["missing"] == ["TLT"]
    assert cov["mc_predictions"]["present"] == 1
    assert cov["has_competition"] is False
    assert cov["signal_learning_chars"] == 12


def test_record_signal_gemini_query_never_raises_when_audit_disabled():
    with patch.object(config, "AUDIT_ENABLED", False):
        record_signal_gemini_query(
            system="internal",
            path="adk_brief",
            query_text="test query",
            payload={},
        )


def test_record_signal_gemini_query_swallows_audit_errors():
    with patch.object(config, "AUDIT_ENABLED", True):
        with patch(
            "src.audit.record_event",
            side_effect=RuntimeError("disk full"),
        ):
            record_signal_gemini_query(
                system="internal",
                path="adk_brief",
                query_text="test query",
                payload={"valid_tickers": ["SPY"]},
            )
