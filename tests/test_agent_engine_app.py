"""Tests for SchedulerDirectAdkApp stream_query (scheduler StopIteration fix)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from src.adk.agent_engine_app import build_scheduler_direct_app


def test_scheduler_daily_default_respects_calendar():
    mock_agent = object()
    app = build_scheduler_direct_app(mock_agent, system="internal")
    result = {"success": True, "skipped": True, "skip_reason": "Weekend"}

    with patch(
        "src.adk.tools.alpaca_tools.run_daily_trading_workflow",
        new_callable=AsyncMock,
        return_value=result,
    ) as mock_run:
        events = list(
            app.stream_query(
                message="Run daily trading workflow for internal",
                user_id="scheduler",
            )
        )

    mock_run.assert_called_once_with(system="internal", skip_calendar=False)
    assert len(events) == 1


def test_scheduler_daily_force_kwarg_bypasses_calendar():
    mock_agent = object()
    app = build_scheduler_direct_app(mock_agent, system="internal")
    result = {"success": True, "orders_placed": 2}

    with patch(
        "src.adk.tools.alpaca_tools.run_daily_trading_workflow",
        new_callable=AsyncMock,
        return_value=result,
    ) as mock_run:
        events = list(
            app.stream_query(
                message="Run daily trading workflow.",
                user_id="scheduler",
                force=True,
            )
        )

    mock_run.assert_called_once_with(system="internal", skip_calendar=True)
    assert len(events) == 1
    text = events[0]["content"]["parts"][0]["text"]
    assert "orders_placed" in text
