"""Tests for Agent Engine stream_query patch (force kwarg + direct daily)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import src.adk.agent_engine_app as agent_engine_app
from src.adk.agent_engine_app import install_agent_engine_stream_query_patch


def test_patch_strips_force_and_runs_direct_daily():
    agent_engine_app._PATCHED_CLASSES.clear()
    install_agent_engine_stream_query_patch("internal")
    from vertexai import agent_engines

    adk_app = agent_engines.AdkApp(agent=MagicMock())
    adk_app._tmpl_attrs = {
        "runner": MagicMock(),
        "app_name": "twin_ledger_internal",
    }
    result = {"success": True, "orders_placed": 1, "skip_calendar": True}

    with patch(
        "src.adk.tools.alpaca_tools.run_daily_trading_workflow",
        new_callable=AsyncMock,
        return_value=result,
    ) as mock_run:
        events = list(
            adk_app.stream_query(
                message="Run daily trading workflow.",
                user_id="manual",
                force=True,
            )
        )

    mock_run.assert_called_once_with(system="internal", skip_calendar=True)
    assert len(events) == 1
    adk_app._tmpl_attrs["runner"].run.assert_not_called()


def test_patch_default_respects_calendar():
    agent_engine_app._PATCHED_CLASSES.clear()
    install_agent_engine_stream_query_patch("internal")
    from vertexai import agent_engines

    adk_app = agent_engines.AdkApp(agent=MagicMock())
    adk_app._tmpl_attrs = {"app_name": "twin_ledger_internal"}
    result = {"success": True, "skipped": True}

    with patch(
        "src.adk.tools.alpaca_tools.run_daily_trading_workflow",
        new_callable=AsyncMock,
        return_value=result,
    ) as mock_run:
        list(
            adk_app.stream_query(
                message="Run daily trading workflow.",
                user_id="scheduler",
            )
        )

    mock_run.assert_called_once_with(system="internal", skip_calendar=False)
