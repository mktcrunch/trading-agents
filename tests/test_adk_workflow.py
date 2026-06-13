"""Tests for ADK Workflow routing and helpers."""
from unittest.mock import AsyncMock, patch

import pytest

from src.adk.tools import (
    baseline_data_tools,
    baseline_signal_tools,
    coordinator_execution_tools,
    internal_data_tools,
    internal_signal_tools,
)
from src import config
from src.adk.workflows.common import ledger_to_state, parse_adk_signal_output
from src.agents.ledger_utils import SignalLedgerResult
from src.models.trading_decision import TradingDecision


def test_signal_tools_exclude_execution():
    data_names = {t.name for t in baseline_data_tools()}
    signal_names = {t.name for t in baseline_signal_tools()}
    exec_names = {t.name for t in coordinator_execution_tools()}

    assert "run_daily_trading_workflow" in exec_names
    assert "execute_trading_decisions" in exec_names
    assert "run_daily_trading_workflow" not in data_names
    assert "run_daily_trading_workflow" not in signal_names
    assert signal_names == data_names


def test_internal_signal_tools_exclude_proprietary_fetch():
    internal_names = {t.name for t in internal_signal_tools()}
    data_names = {t.name for t in internal_data_tools()}
    baseline_names = {t.name for t in baseline_signal_tools()}
    assert internal_names == baseline_names
    assert "get_marketcrunch_predictions" in data_names
    assert "get_databento_features" in data_names
    assert "get_marketcrunch_predictions" not in internal_names
    assert "get_databento_features" not in internal_names


def test_parse_adk_signal_output_from_dict():
    ticker = config.TICKER_UNIVERSE[0]
    ledger = parse_adk_signal_output(
        {
            "decisions": [
                {"ticker": ticker, "action": "HOLD", "confidence": 0.5, "rationale": "test"},
            ],
            "no_action_rationale": "none",
        }
    )
    assert len(ledger.decisions) == 1
    assert ledger.decisions[0].ticker == ticker


def test_ledger_to_state_roundtrip():
    ticker = config.TICKER_UNIVERSE[0]
    decision = TradingDecision(ticker=ticker, action="BUY", confidence=0.8, rationale="x")
    ledger = SignalLedgerResult(decisions=[decision], no_action_rationale="go")
    state = ledger_to_state(ledger)
    assert state["no_action_rationale"] == "go"
    assert state["decisions"][0]["ticker"] == ticker


@pytest.mark.asyncio
async def test_daily_pipeline_routes_to_adk_workflow():
    with patch("src.config.USE_ADK_WORKFLOW", True):
        with patch(
            "src.adk.workflows.baseline_daily.run_baseline_daily_adk",
            new_callable=AsyncMock,
            return_value={"success": True, "pipeline": "adk_workflow"},
        ) as mock_baseline:
            from src.adk.workflows.daily_pipeline import run_daily_trading_pipeline

            result = await run_daily_trading_pipeline("baseline")
            mock_baseline.assert_awaited_once()
            assert result["pipeline"] == "adk_workflow"


@pytest.mark.asyncio
async def test_daily_pipeline_routes_internal_adk():
    with patch("src.config.USE_ADK_WORKFLOW", True):
        with patch(
            "src.adk.workflows.internal_daily.run_internal_daily_adk",
            new_callable=AsyncMock,
            return_value={"success": True, "pipeline": "adk_workflow", "system": "internal"},
        ) as mock_internal:
            from src.adk.workflows.daily_pipeline import run_daily_trading_pipeline

            result = await run_daily_trading_pipeline("internal")
            mock_internal.assert_awaited_once()
            assert result["system"] == "internal"
