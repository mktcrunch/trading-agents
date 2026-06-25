"""Tests for dashboard performance metrics tool."""
import sys
import types
from unittest.mock import patch

# dashboard_tools is imported without pulling google.adk via src.adk.tools.__init__
if "google.adk.tools.function_tool" not in sys.modules:
    _google = types.ModuleType("google")
    _adk = types.ModuleType("google.adk")
    _tools = types.ModuleType("google.adk.tools")
    _function_tool = types.ModuleType("google.adk.tools.function_tool")
    _function_tool.FunctionTool = lambda fn: fn
    _tools.function_tool = _function_tool
    _adk.tools = _tools
    _google.adk = _adk
    sys.modules["google"] = _google
    sys.modules["google.adk"] = _adk
    sys.modules["google.adk.tools"] = _tools
    sys.modules["google.adk.tools.function_tool"] = _function_tool

from src.adk.tools.dashboard_tools import (
    format_performance_metrics_report,
    get_performance_metrics,
)
from src.agents.competition_context import (
    PERFORMANCE_METRICS_METHODOLOGY,
    build_perspective_quant_view,
)


_SAMPLE_METRICS = {
    "observation_days": 12,
    "latest_date": "2026-06-24",
    "baseline": {
        "total_return_pct": -0.02,
        "daily_return_pct": -0.12,
        "sharpe": -0.07,
        "max_drawdown_pct": 0.99,
    },
    "internal": {
        "total_return_pct": 0.18,
        "daily_return_pct": -0.02,
        "sharpe": 0.81,
        "max_drawdown_pct": 0.56,
    },
    "comparison": {
        "sign_convention": "internal_minus_baseline",
        "formula": "internal_value - baseline_value",
        "total_return_diff_pct": 0.2,
        "daily_delta_pct": 0.1,
        "mean_daily_alpha_pct": 0.017,
        "sharpe_diff": 0.88,
        "max_drawdown_diff_pct": -0.43,
        "internal_minus_baseline": {
            "excess_return_pct": 0.2,
            "daily_delta_pct": 0.1,
            "sharpe_diff": 0.88,
            "max_drawdown_diff_pct": -0.43,
            "mean_daily_alpha_pct": 0.017,
        },
        "significance": {
            "total_return_diff": {
                "p_value": 0.87,
                "significant_95": False,
                "days_remaining_95": 100,
            },
            "daily_alpha": {
                "p_value": 0.88,
                "significant_95": False,
                "days_remaining_95": 200,
            },
            "sharpe_diff": {
                "p_value": 0.9,
                "significant_95": False,
                "days_remaining_95": 150,
            },
            "max_drawdown_diff": {
                "p_value": 0.34,
                "significant_95": False,
                "days_remaining_95": 39,
            },
        },
    },
}


@patch("src.agents.competition_context.get_competition_snapshot")
@patch("src.agents.competition_context.fetch_quant_head_to_head")
def test_get_performance_metrics_returns_methodology(mock_fetch, mock_live):
    perspectives = {
        "baseline": build_perspective_quant_view(_SAMPLE_METRICS, "baseline"),
        "internal": build_perspective_quant_view(_SAMPLE_METRICS, "internal"),
    }
    mock_fetch.return_value = {
        "since_hours": 720,
        "history_points": {"baseline": 12, "internal": 12},
        "data_quality": {"status": "ok", "paired_days": 12},
        "metrics": _SAMPLE_METRICS,
        "perspectives": perspectives,
    }
    mock_live.return_value = {"leaderboard": {"leader": "internal", "gap_usd": 333.98}}

    out = get_performance_metrics(hours=720, perspective="baseline")

    assert out["success"] is True
    assert out["metrics"]["comparison"]["total_return_diff_pct"] == 0.2
    assert out["methodology"] == PERFORMANCE_METRICS_METHODOLOGY
    assert out["for_you"]["perspective"] == "baseline"
    assert out["for_you"]["for_you"]["excess_return_pct"] == -0.2
    assert "Excess return" in out["report"]
    assert "Perspective:" in out["report"]


def test_format_performance_metrics_report_empty():
    assert "unavailable" in format_performance_metrics_report(None).lower()


def test_format_performance_metrics_report_includes_significance():
    report = format_performance_metrics_report(
        _SAMPLE_METRICS,
        {"leaderboard": {"leader": "internal", "gap_usd": 100.0}},
    )
    assert "not significant" in report
    assert "39 more paired days" in report
    assert "Sign convention" in report
