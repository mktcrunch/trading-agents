"""Tests for quant metrics embedded in competition context."""
from unittest.mock import MagicMock, patch

from src.agents.competition_context import (
    build_competition_context,
    build_perspective_quant_view,
    fetch_quant_head_to_head,
    format_quant_learning_block,
    get_competition_snapshot,
)
from src.analytics.performance_metrics import compute_head_to_head_metrics


def _mock_history():
    return [
        {"timestamp": f"2026-06-{10 + i:02d}T21:00:00+00:00", "portfolio_value": v}
        for i, v in enumerate([100_000, 100_100, 100_200, 100_150, 100_300, 100_250])
    ]


def _head_to_head_metrics():
    baseline = _mock_history()
    internal = [
        {"timestamp": f"2026-06-{10 + i:02d}T21:00:00+00:00", "portfolio_value": v}
        for i, v in enumerate([100_000, 100_200, 100_180, 100_400, 100_350, 100_500])
    ]
    return compute_head_to_head_metrics(baseline, internal)


@patch("src.agents.competition_context.AlpacaClient")
def test_fetch_quant_head_to_head(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.get_portfolio_history_series.side_effect = [
        _mock_history(),
        _mock_history(),
    ]

    out = fetch_quant_head_to_head(since_hours=720)
    assert out["metrics"]["observation_days"] == 5
    assert "methodology" in out
    assert "perspectives" in out
    assert out["data_quality"]["status"] == "ok"


def test_perspective_flips_sign_for_baseline():
    metrics = _head_to_head_metrics()
    cmp = metrics["comparison"]
    internal_view = build_perspective_quant_view(metrics, "internal")
    baseline_view = build_perspective_quant_view(metrics, "baseline")

    raw_daily = cmp["daily_delta_pct"]
    assert raw_daily is not None
    assert internal_view["for_you"]["daily_delta_pct"] == raw_daily
    assert baseline_view["for_you"]["daily_delta_pct"] == -raw_daily
    assert internal_view["interpretation"]["daily_delta"] == "you_ahead"
    assert baseline_view["interpretation"]["daily_delta"] == "you_behind"


def test_drawdown_advantage_positive_when_shallower():
    metrics = _head_to_head_metrics()
    cmp = metrics["comparison"]
    raw_dd = cmp["max_drawdown_diff_pct"]
    assert raw_dd is not None

    internal_view = build_perspective_quant_view(metrics, "internal")
    baseline_view = build_perspective_quant_view(metrics, "baseline")
    # Negative internal_minus_baseline => Internal shallower => positive for Internal.
    if raw_dd < 0:
        assert internal_view["for_you"]["drawdown_advantage_pp"] > 0
        assert baseline_view["for_you"]["drawdown_advantage_pp"] < 0


def test_perspective_unavailable_when_no_paired_days():
    metrics = compute_head_to_head_metrics([], [])
    view = build_perspective_quant_view(metrics, "baseline")
    assert view["status"] == "unavailable"
    assert view["for_you"] is None


@patch("src.agents.competition_context._attach_quant_head_to_head")
@patch("src.agents.competition_context._account_snapshot")
def test_competition_context_includes_quant(mock_snap, mock_attach):
    mock_snap.return_value = {
        "system": "baseline",
        "label": "Baseline",
        "portfolio_value": 100_000,
        "cash": 50_000,
        "equity": 100_000,
        "pnl": 0,
        "pnl_pct": 0,
        "position_count": 0,
        "positions": [],
    }
    mock_attach.side_effect = lambda ctx, **_: {
        **ctx,
        "quant_head_to_head": {"for_you": {"for_you": {"daily_delta_pct": -0.1}}},
    }

    ctx = build_competition_context("baseline")
    assert "quant_head_to_head" in ctx
    mock_attach.assert_called_once()


@patch("src.agents.competition_context.fetch_quant_head_to_head")
def test_format_quant_learning_block_uses_for_you(mock_fetch):
    metrics = _head_to_head_metrics()
    mock_fetch.return_value = {
        "metrics": metrics,
        "perspectives": {
            "internal": build_perspective_quant_view(metrics, "internal"),
            "baseline": build_perspective_quant_view(metrics, "baseline"),
        },
    }
    block = format_quant_learning_block("internal")
    assert "HEAD-TO-HEAD QUANT" in block
    assert "you_ahead" in block
    assert "Internal − Baseline" in block

    baseline_block = format_quant_learning_block("baseline")
    assert "you_behind" in baseline_block or "you_ahead" in baseline_block
