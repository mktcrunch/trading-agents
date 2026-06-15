"""Trailing planner JSON parsing (single object vs decisions array)."""
import json

from src.risk.trailing_planner import (
    TrailingPlanner,
    format_position_side_line,
    parse_trailing_plan_response,
    position_side_label,
)


def test_parse_single_object():
    text = json.dumps({
        "activation_threshold": 0.008,
        "profit_lock_fraction": 0.75,
        "rationale": "Tighten lock while MC confidence remains high.",
    })
    parsed = parse_trailing_plan_response(text)
    assert parsed["activation_threshold"] == 0.008
    assert parsed["profit_lock_fraction"] == 0.75
    assert "MC confidence" in parsed["rationale"]


def test_parse_one_element_array():
    text = json.dumps([{
        "activation_threshold": 0.012,
        "profit_lock_fraction": 0.58,
        "rationale": "Baseline volatility warrants wider activation.",
    }])
    parsed = parse_trailing_plan_response(text)
    assert parsed["activation_threshold"] == 0.012
    assert "volatility" in parsed["rationale"]


def test_parse_decisions_wrapper():
    text = json.dumps({
        "decisions": [{
            "activation_threshold": 0.01,
            "profit_lock_fraction": 0.8,
            "rationale": "Wrapped format still works.",
        }]
    })
    parsed = parse_trailing_plan_response(text)
    assert parsed["profit_lock_fraction"] == 0.8


def test_bare_object_not_dropped_like_ledger_parser():
    """Regression: parse_ledger_response returned [] for bare objects."""
    from src.agents.ledger_utils import parse_ledger_response

    text = json.dumps({
        "activation_threshold": 0.009,
        "profit_lock_fraction": 0.72,
        "rationale": "Should not default to 1%/70%.",
    })
    assert parse_ledger_response(text) == []
    assert parse_trailing_plan_response(text)["activation_threshold"] == 0.009


def test_position_side_label():
    assert position_side_label(13) == "LONG"
    assert position_side_label(-188) == "SHORT"


def test_baseline_prompt_includes_short_side_and_pnl_help():
    planner = TrailingPlanner(system="baseline")
    prompt = planner._build_baseline_prompt(
        "XLF", 53.81, 53.85, -0.00065, -188, 0.013, 0.036
    )
    assert "Side: SHORT (qty -188)" in prompt
    assert "positive = winning, negative = losing" in prompt
    assert "Base stop-loss is handled separately" in prompt


def test_internal_prompt_includes_long_side():
    planner = TrailingPlanner(system="internal")
    prompt = planner._build_internal_prompt(
        "SPY", 752.0, 756.0, 0.0053, 13, 0.008, 0.01, {"ai_estimate": {}}
    )
    assert "Side: LONG (qty 13)" in prompt
    assert "scripted ATR/fixed rules" in prompt
