"""Base stop planner merge and parsing."""
import json

from src.risk.base_stop_planner import BaseStopPlanner, merge_hybrid_base_stop
from src.risk.planner_parse import parse_planner_object


def test_parse_planner_object_single():
    text = json.dumps({"stop_loss_threshold": -0.012, "rationale": "Tight stop."})
    parsed = parse_planner_object(text)
    assert parsed["stop_loss_threshold"] == -0.012


def test_merge_hybrid_base_stop_tighter_wins():
    scripted = -0.01
    llm_plan = {"stop_loss_threshold": -0.008, "rationale": "Tighten"}
    effective, policy = merge_hybrid_base_stop(scripted, llm_plan)
    assert effective == -0.008
    assert "hybrid" in policy


def test_merge_hybrid_base_stop_scripted_floor():
    scripted = -0.018
    llm_plan = {"stop_loss_threshold": -0.025, "rationale": "Looser LLM"}
    effective, policy = merge_hybrid_base_stop(scripted, llm_plan)
    assert effective == -0.018
    assert "scripted(-1.8%)" in policy


def test_scripted_base_stop_uses_atr_when_available():
    from datetime import datetime

    from src.risk.risk_monitor import RiskMonitor
    from src.models.position import Position

    monitor = RiskMonitor(system="internal")
    pos = Position(
        ticker="QQQ",
        qty=-11,
        avg_entry_price=500.0,
        current_price=503.0,
        entry_date=datetime.now(),
    )
    # ATR=5 → dist=7.5 → short stop at 507.5 → return (500-507.5)/500 = -0.015
    monitor._get_atr = lambda _t: 5.0
    threshold, policy = monitor._scripted_base_stop_threshold("QQQ", pos)
    assert threshold == -0.015
    assert policy == "atr_1.5x(-1.5%)"


def test_scripted_base_stop_falls_back_to_fixed_without_atr():
    from datetime import datetime

    from src.risk.risk_monitor import RiskMonitor
    from src.models.position import Position

    monitor = RiskMonitor(system="internal")
    pos = Position(
        ticker="QQQ",
        qty=10,
        avg_entry_price=500.0,
        current_price=495.0,
        entry_date=datetime.now(),
    )
    monitor._get_atr = lambda _t: None
    threshold, policy = monitor._scripted_base_stop_threshold("QQQ", pos)
    assert threshold == -0.01
    assert policy == "fixed_-1.0%"


def test_baseline_prompt_no_fixed_one_percent():
    planner = BaseStopPlanner(system="baseline")
    prompt = planner._build_baseline_prompt(
        "SPY", 500.0, 495.0, -0.01, 10, 0.012, 0.02
    )
    assert "PURE LLM base stops" in prompt
    assert "no fixed -1% rule" in prompt


def test_internal_prompt_includes_scripted_floor():
    planner = BaseStopPlanner(system="internal")
    prompt = planner._build_internal_prompt(
        "SPY", 500.0, 495.0, -0.01, 10, 0.012, 0.02, -0.01, {"ai_estimate": {}}
    )
    assert "Scripted base-stop floor" in prompt
    assert "-1.0%" in prompt
