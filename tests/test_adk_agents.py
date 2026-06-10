"""Import smoke tests for ADK agent tree."""
from src.adk.agents.coordinators import build_baseline_root_agent, build_internal_root_agent
from src.adk.agents.signal_agents import build_baseline_signal_agent


def test_baseline_coordinator_has_sub_agents():
    root = build_baseline_root_agent()
    names = [a.name for a in (root.sub_agents or [])]
    assert "baseline_data" in names
    assert "baseline_signal" in names


def test_internal_coordinator_has_sub_agents():
    root = build_internal_root_agent()
    names = [a.name for a in (root.sub_agents or [])]
    assert "internal_data" in names
    assert "internal_signal" in names


def test_baseline_signal_agent_builds():
    agent = build_baseline_signal_agent()
    assert agent.name == "baseline_signal"
