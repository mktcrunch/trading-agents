"""Tests for explicit force/skip_calendar resolution."""
from src.adk.agents.scheduler_callbacks import resolve_skip_calendar


def test_default_no_bypass():
    assert resolve_skip_calendar("Run daily trading workflow.") is False


def test_force_kwarg():
    assert resolve_skip_calendar("Run daily trading workflow.", force=True) is True
    assert resolve_skip_calendar("Run daily trading workflow.", skip_calendar=True) is True


def test_force_string_kwarg():
    assert resolve_skip_calendar("hello", force="true") is True


def test_force_in_message():
    assert resolve_skip_calendar("Run daily trading workflow force.") is True
    assert resolve_skip_calendar("retry overnight internal") is True
