"""Tests for no-action audit rationale finalization."""
from src.agents.ledger_utils import finalize_signal_ledger, emit_signal_ledger_audit
from src.agents.ledger_utils import SignalLedgerResult
from src.models.trading_decision import TradingDecision


class _StubAgent:
    def __init__(self):
        self.events = []

    def log_action(self, action, data=None, event_type="agent_action"):
        self.events.append({"action": action, "data": data or {}, "event_type": event_type})


def test_finalize_prefers_explicit_no_action_rationale():
    result = finalize_signal_ledger(
        SignalLedgerResult(decisions=[], no_action_rationale="Staying flat ahead of Fed.")
    )
    assert result.no_action_rationale == "Staying flat ahead of Fed."


def test_emit_audit_portfolio_hold_uses_no_action_rationale():
    agent = _StubAgent()
    emit_signal_ledger_audit(
        agent,
        SignalLedgerResult(decisions=[], no_action_rationale="No setups tonight."),
        {"leaderboard": {"your_rank": 1, "status": "ahead", "value_gap_usd": 100}},
    )
    ledger_events = [e for e in agent.events if e["event_type"] == "ledger_decision"]
    assert len(ledger_events) == 1
    assert ledger_events[0]["data"]["rationale"] == "No setups tonight."
    assert ledger_events[0]["data"]["ticker"] == "PORTFOLIO"
