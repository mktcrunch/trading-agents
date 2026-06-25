"""Format learning memory for agent prompts (Tier 1 + 2)."""
from __future__ import annotations

from typing import Any, Dict, List

from src.learning.analyzer import analyze_risk_outcomes
from src.learning.store import load_learning


def _format_decision_lines(decisions: List[Dict[str, Any]], limit: int = 8) -> str:
    if not decisions:
        return "- (no scored decisions yet)"
    lines = []
    for d in decisions[-limit:]:
        ret = d.get("forward_return_pct")
        ret_s = f"{ret:+.2f}%" if ret is not None else "n/a"
        lines.append(
            f"- [{d.get('timestamp', '')[:10]}] {d.get('action')} {d.get('ticker')} "
            f"→ {d.get('outcome', '?')} ({ret_s}): {(d.get('rationale') or '')[:80]}"
        )
    return "\n".join(lines)


def _format_ticker_stats(stats: Dict[str, Any]) -> str:
    if not stats:
        return "- (no ticker stats yet)"
    lines = []
    for ticker, s in sorted(stats.items(), key=lambda kv: kv[1].get("avg_return_pct", 0)):
        if not s.get("count"):
            continue
        lines.append(
            f"- {ticker}: {s.get('wins', 0)}W/{s.get('losses', 0)}L "
            f"avg {s.get('avg_return_pct', 0):+.2f}%"
        )
    return "\n".join(lines) or "- (no ticker stats yet)"


def build_signal_learning_block(system: str) -> str:
    """Text block injected into Signal agent prompts."""
    state = load_learning(system, "signal")
    if not state.get("updated_at") and not state.get("recent_decisions"):
        return ""

    score = state.get("scorecard") or {}
    logged = score.get("decisions_logged", 0)
    no_action = score.get("no_action_logged", 0)
    pending = score.get("decisions_pending", 0)
    no_action_sessions = state.get("no_action_sessions") or []
    bad = state.get("bad_patterns") or []
    do_more = state.get("do_more") or []

    block = f"""LEARNING FROM RECENT OUTCOMES (last {state.get('lookback_days', 7)} days — do NOT repeat mistakes):

Summary: {state.get('lessons_learned') or 'Building history from audit trail.'}
Decisions logged: {logged} · scored: {score.get('decisions_scored', 0)} · no-action nights: {no_action} · pending next-day: {pending}
Win rate (scored only): {score.get('win_rate_pct', 'n/a')}% ({score.get('wins', 0)}W / {score.get('losses', 0)}L)

Ticker performance:
{_format_ticker_stats(state.get('ticker_stats') or {})}

Recent scored decisions:
{_format_decision_lines(state.get('recent_decisions') or [])}"""

    from src.agents.competition_context import format_quant_learning_block

    quant_block = format_quant_learning_block(system)
    if quant_block:
        block += f"\n\n{quant_block}"

    if no_action_sessions:
        lines = []
        for s in no_action_sessions[-3:]:
            lines.append(
                f"- [{(s.get('timestamp') or '')[:10]}] PORTFOLIO HOLD — "
                f"{(s.get('rationale') or '')[:120]}"
            )
        block += "\n\nRecent no-action rationales:\n" + "\n".join(lines)

    if bad:
        block += "\n\nAvoid repeating:\n" + "\n".join(f"- {p}" for p in bad[:5])
    if do_more:
        block += "\n\nDo more of:\n" + "\n".join(f"- {p}" for p in do_more[:5])

    block += "\n\nUse this memory to improve today's decisions — reduce churn on losing setups."
    return block


def build_risk_learning_block(system: str, live_hours: int = 48) -> str:
    """Text block injected into Risk trailing planner (Tier 3 + live audit)."""
    state = load_learning(system, "risk")
    live = analyze_risk_outcomes(system, lookback_days=max(2, live_hours // 24))

    exits = (live.get("recent_exits") or []) + (state.get("recent_exits") or [])
    seen = set()
    unique_exits = []
    for ex in reversed(exits):
        key = (ex.get("timestamp"), ex.get("ticker"), ex.get("event_type"))
        if key in seen:
            continue
        seen.add(key)
        unique_exits.append(ex)
    unique_exits = list(reversed(unique_exits))[-10:]

    score = state.get("scorecard") or {}
    live_score = live.get("scorecard") or {}

    exit_lines = []
    for ex in unique_exits:
        ret = ex.get("return_pct")
        ret_s = f"{ret:+.2f}%" if ret is not None else "n/a"
        exit_lines.append(
            f"- {ex.get('ticker')} {ex.get('event_type')}: {ret_s} — {(ex.get('reason') or '')[:60]}"
        )
    if not exit_lines:
        exit_lines = ["- (no recent exits logged)"]

    bad = state.get("bad_patterns") or []
    do_more = state.get("do_more") or []

    block = f"""RISK LEARNING (recent stop/EOD outcomes — adjust trailing accordingly):

{state.get('lessons_learned') or 'Building exit history from audit trail.'}
Stored: {score.get('stop_exits', 0)} stops, avg exit {score.get('avg_exit_return_pct', 'n/a')}%
Live ({live_hours}h): {live_score.get('stop_exits', 0)} stops, {live_score.get('held_count', 0)} held

Recent exits:
{chr(10).join(exit_lines)}"""

    if bad:
        block += "\n\nTrailing adjustments — avoid:\n" + "\n".join(f"- {p}" for p in bad[:4])
    if do_more:
        block += "\n\nTrailing adjustments — prefer:\n" + "\n".join(f"- {p}" for p in do_more[:4])

    return block
