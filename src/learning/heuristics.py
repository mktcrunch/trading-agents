"""Rule-based lessons per agent role (before optional LLM polish)."""
from __future__ import annotations

from typing import Any, Dict, List


def _base(lessons: str, bad: List[str], do_more: List[str]) -> Dict[str, Any]:
    return {
        "lessons_learned": lessons,
        "bad_patterns": bad[:5],
        "do_more": do_more[:5],
    }


def heuristic_for_role(role: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    if role == "signal":
        return _heuristic_signal(analysis)
    if role == "risk":
        return _heuristic_risk(analysis)
    if role == "coordinator":
        return _heuristic_coordinator(analysis)
    if role == "data":
        return _heuristic_data(analysis)
    if role == "execution":
        return _heuristic_execution(analysis)
    if role == "monitor":
        return _heuristic_monitor(analysis)
    if role == "discovery":
        return _heuristic_discovery(analysis)
    return _base("No heuristics for this role.", [], [])


def _heuristic_signal(analysis: Dict[str, Any]) -> Dict[str, Any]:
    score = analysis.get("scorecard") or {}
    logged = score.get("decisions_logged", 0)
    pending = score.get("decisions_pending", 0)
    bad: List[str] = []
    do_more: List[str] = []
    wr = score.get("win_rate_pct")

    if logged == 0:
        bad.append(
            f"No ledger decisions in the last {analysis.get('lookback_days', 7)} days — "
            "check overnight scheduler."
        )
    elif score.get("decisions_scored", 0) == 0 and pending > 0:
        do_more.append(
            f"{logged} decision(s) logged; {pending} pending next-day outcome scoring (normal for same-day)."
        )
    elif wr is not None and wr < 45:
        bad.append(f"Win rate {wr}% on scored trades — tighten entry filters.")
    elif wr is not None and wr >= 55:
        do_more.append(f"Win rate {wr}% — keep discipline on sizing.")

    if logged == 0:
        lessons = "No signal decisions in audit log."
    elif score.get("decisions_scored", 0) == 0:
        lessons = f"{logged} logged; {pending} pending outcome score (need ≥1 trading day)."
    else:
        lessons = (
            f"Scored {score.get('decisions_scored', 0)}/{logged}: "
            f"{score.get('wins', 0)}W / {score.get('losses', 0)}L."
        )
    return _base(lessons, bad, do_more)


def _heuristic_risk(analysis: Dict[str, Any]) -> Dict[str, Any]:
    score = analysis.get("scorecard") or {}
    stops = score.get("stop_exits", 0)
    bad: List[str] = []
    do_more: List[str] = []
    if stops == 0:
        do_more.append("No stop exits — normal if no open positions or no triggers yet.")
    if score.get("profitable_stops", 0) and stops and score["profitable_stops"] / stops > 0.5:
        bad.append("Many stops exited in profit — trailing may be too tight.")
    if score.get("held_count", 0) >= 2:
        do_more.append(f"Prediction gate held {score['held_count']} exits — review deferrals.")
    lessons = (
        f"{stops} stop exits, {score.get('eod_exits', 0)} EOD, {score.get('held_count', 0)} held."
    )
    avg = score.get("avg_exit_return_pct")
    if avg is not None:
        lessons += f" Avg exit {avg:+.2f}%."
    return _base(lessons, bad, do_more)


def _heuristic_coordinator(analysis: Dict[str, Any]) -> Dict[str, Any]:
    score = analysis.get("scorecard") or {}
    started = score.get("jobs_started", 0)
    failed = score.get("jobs_failed", 0)
    bad: List[str] = []
    do_more: List[str] = []
    if started == 0:
        bad.append("No scheduler jobs started — verify Cloud Scheduler / cron path.")
    if failed:
        bad.append(f"{failed} job(s) failed completion — inspect trace logs.")
    if started and not failed:
        do_more.append(f"{score.get('jobs_ok', 0)} jobs completed cleanly.")
    lessons = f"Coordinator routed {started} job(s); {score.get('jobs_completed', 0)} completed."
    return _base(lessons, bad, do_more)


def _heuristic_data(analysis: Dict[str, Any]) -> Dict[str, Any]:
    score = analysis.get("scorecard") or {}
    n = score.get("data_events", 0)
    bad: List[str] = []
    do_more: List[str] = []
    if n == 0:
        bad.append("No data agent events — check overnight data fetch step.")
    else:
        do_more.append(f"{n} data fetch/enrichment event(s) logged.")
    if score.get("enrichment_runs"):
        do_more.append(f"{score['enrichment_runs']} vendor enrichment run(s).")
    if score.get("mc_context_loads"):
        do_more.append(f"{score['mc_context_loads']} MC Internal context load(s).")
    lessons = f"Data agent: {n} events in window."
    return _base(lessons, bad, do_more)


def _heuristic_execution(analysis: Dict[str, Any]) -> Dict[str, Any]:
    score = analysis.get("scorecard") or {}
    placed = score.get("orders_placed", 0)
    skipped = score.get("orders_skipped", 0)
    chased = score.get("orders_chased", 0)
    bad: List[str] = []
    do_more: List[str] = []
    if placed == 0 and skipped == 0:
        bad.append("No orders placed or skipped — signal may be all HOLD or execution blocked.")
    if skipped > placed and skipped >= 3:
        bad.append(f"High skip rate ({skipped} skipped vs {placed} placed) — review limits/chase gates.")
    if chased:
        do_more.append(f"{chased} post-open chase(s) executed.")
    lessons = f"Execution: {placed} placed, {skipped} skipped, {chased} chased."
    return _base(lessons, bad, do_more)


def _heuristic_monitor(analysis: Dict[str, Any]) -> Dict[str, Any]:
    score = analysis.get("scorecard") or {}
    snaps = score.get("snapshots", 0)
    delta = score.get("portfolio_delta_pct")
    bad: List[str] = []
    do_more: List[str] = []
    if snaps == 0:
        bad.append("No portfolio snapshots — monitor step may not be running after daily jobs.")
    else:
        do_more.append(f"{snaps} EOD snapshot(s) captured.")
    lessons = f"Monitor logged {snaps} snapshot(s)."
    if delta is not None:
        lessons += f" Portfolio change over window: {delta:+.2f}%."
    return _base(lessons, bad, do_more)


def _heuristic_discovery(analysis: Dict[str, Any]) -> Dict[str, Any]:
    score = analysis.get("scorecard") or {}
    runs = score.get("probe_runs", 0)
    bad: List[str] = []
    do_more: List[str] = []
    if runs == 0:
        bad.append("No discovery probes — Internal overnight may skip if cache fresh.")
    else:
        do_more.append(
            f"{runs} probe run(s); {score.get('features_approved', 0)} approved, "
            f"{score.get('features_rejected', 0)} rejected features."
        )
    lessons = f"Discovery (MC Internal IP): {runs} probe run(s) in window."
    return _base(lessons, bad, do_more)
