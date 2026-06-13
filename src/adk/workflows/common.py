"""Shared helpers for ADK Workflow daily pipelines."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src import config
from src.adk.schemas import TradingDecisionsResponse
from google.adk.workflow.utils._workflow_graph_utils import build_node
from src.agents.ledger_utils import (
    SIGNAL_JSON_PARSE_ATTEMPTS,
    SignalLedgerResult,
    is_malformed_json_error,
)
from src.agents.signal_context import fetch_signal_news
from src.logger import setup_logger
from src.models.trading_decision import TradingDecision

logger = setup_logger(__name__)


def parse_adk_signal_output(output: Any) -> SignalLedgerResult:
    """Normalize ADK task-agent output into SignalLedgerResult."""
    if output is None:
        return SignalLedgerResult(decisions=[], no_action_rationale="Empty signal output")

    if isinstance(output, SignalLedgerResult):
        return output

    if isinstance(output, TradingDecisionsResponse):
        decisions = [
            TradingDecision.from_dict(d.model_dump())
            for d in output.decisions
            if TradingDecision.from_dict(d.model_dump())
        ]
        return SignalLedgerResult(decisions=decisions, no_action_rationale="")

    if isinstance(output, dict):
        if "output" in output and len(output) == 1:
            return parse_adk_signal_output(output["output"])
        if "decisions" in output:
            try:
                parsed = TradingDecisionsResponse.model_validate(output)
                return parse_adk_signal_output(parsed)
            except Exception:
                pass
        raw = output.get("decisions", [])
        no_action = str(output.get("no_action_rationale") or "").strip()
        decisions: List[TradingDecision] = []
        valid = set(config.TICKER_UNIVERSE)
        for item in raw if isinstance(raw, list) else []:
            decision = TradingDecision.from_dict(item)
            if decision and decision.ticker in valid:
                decisions.append(decision)
        return SignalLedgerResult(decisions=decisions, no_action_rationale=no_action)

    if isinstance(output, str):
        from src.agents.ledger_utils import parse_signal_ledger_response

        return parse_signal_ledger_response(output, list(config.TICKER_UNIVERSE))

    return SignalLedgerResult(
        decisions=[],
        no_action_rationale=f"Unrecognized signal output type: {type(output).__name__}",
    )


async def invoke_signal_agent(ctx, system: str, payload: Dict[str, Any]) -> SignalLedgerResult:
    """Run ADK signal LlmAgent via ctx.run_node from a workflow function node."""
    if system == "baseline":
        from src.adk.agents.signal_agents import build_baseline_signal_agent

        agent = build_baseline_signal_agent()
    elif system == "internal":
        from src.adk.agents.signal_agents import build_internal_signal_agent

        agent = build_internal_signal_agent()
    else:
        raise ValueError(f"Invalid system: {system}")

    brief = {
        "request": (
            "Generate overnight Twin Ledger decisions from the JSON context below. "
            "Prefer the provided data; only call tools if required fields are missing.\n\n"
            + json.dumps(payload, default=str)
        )
    }
    wrapped = build_node(agent)
    logger.info(f"[ADK Workflow] Invoking {agent.name} via ctx.run_node")
    last_err: Exception | None = None
    for attempt in range(1, SIGNAL_JSON_PARSE_ATTEMPTS + 1):
        try:
            output = await ctx.run_node(wrapped, node_input=brief)
            return parse_adk_signal_output(output)
        except json.JSONDecodeError as exc:
            last_err = exc
            if attempt < SIGNAL_JSON_PARSE_ATTEMPTS:
                logger.warning(
                    f"[ADK Workflow] Signal JSON parse failed for {system} "
                    f"(attempt {attempt}/{SIGNAL_JSON_PARSE_ATTEMPTS}), retrying: {exc}"
                )
                continue
            raise
        except Exception as exc:
            if is_malformed_json_error(exc) and attempt < SIGNAL_JSON_PARSE_ATTEMPTS:
                last_err = exc
                logger.warning(
                    f"[ADK Workflow] Signal parse failed for {system} "
                    f"(attempt {attempt}/{SIGNAL_JSON_PARSE_ATTEMPTS}), retrying: {exc}"
                )
                continue
            raise
    if last_err:
        raise last_err
    return SignalLedgerResult(decisions=[], no_action_rationale="Empty signal output")


async def workflow_daily_setup(
    system: str,
    *,
    skip_calendar: bool = False,
) -> Optional[Dict[str, Any]]:
    """Hydrate audit, start trace, optional learning refresh. Returns skip dict if calendar blocks."""
    try:
        from src.gcs.store import get_gcs_store

        get_gcs_store().hydrate_audit_log()
    except Exception as e:
        logger.warning(f"GCS audit hydrate failed: {e}")

    from src.audit import start_trace

    dry = config.is_dry_run()
    start_trace(
        "daily",
        system=system,
        meta={
            "pipeline": "adk_workflow",
            "dry_run": dry,
            "skip_calendar": skip_calendar,
        },
    )
    logger.info(
        f"[ADK Workflow] Starting daily workflow for {system}"
        + (" (DRY RUN — no orders)" if dry else "")
    )

    from src.market.calendar import check_overnight_trading_session

    session_ok, session_reason = check_overnight_trading_session(
        system=system,
        skip_calendar=skip_calendar,
    )
    if not session_ok:
        from src.audit import end_trace

        logger.info(f"[ADK Workflow] Skipping overnight for {system}: {session_reason}")
        end_trace(
            "daily",
            system=system,
            success=True,
            summary={"skipped": True, "skip_reason": session_reason, "orders_placed": 0},
        )
        return {
            "success": True,
            "skipped": True,
            "skip_reason": session_reason,
            "pipeline": "adk_workflow",
            "system": system,
            "orders_placed": 0,
            "message": f"Overnight skipped: {session_reason}",
        }

    if config.LEARNING_ENABLED:
        try:
            from src.learning.reflection import refresh_system_learning

            await refresh_system_learning(system)
        except Exception as e:
            logger.warning(f"[ADK Workflow] Learning refresh failed: {e}")

    return None


def fetch_news_for_universe() -> Dict[str, Any]:
    return fetch_signal_news(list(config.TICKER_UNIVERSE)).get("news") or {}


def ledger_to_state(ledger: SignalLedgerResult) -> Dict[str, Any]:
    return {
        "decisions": [d.to_dict() for d in ledger.decisions],
        "no_action_rationale": ledger.no_action_rationale or "",
    }
